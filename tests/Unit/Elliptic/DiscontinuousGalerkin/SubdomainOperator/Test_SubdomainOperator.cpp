// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/Rectangle.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/InterfaceHelpers.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/DiscontinuousGalerkin/NumericalFluxes/FirstOrderInternalPenalty.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/SubdomainOperator.hpp"
#include "Elliptic/Systems/Poisson/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Poisson/Geometry.hpp"
#include "Elliptic/Tags.hpp"  // Needed by the numerical flux (for now)
#include "Framework/TestHelpers.hpp"
#include "Helpers/Elliptic/DiscontinuousGalerkin/TestHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/BoundaryFlux.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleBoundaryData.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace helpers = TestHelpers::elliptic::dg;

namespace {

struct DummyOptionsGroup {};

template <size_t Dim>
void test_subdomain_operator(const DomainCreator<Dim>& domain_creator,
                             const size_t overlap) noexcept {
  CAPTURE(Dim);
  CAPTURE(overlap);
  static constexpr size_t volume_dim = Dim;

  // Choose a system
  using system =
      Poisson::FirstOrderSystem<volume_dim, Poisson::Geometry::Euclidean>;
  using Vars = db::item_type<typename system::fields_tag>;
  const typename system::fluxes fluxes_computer{};
  using fluxes_computer_tag =
      elliptic::Tags::FluxesComputer<typename system::fluxes>;

  // Choose a numerical flux
  using NumericalFlux =
      elliptic::dg::NumericalFluxes::FirstOrderInternalPenalty<
          volume_dim, fluxes_computer_tag, typename system::primal_fields,
          typename system::auxiliary_fields>;
  const NumericalFlux numerical_fluxes_computer{6.75};  // C=1.5

  // Shortcuts for tags
  using field_tag = Poisson::Tags::Field;
  using field_gradient_tag =
      ::Tags::deriv<field_tag, tmpl::size_t<volume_dim>, Frame::Inertial>;
  using all_fields_tags =
      db::get_variables_tags_list<typename system::fields_tag>;
  using fluxes_tags =
      db::wrap_tags_in<::Tags::Flux, all_fields_tags, tmpl::size_t<volume_dim>,
                       Frame::Inertial>;
  using div_fluxes_tags = db::wrap_tags_in<::Tags::div, fluxes_tags>;
  using n_dot_fluxes_tags =
      db::wrap_tags_in<::Tags::NormalDotFlux, all_fields_tags>;

  // Setup the elements in the domain
  const auto elements = helpers::create_elements(domain_creator);

  // Choose a subdomain that has internal and external faces
  const auto& subdomain_center = elements.begin()->first;
  CAPTURE(subdomain_center);
  const auto& central_element = elements.at(subdomain_center);

  // Setup the faces and mortars in the subdomain
  const auto central_mortars =
      helpers::create_mortars(subdomain_center, elements);
  std::unordered_map<Direction<volume_dim>, tnsr::i<DataVector, volume_dim>>
      internal_face_normals;
  std::unordered_map<Direction<volume_dim>, tnsr::i<DataVector, volume_dim>>
      boundary_face_normals;
  std::unordered_map<Direction<volume_dim>, Scalar<DataVector>>
      internal_face_normal_magnitudes;
  std::unordered_map<Direction<volume_dim>, Scalar<DataVector>>
      boundary_face_normal_magnitudes;
  dg::MortarMap<volume_dim, Mesh<volume_dim - 1>> central_mortar_meshes;
  dg::MortarMap<volume_dim, dg::MortarSize<volume_dim - 1>>
      central_mortar_sizes;
  for (const auto& mortar : central_mortars) {
    const auto& mortar_id = mortar.first;
    const auto& mortar_mesh_and_size = mortar.second;
    const auto& direction = mortar_id.first;
    const size_t dimension = direction.dimension();
    auto face_normal =
        unnormalized_face_normal(central_element.mesh.slice_away(dimension),
                                 central_element.element_map, direction);
    // TODO: Use system's magnitude
    auto normal_magnitude = magnitude(face_normal);
    for (size_t d = 0; d < volume_dim; d++) {
      face_normal.get(d) /= get(normal_magnitude);
    }
    if (mortar_id.second == ElementId<volume_dim>::external_boundary_id()) {
      boundary_face_normals[direction] = std::move(face_normal);
      boundary_face_normal_magnitudes[direction] = std::move(normal_magnitude);
    } else {
      internal_face_normals[direction] = std::move(face_normal);
      internal_face_normal_magnitudes[direction] = std::move(normal_magnitude);
    }
    central_mortar_meshes[mortar_id] = mortar_mesh_and_size.first;
    central_mortar_sizes[mortar_id] = mortar_mesh_and_size.second;
  }

  // Create workspace vars for each element. Fill the operand with zeros
  // everywhere but within the subdomain.
  std::unordered_map<ElementId<volume_dim>, Vars> workspace{};
  for (const auto& id_and_element : elements) {
    const auto& element_id = id_and_element.first;
    const size_t num_points =
        id_and_element.second.mesh.number_of_grid_points();
    Vars element_data{num_points, element_id == subdomain_center ? 1. : 0.};
    workspace[element_id] = std::move(element_data);
  }
  // Above we only filled the central element with ones. Now do the same for the
  // overlaps with its neighbors.
  using SubdomainDataType = LinearSolver::schwarz_detail::SubdomainData<
      Dim, Variables<all_fields_tags>,
      elliptic::dg::SubdomainOperator_detail::OverlapData<Dim,
                                                          all_fields_tags>>;
  typename SubdomainDataType::BoundaryDataType subdomain_boundary_data{};
  for (const auto& direction_and_neighbors :
       central_element.element.neighbors()) {
    const auto& direction = direction_and_neighbors.first;
    // const size_t dimension = direction.dimension();
    const auto& neighbors = direction_and_neighbors.second;
    const auto& orientation = neighbors.orientation();
    const auto direction_from_neighbor = orientation(direction.opposite());
    const size_t dimension_in_neighbor = direction_from_neighbor.dimension();
    for (const auto& neighbor_id : neighbors) {
      const dg::MortarId<volume_dim> mortar_id{direction, neighbor_id};
      const auto& neighbor = elements.at(neighbor_id);
      const auto overlap_extents =
          LinearSolver::schwarz_detail::overlap_extents(
              neighbor.mesh.extents(), overlap, dimension_in_neighbor);
      const Vars overlap_vars{overlap_extents.product(), 1.};
      workspace[neighbor_id] =
          LinearSolver::schwarz_detail::extended_overlap_data(
              overlap_vars, neighbor.mesh.extents(), overlap_extents,
              direction_from_neighbor);

      // Setup neighbor mortars perpendicular to the overlap direction
      const auto perpendicular_neighbor_mortars =
          LinearSolver::schwarz_detail::perpendicular(
              helpers::create_mortars(neighbor_id, elements), direction);
      ::dg::MortarMap<volume_dim, Mesh<volume_dim - 1>>
          perpendicular_mortar_meshes{};
      ::dg::MortarMap<volume_dim, ::dg::MortarSize<volume_dim - 1>>
          perpendicular_mortar_sizes{};
      std::transform(perpendicular_neighbor_mortars.begin(),
                     perpendicular_neighbor_mortars.end(),
                     std::inserter(perpendicular_mortar_meshes,
                                   perpendicular_mortar_meshes.end()),
                     [](auto const& id_and_mortar) {
                       return std::make_pair(id_and_mortar.first,
                                             id_and_mortar.second.first);
                     });
      std::transform(perpendicular_neighbor_mortars.begin(),
                     perpendicular_neighbor_mortars.end(),
                     std::inserter(perpendicular_mortar_sizes,
                                   perpendicular_mortar_sizes.end()),
                     [](auto const& id_and_mortar) {
                       return std::make_pair(id_and_mortar.first,
                                             id_and_mortar.second.second);
                     });

      // The overlap data should be oriented from the perspective of the central
      // element. We create it from the neighbor's perspective (because that's
      // the data we have available) and then re-orient.
      typename SubdomainDataType::overlap_data_type overlap_data{
          std::move(overlap_vars),
          neighbor.mesh,
          neighbor.element_map,
          direction_from_neighbor,
          overlap_extents,
          std::move(perpendicular_mortar_meshes),
          std::move(perpendicular_mortar_sizes)};
      overlap_data.orient(orientation.inverse_map());
      subdomain_boundary_data[mortar_id] = std::move(overlap_data);
    }
  }

  // (1) Apply the full DG operator
  // We use the StrongFirstOrder scheme, so we'll need the n.F on the boundaries
  // and the data needed by the numerical flux.
  using BoundaryData = ::dg::FirstOrderScheme::BoundaryData<NumericalFlux>;
  const auto package_boundary_data =
      [&numerical_fluxes_computer, &fluxes_computer](
          const Mesh<Dim - 1>& face_mesh,
          const tnsr::i<DataVector, Dim>& face_normal,
          const Variables<n_dot_fluxes_tags>& n_dot_fluxes,
          const Variables<div_fluxes_tags>& div_fluxes) -> BoundaryData {
    return ::dg::FirstOrderScheme::package_boundary_data(
        numerical_fluxes_computer, face_mesh, n_dot_fluxes,
        get<::Tags::NormalDotFlux<field_gradient_tag>>(n_dot_fluxes),
        get<::Tags::div<::Tags::Flux<field_gradient_tag, tmpl::size_t<Dim>,
                                     Frame::Inertial>>>(div_fluxes),
        face_normal, fluxes_computer);
  };
  const auto apply_boundary_contribution =
      [&numerical_fluxes_computer](
          const auto result, const BoundaryData& local_boundary_data,
          const BoundaryData& remote_boundary_data,
          const Scalar<DataVector>& magnitude_of_face_normal,
          const Mesh<Dim>& mesh, const ::dg::MortarId<Dim>& mortar_id,
          const Mesh<Dim - 1>& mortar_mesh,
          const ::dg::MortarSize<Dim - 1>& mortar_size) {
        const size_t dimension = mortar_id.first.dimension();
        auto boundary_contribution =
            std::decay_t<decltype(*result)>{dg::FirstOrderScheme::boundary_flux(
                local_boundary_data, remote_boundary_data,
                numerical_fluxes_computer, magnitude_of_face_normal,
                mesh.extents(dimension), mesh.slice_away(dimension),
                mortar_mesh, mortar_size)};
        add_slice_to_data(result, std::move(boundary_contribution),
                          mesh.extents(), dimension,
                          index_to_slice_at(mesh.extents(), mortar_id.first));
      };
  std::unordered_map<ElementId<volume_dim>, Vars>
      operator_applied_to_workspace{};
  for (const auto& id_and_element : elements) {
    const auto& element_id = id_and_element.first;
    operator_applied_to_workspace[element_id] =
        helpers::apply_first_order_dg_operator<system>(
            element_id, elements, workspace, fluxes_computer,
            [](const auto&... /* unused */) { return std::tuple<>{}; },
            [](const auto&... /* unused */) { return std::tuple<>{}; },
            package_boundary_data, apply_boundary_contribution);
  }

  // (2) Apply the subdomain operator to the restricted data (as opposed to
  // applying the full DG operator to the full data and then restricting)
  const SubdomainDataType subdomain_operand{workspace.at(subdomain_center),
                                            std::move(subdomain_boundary_data)};
  const auto subdomain_result =
      elliptic::dg::apply_subdomain_operator<typename system::primal_fields,
                                             typename system::auxiliary_fields,
                                             typename system::sources>(
          central_element.mesh, central_element.inv_jacobian, fluxes_computer,
          numerical_fluxes_computer, internal_face_normals,
          boundary_face_normals, internal_face_normal_magnitudes,
          boundary_face_normal_magnitudes, central_mortar_meshes,
          central_mortar_sizes, subdomain_operand);

  // (3) Check the subdomain operator is equivalent to the full DG operator
  // restricted to the subdomain
  CHECK_VARIABLES_APPROX(subdomain_result.element_data,
                         operator_applied_to_workspace.at(subdomain_center));
  CHECK(subdomain_result.boundary_data.size() ==
        subdomain_operand.boundary_data.size());
  for (const auto& mortar_id_and_overlap_result :
       subdomain_result.boundary_data) {
    const auto& mortar_id = mortar_id_and_overlap_result.first;
    auto overlap_result = mortar_id_and_overlap_result.second;
    const auto& direction = mortar_id.first;
    CAPTURE(direction);
    const auto& neighbor_id = mortar_id.second;
    CAPTURE(neighbor_id);
    const auto& neighbor = elements.at(neighbor_id);
    const auto& orientation =
        central_element.element.neighbors().at(direction).orientation();
    const auto& direction_from_neighbor = orientation(direction.opposite());
    overlap_result.orient(orientation);
    const auto overlap_result_from_workspace =
        LinearSolver::schwarz_detail::data_on_overlap(
            operator_applied_to_workspace.at(neighbor_id),
            neighbor.mesh.extents(), overlap_result.overlap_extents,
            direction_from_neighbor);
    CHECK_VARIABLES_APPROX(overlap_result.field_data,
                           overlap_result_from_workspace);
  }

  // (4) Check the subdomain operator works with the DataBox
  using numerical_flux_tag = ::Tags::NumericalFlux<NumericalFlux>;
  using subdomain_operator = elliptic::dg::SubdomainOperator<
      volume_dim, typename system::primal_fields,
      typename system::auxiliary_fields, fluxes_computer_tag,
      typename system::sources, numerical_flux_tag, DummyOptionsGroup>;
  auto box = db::create<db::AddSimpleTags<
      domain::Tags::Mesh<Dim>,
      domain::Tags::InverseJacobian<Dim, Frame::Logical, Frame::Inertial>,
      fluxes_computer_tag, numerical_flux_tag,
      domain::Tags::Interface<
          domain::Tags::InternalDirections<Dim>,
          ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>,
      domain::Tags::Interface<
          domain::Tags::BoundaryDirectionsInterior<Dim>,
          ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>,
      domain::Tags::Interface<
          domain::Tags::InternalDirections<Dim>,
          ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>>,
      domain::Tags::Interface<
          domain::Tags::BoundaryDirectionsInterior<Dim>,
          ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>>,
      ::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>,
      ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>>>(
      central_element.mesh, central_element.inv_jacobian, fluxes_computer,
      numerical_fluxes_computer, internal_face_normals, boundary_face_normals,
      internal_face_normal_magnitudes, boundary_face_normal_magnitudes,
      central_mortar_meshes, central_mortar_sizes);
  const auto subdomain_result_db =
      db::apply<subdomain_operator>(box, subdomain_operand);
  // TODO: Check full equivalence
  CHECK(subdomain_result_db.element_data == subdomain_result.element_data);
  // CHECK(subdomain_result_db.boundary == subdomain_result.element_data);

  // (5) Check collecting the overlap data from the DataBox
  const auto overlap_box = db::create_from<
      db::RemoveTags<>,
      db::AddSimpleTags<
          domain::Tags::ElementMap<Dim>, domain::Tags::Element<Dim>,
          LinearSolver::schwarz_detail::Tags::Overlap<DummyOptionsGroup>>,
      db::AddComputeTags<
          domain::Tags::InternalDirections<Dim>,
          domain::Tags::InterfaceCompute<domain::Tags::InternalDirections<Dim>,
                                         domain::Tags::Direction<Dim>>>>(
      std::move(box), central_element.element_map, central_element.element,
      size_t{2});
  const auto collected_boundary_data = interface_apply<
      domain::Tags::InternalDirections<Dim>,
      typename subdomain_operator::collect_overlap_data::argument_tags,
      typename subdomain_operator::collect_overlap_data::volume_tags>(
      typename subdomain_operator::collect_overlap_data{}, overlap_box,
      workspace.at(subdomain_center));
  // TODO: Check full equivalence
  CHECK(collected_boundary_data.size() == subdomain_boundary_data.size());
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.DG.SubdomainOperator", "[Unit][Elliptic]") {
  {
    const domain::creators::Interval domain_creator{
        {{-2.}}, {{2.}}, {{false}}, {{1}}, {{3}}};
    for (size_t overlap = 1; overlap <= 4; overlap++) {
      test_subdomain_operator(domain_creator, overlap);
    }
  }
  {
    const domain::creators::Rectangle domain_creator{
        {{-2., 0.}}, {{2., 1.}}, {{false, false}}, {{1, 1}}, {{3, 3}}};
    for (size_t overlap = 1; overlap <= 4; overlap++) {
      test_subdomain_operator(domain_creator, overlap);
    }
  }
  {
    const domain::creators::Brick domain_creator{{{-2., 0., -1.}},
                                                 {{2., 1., 1.}},
                                                 {{false, false, false}},
                                                 {{1, 1, 1}},
                                                 {{3, 3, 3}}};
    for (size_t overlap = 1; overlap <= 4; overlap++) {
      test_subdomain_operator(domain_creator, overlap);
    }
  }
}
