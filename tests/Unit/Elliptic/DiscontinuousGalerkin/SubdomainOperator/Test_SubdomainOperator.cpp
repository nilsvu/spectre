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
#include "Elliptic/Systems/Elasticity/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Poisson/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Poisson/Geometry.hpp"
#include "Elliptic/Tags.hpp"  // Needed by the numerical flux (for now)
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Elliptic/DiscontinuousGalerkin/TestHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/BoundaryFlux.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleBoundaryData.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Overloader.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/Tuple.hpp"

namespace helpers = TestHelpers::elliptic::dg;

namespace {

struct DummyOptionsGroup {};

template <typename System, typename FluxesArgsTags,
          typename FluxesArgsVolumeTags, typename SourcesArgsTags,
          typename SourcesArgsVolumeTags, typename PackageFluxesArgs,
          typename PackageSourcesArgs, typename... PrimalFields,
          typename... AuxiliaryFields>
void test_subdomain_operator_impl(
    const DomainCreator<System::volume_dim>& domain_creator,
    const size_t overlap, const double penalty_parameter,
    PackageFluxesArgs&& package_fluxes_args,
    PackageSourcesArgs&& package_sources_args,
    tmpl::list<PrimalFields...> /*meta*/,
    tmpl::list<AuxiliaryFields...> /*meta*/) noexcept {
  CAPTURE(overlap);
  CAPTURE(penalty_parameter);

  MAKE_GENERATOR(gen);
  UniformCustomDistribution<double> dist{-1., 1.};

  using system = System;
  static constexpr size_t volume_dim = system::volume_dim;
  const typename system::fluxes fluxes_computer{};

  // Get fluxes and sources arg types from the simple tags that are supplied for
  // the DataBox test below.
  // TODO: Perhaps the fluxes and sources arg types should be part of the system
  using FluxesArgs = tmpl::transform<FluxesArgsTags,
                                     tmpl::bind<db::const_item_type, tmpl::_1>>;
  using SourcesArgs =
      tmpl::transform<SourcesArgsTags,
                      tmpl::bind<db::const_item_type, tmpl::_1>>;

  // Shortcuts for tags
  using primal_fields = typename system::primal_fields;
  using auxiliary_fields = typename system::auxiliary_fields;
  using all_fields_tags =
      db::get_variables_tags_list<typename system::fields_tag>;
  using Vars = db::item_type<typename system::fields_tag>;
  using fluxes_computer_tag =
      elliptic::Tags::FluxesComputer<typename system::fluxes>;
  using fluxes_tags =
      db::wrap_tags_in<::Tags::Flux, all_fields_tags, tmpl::size_t<volume_dim>,
                       Frame::Inertial>;
  using div_fluxes_tags = db::wrap_tags_in<::Tags::div, fluxes_tags>;
  using n_dot_fluxes_tags =
      db::wrap_tags_in<::Tags::NormalDotFlux, all_fields_tags>;

  // Choose a numerical flux
  using NumericalFlux =
      elliptic::dg::NumericalFluxes::FirstOrderInternalPenalty<
          volume_dim, fluxes_computer_tag, primal_fields, auxiliary_fields>;
  const NumericalFlux numerical_fluxes_computer{penalty_parameter};  // C=1.5

  // Setup the elements in the domain
  const auto elements = helpers::create_elements(domain_creator);

  // Choose a subdomain that has internal and external faces
  const auto& subdomain_center = elements.begin()->first;
  CAPTURE(subdomain_center);
  const auto& central_element = elements.at(subdomain_center);

  // Setup the faces and mortars in the subdomain
  // TODO: Support h-refinement in this test
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

  // Create workspace vars for each element. Fill the operand with random values
  // within the subdomain and with zeros outside.
  std::unordered_map<ElementId<volume_dim>, Vars> workspace{};
  for (const auto& id_and_element : elements) {
    const auto& element_id = id_and_element.first;
    const size_t num_points =
        id_and_element.second.mesh.number_of_grid_points();
    if (element_id == subdomain_center) {
      workspace[element_id] = make_with_random_values<Vars>(
          make_not_null(&gen), make_not_null(&dist), DataVector{num_points});
    } else {
      workspace[element_id] = Vars{num_points, 0.};
    }
  }
  // Above we only filled the central element with random values. Now do the
  // same for the regions where the subdomain overlaps with neighbors.
  using SubdomainDataType = LinearSolver::schwarz_detail::SubdomainData<
      volume_dim, Variables<all_fields_tags>,
      elliptic::dg::SubdomainOperator_detail::OverlapData<
          volume_dim, all_fields_tags, FluxesArgs, SourcesArgs>>;
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
      const auto overlap_vars = make_with_random_values<Vars>(
          make_not_null(&gen), make_not_null(&dist),
          DataVector{overlap_extents.product()});
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
          std::move(perpendicular_mortar_sizes),
          package_fluxes_args(neighbor_id, neighbor),
          package_sources_args(neighbor_id, neighbor)};
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
          const Mesh<volume_dim - 1>& face_mesh,
          const tnsr::i<DataVector, volume_dim>& face_normal,
          const Variables<n_dot_fluxes_tags>& n_dot_fluxes,
          const Variables<div_fluxes_tags>& div_fluxes,
          const auto& fluxes_args) -> BoundaryData {
    return std::apply(
        [&numerical_fluxes_computer, &face_mesh, &n_dot_fluxes, &div_fluxes,
         &face_normal, &fluxes_computer](const auto&... expanded_fluxes_args) {
          return ::dg::FirstOrderScheme::package_boundary_data(
              numerical_fluxes_computer, face_mesh, n_dot_fluxes,
              get<::Tags::NormalDotFlux<AuxiliaryFields>>(n_dot_fluxes)...,
              get<::Tags::div<::Tags::Flux<
                  AuxiliaryFields, tmpl::size_t<volume_dim>, Frame::Inertial>>>(
                  div_fluxes)...,
              face_normal, fluxes_computer, expanded_fluxes_args...);
        },
        fluxes_args);
  };
  const auto apply_boundary_contribution =
      [&numerical_fluxes_computer](
          const auto result, const BoundaryData& local_boundary_data,
          const BoundaryData& remote_boundary_data,
          const Scalar<DataVector>& magnitude_of_face_normal,
          const Mesh<volume_dim>& mesh,
          const ::dg::MortarId<volume_dim>& mortar_id,
          const Mesh<volume_dim - 1>& mortar_mesh,
          const ::dg::MortarSize<volume_dim - 1>& mortar_size) {
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
            package_fluxes_args, package_sources_args, package_boundary_data,
            apply_boundary_contribution);
  }

  // (2) Apply the subdomain operator to the restricted data (as opposed to
  // applying the full DG operator to the full data and then restricting)
  const SubdomainDataType subdomain_operand{workspace.at(subdomain_center),
                                            std::move(subdomain_boundary_data)};
  const size_t center_num_points = central_element.mesh.number_of_grid_points();
  Variables<fluxes_tags> central_fluxes_buffer{center_num_points};
  Variables<div_fluxes_tags> central_div_fluxes_buffer{center_num_points};
  SubdomainDataType subdomain_result{center_num_points};
  elliptic::dg::apply_subdomain_center_volume<typename system::primal_fields,
                                              typename system::auxiliary_fields,
                                              typename system::sources>(
      make_not_null(&subdomain_result.element_data),
      make_not_null(&central_fluxes_buffer),
      make_not_null(&central_div_fluxes_buffer), fluxes_computer,
      central_element.mesh, central_element.inv_jacobian,
      package_fluxes_args(subdomain_center, central_element),
      package_sources_args(subdomain_center, central_element),
      subdomain_operand.element_data);
  for (const auto& direction_and_face_normal : internal_face_normals) {
    const auto& direction = direction_and_face_normal.first;
    elliptic::dg::apply_subdomain_face<typename system::primal_fields,
                                       typename system::auxiliary_fields,
                                       typename system::sources>(
        make_not_null(&subdomain_result), central_element.mesh, fluxes_computer,
        numerical_fluxes_computer, direction, direction_and_face_normal.second,
        internal_face_normal_magnitudes.at(direction), central_mortar_meshes,
        central_mortar_sizes,
        package_fluxes_args(subdomain_center, central_element, direction),
        subdomain_operand, central_fluxes_buffer, central_div_fluxes_buffer);
  }
  for (const auto& direction_and_face_normal : boundary_face_normals) {
    const auto& direction = direction_and_face_normal.first;
    elliptic::dg::apply_subdomain_face<typename system::primal_fields,
                                       typename system::auxiliary_fields,
                                       typename system::sources>(
        make_not_null(&subdomain_result), central_element.mesh, fluxes_computer,
        numerical_fluxes_computer, direction, direction_and_face_normal.second,
        boundary_face_normal_magnitudes.at(direction), central_mortar_meshes,
        central_mortar_sizes,
        package_fluxes_args(subdomain_center, central_element, direction),
        subdomain_operand, central_fluxes_buffer, central_div_fluxes_buffer);
  }

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
  using SubdomainOperator = elliptic::dg::SubdomainOperator<
      volume_dim, typename system::primal_fields,
      typename system::auxiliary_fields, fluxes_computer_tag, FluxesArgs,
      typename system::sources, SourcesArgs, numerical_flux_tag,
      DummyOptionsGroup>;
  SubdomainOperator subdomain_operator{center_num_points};
  auto initial_box = db::create<
      db::AddSimpleTags<
          domain::Tags::Element<volume_dim>, domain::Tags::Mesh<volume_dim>,
          domain::Tags::InverseJacobian<volume_dim, Frame::Logical,
                                        Frame::Inertial>,
          fluxes_computer_tag, numerical_flux_tag,
          domain::Tags::Interface<
              domain::Tags::InternalDirections<volume_dim>,
              ::Tags::Normalized<
                  domain::Tags::UnnormalizedFaceNormal<volume_dim>>>,
          domain::Tags::Interface<
              domain::Tags::BoundaryDirectionsInterior<volume_dim>,
              ::Tags::Normalized<
                  domain::Tags::UnnormalizedFaceNormal<volume_dim>>>,
          domain::Tags::Interface<
              domain::Tags::InternalDirections<volume_dim>,
              ::Tags::Magnitude<
                  domain::Tags::UnnormalizedFaceNormal<volume_dim>>>,
          domain::Tags::Interface<
              domain::Tags::BoundaryDirectionsInterior<volume_dim>,
              ::Tags::Magnitude<
                  domain::Tags::UnnormalizedFaceNormal<volume_dim>>>,
          ::Tags::Mortars<domain::Tags::Mesh<volume_dim - 1>, volume_dim>,
          ::Tags::Mortars<::Tags::MortarSize<volume_dim - 1>, volume_dim>>,
      db::AddComputeTags<
          domain::Tags::InternalDirections<volume_dim>,
          domain::Tags::BoundaryDirectionsInterior<volume_dim>,
          domain::Tags::InterfaceCompute<
              domain::Tags::InternalDirections<volume_dim>,
              domain::Tags::Direction<volume_dim>>,
          domain::Tags::InterfaceCompute<
              domain::Tags::BoundaryDirectionsInterior<volume_dim>,
              domain::Tags::Direction<volume_dim>>>>(
      central_element.element, central_element.mesh,
      central_element.inv_jacobian, fluxes_computer, numerical_fluxes_computer,
      internal_face_normals, boundary_face_normals,
      internal_face_normal_magnitudes, boundary_face_normal_magnitudes,
      central_mortar_meshes, central_mortar_sizes);
  auto box_with_fluxes_args = std::apply(
      [&initial_box](const auto&... expanded_fluxes_args) {
        return db::create_from<db::RemoveTags<>, FluxesArgsTags>(
            std::move(initial_box), expanded_fluxes_args...);
      },
      package_fluxes_args(subdomain_center, central_element));
  //   using fluxes_args_interface_tags =
  //       tmpl::list_difference<FluxesArgsTags, FluxesArgsVolumeTags>;
  // //   tuple_from_typelist<tmpl::transform<
  // //       tmpl::transform<fluxes_args_interface_tags,
  // //                       tmpl::bind<db::item_type, tmpl::_1>>,
  // //       tmpl::bind<std::unordered_map, Direction<volume_dim>, tmpl::_1>>>
  // //       fluxes_interface_args{};
  //   tuples::tagged_tuple_from_typelist<tmpl::transform<
  //       fluxes_args_interface_tags,
  //       tmpl::bind<std::unordered_map, Direction<volume_dim>, tmpl::_1>>>
  //       fluxes_interface_args{};
  //   for (const auto& direction :
  //        get<domain::Tags::InternalDirections<volume_dim>>(
  //            box_with_fluxes_args)) {
  //     auto this_direction_args = std::apply(
  //         [](const auto&... expanded_fluxes_interface_args) {
  //           return tuples::tagged_tuple_from_typelist<FluxesArgsTags>(
  //               expanded_fluxes_interface_args...);
  //         },
  //         package_fluxes_args(subdomain_center, central_element, direction));
  //     get<I>(fluxes_interface_args)[direction] = get<I +
  //     J>(this_direction_args);
  //   }
  //   auto box_with_fluxes_interface_args = std::apply(
  //       [&box_with_fluxes_args](const auto&...
  //       expanded_fluxes_interface_args) {
  //         return db::create_from<
  //             db::RemoveTags<>,
  //             tmpl::transform<
  //                 fluxes_args_interface_tags,
  //                 tmpl::bind<domain::Tags::Interface,
  //                            domain::Tags::InternalDirections<volume_dim>,
  //                            tmpl::_1>>>(std::move(box_with_fluxes_args),
  //                                        expanded_fluxes_interface_args...);
  //       },
  //       fluxes_interface_args);
  auto box_for_operator = std::apply(
      [&box_with_fluxes_args](const auto&... expanded_sources_args) {
        return db::create_from<db::RemoveTags<>, SourcesArgsTags>(
            std::move(box_with_fluxes_args), expanded_sources_args...);
      },
      package_sources_args(subdomain_center, central_element));

  db::apply<typename SubdomainOperator::volume_operator>(
      box_for_operator, subdomain_operand, make_not_null(&subdomain_operator));
  interface_apply<domain::Tags::InternalDirections<volume_dim>,
                  typename SubdomainOperator::face_operator::argument_tags,
                  get_volume_tags<typename SubdomainOperator::face_operator>>(
      typename SubdomainOperator::face_operator{}, box_for_operator,
      subdomain_operand, make_not_null(&subdomain_operator));
  interface_apply<domain::Tags::BoundaryDirectionsInterior<volume_dim>,
                  typename SubdomainOperator::face_operator::argument_tags,
                  get_volume_tags<typename SubdomainOperator::face_operator>>(
      typename SubdomainOperator::face_operator{}, box_for_operator,
      subdomain_operand, make_not_null(&subdomain_operator));
  const auto& subdomain_result_db = subdomain_operator.result();
  // TODO: Check full equivalence
  CHECK(subdomain_result_db.element_data == subdomain_result.element_data);
  // CHECK(subdomain_result_db.boundary == subdomain_result.element_data);

  // (5) Check collecting the overlap data from the DataBox
  const auto overlap_box = db::create_from<
      db::RemoveTags<>,
      db::AddSimpleTags<
          domain::Tags::ElementMap<volume_dim>,
          LinearSolver::schwarz_detail::Tags::Overlap<DummyOptionsGroup>>>(
      std::move(box_for_operator), central_element.element_map, size_t{2});
  const auto collected_boundary_data = interface_apply<
      domain::Tags::InternalDirections<volume_dim>,
      typename SubdomainOperator::collect_overlap_data::argument_tags,
      typename SubdomainOperator::collect_overlap_data::volume_tags>(
      typename SubdomainOperator::collect_overlap_data{}, overlap_box,
      workspace.at(subdomain_center));
  // TODO: Check full equivalence
  CHECK(collected_boundary_data.size() == subdomain_boundary_data.size());
}

template <typename System, typename FluxesArgsTags,
          typename FluxesArgsVolumeTags, typename SourcesArgsTags,
          typename SourcesArgsVolumeTags, typename... Args>
void test_subdomain_operator(Args&&... args) noexcept {
  test_subdomain_operator_impl<System, FluxesArgsTags, FluxesArgsVolumeTags,
                               SourcesArgsTags, SourcesArgsVolumeTags>(
      std::forward<Args>(args)..., typename System::primal_fields{},
      typename System::auxiliary_fields{});
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.DG.SubdomainOperator", "[Unit][Elliptic]") {
  {
    INFO("Poisson 1D");
    using system = Poisson::FirstOrderSystem<1, Poisson::Geometry::Euclidean>;
    const domain::creators::Interval domain_creator{
        {{-2.}}, {{2.}}, {{false}}, {{1}}, {{3}}};
    for (size_t overlap = 1; overlap <= 4; overlap++) {
      test_subdomain_operator<system, tmpl::list<>, tmpl::list<>, tmpl::list<>,
                              tmpl::list<>>(
          domain_creator, overlap, 6.75,
          [](const auto&... /*unused*/) { return std::tuple<>{}; },
          [](const auto&... /*unused*/) { return std::tuple<>{}; });
    }
  }
  {
    INFO("Poisson 2D");
    using system = Poisson::FirstOrderSystem<2, Poisson::Geometry::Euclidean>;
    const domain::creators::Rectangle domain_creator{
        {{-2., 0.}}, {{2., 1.}}, {{false, false}}, {{1, 1}}, {{3, 3}}};
    for (size_t overlap = 1; overlap <= 4; overlap++) {
      test_subdomain_operator<system, tmpl::list<>, tmpl::list<>, tmpl::list<>,
                              tmpl::list<>>(
          domain_creator, overlap, 6.75,
          [](const auto&... /*unused*/) { return std::tuple<>{}; },
          [](const auto&... /*unused*/) { return std::tuple<>{}; });
    }
  }
  {
    INFO("Poisson 3D");
    using system = Poisson::FirstOrderSystem<3, Poisson::Geometry::Euclidean>;
    const domain::creators::Brick domain_creator{{{-2., 0., -1.}},
                                                 {{2., 1., 1.}},
                                                 {{false, false, false}},
                                                 {{1, 1, 1}},
                                                 {{3, 3, 3}}};
    for (size_t overlap = 1; overlap <= 4; overlap++) {
      test_subdomain_operator<system, tmpl::list<>, tmpl::list<>, tmpl::list<>,
                              tmpl::list<>>(
          domain_creator, overlap, 6.75,
          [](const auto&... /*unused*/) { return std::tuple<>{}; },
          [](const auto&... /*unused*/) { return std::tuple<>{}; });
    }
  }
  {
    using system = Elasticity::FirstOrderSystem<3>;
    using ConstitutiveRelationType =
        Elasticity::ConstitutiveRelations::IsotropicHomogeneous<3>;
    ConstitutiveRelationType constitutive_relation{1., 2.};
    const domain::creators::Brick domain_creator{{{-2., 0., -1.}},
                                                 {{2., 1., 1.}},
                                                 {{false, false, false}},
                                                 {{1, 1, 1}},
                                                 {{3, 3, 3}}};
    for (size_t overlap = 1; overlap <= 4; overlap++) {
      test_subdomain_operator<
          system,
          tmpl::list<::Elasticity::Tags::ConstitutiveRelation<
                         ConstitutiveRelationType>,
                     ::domain::Tags::Coordinates<3, Frame::Inertial>>,
          tmpl::list<::Elasticity::Tags::ConstitutiveRelation<
              ConstitutiveRelationType>>,
          tmpl::list<>, tmpl::list<>>(
          domain_creator, overlap, 6.75,
          make_overloader(
              [&constitutive_relation](
                  const ElementId<3>& /*element_id*/,
                  const helpers::DgElement<3>& dg_element) {
                return std::make_tuple(
                    constitutive_relation,
                    dg_element.element_map(
                        logical_coordinates(dg_element.mesh)));
              },
              [&constitutive_relation](const ElementId<3>& /*element_id*/,
                                       const helpers::DgElement<3>& dg_element,
                                       const Direction<3>& direction) {
                return std::make_tuple(
                    constitutive_relation,
                    dg_element.element_map(interface_logical_coordinates(
                        dg_element.mesh.slice_away(direction.dimension()),
                        direction)));
              }),
          [](const auto&... /*unused*/) { return std::tuple<>{}; });
    }
  }
}
