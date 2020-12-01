// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <boost/range/join.hpp>
#include <cstddef>

#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/Rectangle.hpp"
#include "Domain/Creators/RotatedBricks.hpp"
#include "Domain/Creators/RotatedIntervals.hpp"
#include "Domain/Creators/RotatedRectangles.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/InterfaceHelpers.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/DiscontinuousGalerkin/ImposeBoundaryConditions.hpp"
#include "Elliptic/DiscontinuousGalerkin/InitializeFirstOrderOperator.hpp"
#include "Elliptic/DiscontinuousGalerkin/NumericalFluxes/FirstOrderInternalPenalty.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/ApplyFace.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/InitializeSubdomain.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/SubdomainOperator.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/Tags.hpp"
#include "Elliptic/FirstOrderOperator.hpp"
#include "Elliptic/Systems/Elasticity/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Poisson/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Poisson/Geometry.hpp"
#include "Elliptic/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/BoundaryFlux.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/FirstOrderScheme.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/LiftFlux.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleBoundaryData.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/Actions/SetData.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/CollectDataForFluxes.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/FluxCommunication.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeDomain.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeInterfaces.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeMortars.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/ElementCenteredSubdomainData.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Protocols.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Overloader.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/Tuple.hpp"

namespace {

// The tests in this file check that the subdomain operator is equivalent to
// applying the full DG-operator to a domain where all points outside the
// subdomain are zero. This should hold for every element in the domain, at any
// refinement level (h and p) and for any number of overlap points.
//
// We use the action-testing framework for these tests because then we can have
// the InitializeSubdomain action do the tedious job of constructing the
// geometry. This has the added benefit that we test the subdomain operator is
// consistent with the InitializeSubdomain action.

struct DummyOptionsGroup {};

struct TemporalIdTag : db::SimpleTag {
  using type = size_t;
};

template <typename Tag>
struct DgOperatorAppliedTo : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};

template <typename Tag>
struct Operand : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};

template <size_t Dim, typename Fields>
struct SubdomainDataTag : db::SimpleTag {
  using type = LinearSolver::Schwarz::ElementCenteredSubdomainData<Dim, Fields>;
};

template <size_t Dim, typename Fields>
struct SubdomainOperatorAppliedToDataTag : db::SimpleTag {
  using type = LinearSolver::Schwarz::ElementCenteredSubdomainData<
      Dim, db::wrap_tags_in<DgOperatorAppliedTo, Fields>>;
};

// Generate some random element-centered subdomain data on each element
template <typename SubdomainOperator, typename Fields>
struct InitializeRandomSubdomainData {
  using simple_tags =
      tmpl::list<SubdomainDataTag<SubdomainOperator::volume_dim, Fields>>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    using SubdomainData = typename SubdomainDataTag<Dim, Fields>::type;

    db::mutate<SubdomainDataTag<Dim, Fields>>(
        make_not_null(&box),
        [](const auto subdomain_data, const auto& mesh, const auto& element,
           const auto& all_overlap_meshes, const auto& all_overlap_extents) {
          MAKE_GENERATOR(gen);
          UniformCustomDistribution<double> dist{-1., 1.};
          subdomain_data->element_data =
              make_with_random_values<typename SubdomainData::ElementData>(
                  make_not_null(&gen), make_not_null(&dist),
                  mesh.number_of_grid_points());
          for (const auto& [direction, neighbors] : element.neighbors()) {
            const auto& orientation = neighbors.orientation();
            for (const auto& neighbor_id : neighbors) {
              const auto overlap_id = std::make_pair(direction, neighbor_id);
              const size_t overlap_extent = all_overlap_extents.at(overlap_id);
              if (overlap_extent == 0) {
                continue;
              }
              subdomain_data->overlap_data.emplace(
                  overlap_id,
                  make_with_random_values<typename SubdomainData::OverlapData>(
                      make_not_null(&gen), make_not_null(&dist),
                      LinearSolver::Schwarz::overlap_num_points(
                          all_overlap_meshes.at(overlap_id).extents(),
                          overlap_extent, orientation(direction).dimension())));
            }
          }
        },
        db::get<domain::Tags::Mesh<Dim>>(box),
        db::get<domain::Tags::Element<Dim>>(box),
        db::get<LinearSolver::Schwarz::Tags::Overlaps<domain::Tags::Mesh<Dim>,
                                                      Dim, DummyOptionsGroup>>(
            box),
        db::get<LinearSolver::Schwarz::Tags::Overlaps<
            elliptic::dg::subdomain_operator::Tags::ExtrudingExtent, Dim,
            DummyOptionsGroup>>(box));
    return {std::move(box)};
  }
};

template <typename SubdomainOperator, typename Fields>
struct ApplySubdomainOperator {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& subdomain_data = db::get<SubdomainDataTag<Dim, Fields>>(box);

    // Apply the subdomain operator
    const auto& mesh = db::get<domain::Tags::Mesh<Dim>>(box);
    SubdomainOperator subdomain_operator{mesh.number_of_grid_points()};
    auto subdomain_result = make_with_value<
        typename SubdomainOperatorAppliedToDataTag<Dim, Fields>::type>(
        subdomain_data, 0.);
    db::apply<typename SubdomainOperator::element_operator>(
        box, subdomain_data, make_not_null(&subdomain_result),
        make_not_null(&subdomain_operator));
    using face_operator_internal =
        typename SubdomainOperator::template face_operator<
            domain::Tags::InternalDirections<Dim>>;
    using face_operator_external =
        typename SubdomainOperator::template face_operator<
            domain::Tags::BoundaryDirectionsInterior<Dim>>;
    interface_apply<domain::Tags::InternalDirections<Dim>,
                    face_operator_internal>(box, subdomain_data,
                                            make_not_null(&subdomain_result),
                                            make_not_null(&subdomain_operator));
    interface_apply<domain::Tags::BoundaryDirectionsInterior<Dim>,
                    face_operator_external>(box, subdomain_data,
                                            make_not_null(&subdomain_result),
                                            make_not_null(&subdomain_operator));

    // Store result in the DataBox for checks
    db::mutate<SubdomainOperatorAppliedToDataTag<Dim, Fields>>(
        make_not_null(&box),
        [&subdomain_result](const auto subdomain_operator_applied_to_data) {
          *subdomain_operator_applied_to_data = std::move(subdomain_result);
        });
    return {std::move(box)};
  }
};

template <typename Metavariables, typename System, typename BoundaryScheme,
          typename SubdomainOperator, typename FaceComputeTags,
          typename ExtraInitActions>
struct ElementArray {
  static constexpr size_t Dim = SubdomainOperator::volume_dim;
  using fields_tag = db::add_tag_prefix<Operand, typename System::fields_tag>;
  using fields = typename fields_tag::tags_list;
  using primal_fields =
      db::wrap_tags_in<Operand, typename System::primal_fields>;
  using auxiliary_fields =
      db::wrap_tags_in<Operand, typename System::auxiliary_fields>;

  using apply_full_dg_operator_actions = tmpl::list<
      ::dg::Actions::CollectDataForFluxes<
          BoundaryScheme, domain::Tags::InternalDirections<Dim>>,
      ::dg::Actions::SendDataForFluxes<BoundaryScheme>,
      ::Actions::MutateApply<
          elliptic::FirstOrderOperator<Dim, DgOperatorAppliedTo, fields_tag>>,
      elliptic::dg::Actions::ImposeHomogeneousDirichletBoundaryConditions<
          fields_tag, primal_fields>,
      ::dg::Actions::CollectDataForFluxes<
          BoundaryScheme, domain::Tags::BoundaryDirectionsInterior<Dim>>,
      ::dg::Actions::ReceiveDataForFluxes<BoundaryScheme>,
      Actions::MutateApply<BoundaryScheme>>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Dim>;
  using const_global_cache_tags = tmpl::list<domain::Tags::Domain<Dim>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<
              ActionTesting::InitializeDataBox<tmpl::list<
                  domain::Tags::InitialRefinementLevels<Dim>,
                  domain::Tags::InitialExtents<Dim>, fields_tag,
                  db::add_tag_prefix<DgOperatorAppliedTo, fields_tag>,
                  SubdomainOperatorAppliedToDataTag<Dim, fields>,
                  TemporalIdTag>>,
              Actions::SetupDataBox, ::dg::Actions::InitializeDomain<Dim>,
              ::dg::Actions::InitializeInterfaces<
                  System, ::dg::Initialization::slice_tags_to_face<>,
                  ::dg::Initialization::slice_tags_to_exterior<>,
                  FaceComputeTags,
                  ::dg::Initialization::exterior_compute_tags<>, false, false>,
              ::dg::Actions::InitializeMortars<BoundaryScheme, true>,
              elliptic::dg::Actions::InitializeFirstOrderOperator<
                  Dim, typename System::fluxes, typename System::sources,
                  fields_tag, primal_fields, auxiliary_fields>,
              elliptic::dg::Actions::InitializeSubdomain<Dim,
                                                         DummyOptionsGroup>,
              InitializeRandomSubdomainData<SubdomainOperator, fields>,
              ExtraInitActions, Parallel::Actions::TerminatePhase>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<apply_full_dg_operator_actions,
                     ApplySubdomainOperator<SubdomainOperator, fields>,
                     Parallel::Actions::TerminatePhase>>>;
};

template <typename System, typename BoundaryScheme, typename SubdomainOperator,
          typename FaceComputeTags, typename ExtraInitActions>
struct Metavariables {
  using element_array =
      ElementArray<Metavariables, System, BoundaryScheme, SubdomainOperator,
                   FaceComputeTags, ExtraInitActions>;
  using component_list = tmpl::list<element_array>;
  using const_global_cache_tags =
      tmpl::list<typename SubdomainOperator::fluxes_computer_tag,
                 typename SubdomainOperator::numerical_fluxes_computer_tag>;
  enum class Phase { Initialization, Testing, Exit };
};

// The test should work for any elliptic system. For systems with fluxes or
// sources that take arguments out of the DataBox this test can insert actions
// that initialize those arguments.
template <typename System, typename FaceComputeTags = tmpl::list<>,
          typename ExtraInitActions = tmpl::list<>,
          typename FluxesArgsTagsFromCenter = tmpl::list<>>
void test_subdomain_operator(
    const DomainCreator<System::volume_dim>& domain_creator,
    // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
    const size_t max_overlap = 3, const double penalty_parameter = 1.2) {
  constexpr size_t Dim = System::volume_dim;
  CAPTURE(Dim);
  CAPTURE(penalty_parameter);

  // We prefix the system fields with an "operand" tag to make sure the
  // subdomain operator works with prefixed variables
  using fields_tag = db::add_tag_prefix<Operand, typename System::fields_tag>;
  using fields = typename fields_tag::tags_list;
  using primal_fields =
      db::wrap_tags_in<Operand, typename System::primal_fields>;
  using auxiliary_fields =
      db::wrap_tags_in<Operand, typename System::auxiliary_fields>;
  using fluxes_computer_tag =
      elliptic::Tags::FluxesComputer<typename System::fluxes>;
  using NumericalFlux =
      elliptic::dg::NumericalFluxes::FirstOrderInternalPenalty<
          Dim, fluxes_computer_tag, primal_fields, auxiliary_fields>;
  using numerical_flux_tag = ::Tags::NumericalFlux<NumericalFlux>;
  const NumericalFlux numerical_fluxes_computer{penalty_parameter};
  using boundary_scheme = ::dg::FirstOrderScheme::FirstOrderScheme<
      Dim, fields_tag, db::add_tag_prefix<DgOperatorAppliedTo, fields_tag>,
      numerical_flux_tag, TemporalIdTag>;
  using SubdomainOperator = elliptic::dg::subdomain_operator::SubdomainOperator<
      Dim, primal_fields, auxiliary_fields, fluxes_computer_tag,
      typename System::sources, numerical_flux_tag, DummyOptionsGroup,
      FluxesArgsTagsFromCenter>;
  using metavariables =
      Metavariables<System, boundary_scheme, SubdomainOperator, FaceComputeTags,
                    ExtraInitActions>;
  using element_array = typename metavariables::element_array;

  // The test should hold for any number of overlap points
  for (size_t overlap = 0; overlap <= max_overlap; overlap++) {
    CAPTURE(overlap);

    // Have to re-create the domain in every iteration of this loop because it's
    // not copyable
    auto domain = domain_creator.create_domain();
    const auto initial_ref_levs = domain_creator.initial_refinement_levels();
    const auto initial_extents = domain_creator.initial_extents();
    const auto element_ids = ::initial_element_ids(initial_ref_levs);
    CAPTURE(element_ids.size());

    ActionTesting::MockRuntimeSystem<metavariables> runner{tuples::TaggedTuple<
        domain::Tags::Domain<Dim>,
        LinearSolver::Schwarz::Tags::MaxOverlap<DummyOptionsGroup>,
        fluxes_computer_tag, numerical_flux_tag>{std::move(domain), overlap,
                                                 typename System::fluxes{},
                                                 numerical_fluxes_computer}};

    // Initialize all elements, generating random subdomain data
    for (const auto& element_id : element_ids) {
      CAPTURE(element_id);
      ActionTesting::emplace_component_and_initialize<element_array>(
          &runner, element_id,
          {initial_ref_levs, initial_extents, typename fields_tag::type{},
           typename db::add_tag_prefix<DgOperatorAppliedTo, fields_tag>::type{},
           typename SubdomainOperatorAppliedToDataTag<Dim, fields>::type{},
           size_t{0}});
      while (
          not ActionTesting::get_terminate<element_array>(runner, element_id)) {
        ActionTesting::next_action<element_array>(make_not_null(&runner),
                                                  element_id);
      }
    }
    ActionTesting::set_phase(make_not_null(&runner),
                             metavariables::Phase::Testing);
    // DataBox shortcuts
    const auto get_tag = [&runner](const ElementId<Dim>& element_id,
                                   auto tag_v) -> decltype(auto) {
      using tag = std::decay_t<decltype(tag_v)>;
      return ActionTesting::get_databox_tag<element_array, tag>(runner,
                                                                element_id);
    };
    const auto set_tag = [&runner](const ElementId<Dim>& element_id, auto tag_v,
                                   const auto& value) {
      using tag = std::decay_t<decltype(tag_v)>;
      ActionTesting::simple_action<element_array,
                                   ::Actions::SetData<tmpl::list<tag>>>(
          make_not_null(&runner), element_id, value);
    };

    // Take each element as the subdomain-center in turn
    for (const auto& subdomain_center : element_ids) {
      CAPTURE(subdomain_center);

      // First, reset the data on all elements to zero
      for (const auto& element_id : element_ids) {
        set_tag(element_id, fields_tag{},
                typename fields_tag::type{
                    get_tag(element_id, domain::Tags::Mesh<Dim>{})
                        .number_of_grid_points(),
                    0.});
      }

      // Set data on the central element and its neighbors to the subdomain data
      const auto& subdomain_data =
          get_tag(subdomain_center, SubdomainDataTag<Dim, fields>{});
      const auto& all_overlap_extents =
          get_tag(subdomain_center,
                  LinearSolver::Schwarz::Tags::Overlaps<
                      elliptic::dg::subdomain_operator::Tags::ExtrudingExtent,
                      Dim, DummyOptionsGroup>{});
      const auto& central_element =
          get_tag(subdomain_center, domain::Tags::Element<Dim>{});
      set_tag(subdomain_center, fields_tag{}, subdomain_data.element_data);
      for (const auto& [overlap_id, overlap_data] :
           subdomain_data.overlap_data) {
        const auto& [direction, neighbor_id] = overlap_id;
        const auto direction_from_neighbor =
            central_element.neighbors().at(direction).orientation()(
                direction.opposite());
        set_tag(
            neighbor_id, fields_tag{},
            LinearSolver::Schwarz::extended_overlap_data(
                overlap_data,
                get_tag(neighbor_id, domain::Tags::Mesh<Dim>{}).extents(),
                all_overlap_extents.at(overlap_id), direction_from_neighbor));
      }

      // Run actions to compute the full DG-operator
      for (const auto& element_id : element_ids) {
        CAPTURE(element_id);
        runner.template mock_distributed_objects<element_array>()
            .at(element_id)
            .force_next_action_to_be(0);
        for (size_t i = 0; i < 5; ++i) {
          ActionTesting::next_action<element_array>(make_not_null(&runner),
                                                    element_id);
        }
      }
      // Break here so all elements have sent mortar data before receiving it
      for (const auto& element_id : element_ids) {
        CAPTURE(element_id);
        for (size_t i = 0; i < 2; ++i) {
          ActionTesting::next_action<element_array>(make_not_null(&runner),
                                                    element_id);
        }
      }

      // Invoke ApplySubdomainOperator action only on the subdomain center
      ActionTesting::next_action<element_array>(make_not_null(&runner),
                                                subdomain_center);

      // Test that the subdomain operator and the full DG-operator computed the
      // same result within the subdomain
      const auto& subdomain_result = get_tag(
          subdomain_center, SubdomainOperatorAppliedToDataTag<Dim, fields>{});
      using dg_operator_applied_to_fields_tag =
          db::add_tag_prefix<DgOperatorAppliedTo, fields_tag>;
      Approx custom_approx = Approx::custom().epsilon(1.e-12).scale(1.0);
      CHECK_VARIABLES_CUSTOM_APPROX(
          subdomain_result.element_data,
          get_tag(subdomain_center, dg_operator_applied_to_fields_tag{}),
          custom_approx);
      REQUIRE(subdomain_result.overlap_data.size() ==
              subdomain_data.overlap_data.size());
      for (const auto& [overlap_id, overlap_result] :
           subdomain_result.overlap_data) {
        CAPTURE(overlap_id);
        const auto& [direction, neighbor_id] = overlap_id;
        const auto direction_from_neighbor =
            central_element.neighbors().at(direction).orientation()(
                direction.opposite());
        const auto expected_overlap_result =
            LinearSolver::Schwarz::data_on_overlap(
                get_tag(neighbor_id, dg_operator_applied_to_fields_tag{}),
                get_tag(neighbor_id, domain::Tags::Mesh<Dim>{}).extents(),
                all_overlap_extents.at(overlap_id), direction_from_neighbor);
        CHECK_VARIABLES_CUSTOM_APPROX(overlap_result, expected_overlap_result,
                                      custom_approx);
      }
    }  // loop over subdomain centers
  }    // loop over overlaps
}

// Add a constitutive relation for elasticity systems to the DataBox
template <size_t Dim>
struct InitializeConstitutiveRelation {
 private:
  using ConstitutiveRelationType =
      Elasticity::ConstitutiveRelations::IsotropicHomogeneous<Dim>;

 public:
  using simple_tags = tmpl::list<
      Elasticity::Tags::ConstitutiveRelation<ConstitutiveRelationType>>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    ::Initialization::mutate_assign<simple_tags>(
        make_not_null(&box), ConstitutiveRelationType{1., 2.});
    return {std::move(box)};
  }
};

// Initialize data on overlaps needed for the elasticity system
template <size_t Dim>
struct InitializeElasticitySubdomain {
 private:
  template <typename Tag>
  using overlaps_tag =
      LinearSolver::Schwarz::Tags::Overlaps<Tag, Dim, DummyOptionsGroup>;
  template <typename ValueType>
  using overlaps = LinearSolver::Schwarz::OverlapMap<Dim, ValueType>;

 public:
  using simple_tags = db::wrap_tags_in<
      overlaps_tag,
      tmpl::list<domain::Tags::Coordinates<Dim, Frame::Inertial>,
                 domain::Tags::Faces<
                     Dim, domain::Tags::Coordinates<Dim, Frame::Inertial>>>>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& overlap_meshes =
        db::get<overlaps_tag<domain::Tags::Mesh<Dim>>>(box);
    const auto& overlap_elements =
        db::get<overlaps_tag<domain::Tags::Element<Dim>>>(box);
    const auto& overlap_element_maps =
        db::get<overlaps_tag<domain::Tags::ElementMap<Dim>>>(box);

    overlaps<tnsr::I<DataVector, Dim>> overlap_inertial_coords{};
    overlaps<DirectionMap<Dim, tnsr::I<DataVector, Dim>>>
        overlap_boundary_inertial_coords{};

    for (const auto& [overlap_id, neighbor_element_map] :
         overlap_element_maps) {
      const auto& neighbor_mesh = overlap_meshes.at(overlap_id);
      const auto& neighbor = overlap_elements.at(overlap_id);

      // Coords on the overlapped neighbor
      overlap_inertial_coords.emplace(
          overlap_id, neighbor_element_map(logical_coordinates(neighbor_mesh)));
      const auto& neighbor_inertial_coords =
          overlap_inertial_coords.at(overlap_id);

      // Coords on the internal faces of the overlapped neighbor
      DirectionMap<Dim, tnsr::I<DataVector, Dim>>
          neighbor_boundary_inertial_coords{};
      for (const auto& direction_from_neighbor :
           boost::join(neighbor.internal_boundaries(),
                       neighbor.external_boundaries())) {
        neighbor_boundary_inertial_coords.emplace(
            direction_from_neighbor,
            data_on_slice(neighbor_inertial_coords, neighbor_mesh.extents(),
                          direction_from_neighbor.dimension(),
                          index_to_slice_at(neighbor_mesh.extents(),
                                            direction_from_neighbor)));
      }
      overlap_boundary_inertial_coords.emplace(
          overlap_id, std::move(neighbor_boundary_inertial_coords));
    }

    ::Initialization::mutate_assign<simple_tags>(
        make_not_null(&box), std::move(overlap_inertial_coords),
        std::move(overlap_boundary_inertial_coords));
    return {std::move(box)};
  }
};

}  // namespace

// This test constructs a selection of domains and tests the subdomain operator
// for _every_ element in those domains and for a range of overlaps. We increase
// the timeout for the test because going over so many elements is relatively
// expensive but also very important to ensure that the subdomain operator
// handles all of these geometries correctly.
// [[TimeOut, 25]]
SPECTRE_TEST_CASE("Unit.Elliptic.DG.SubdomainOperator", "[Unit][Elliptic]") {
  {
    INFO("Aligned elements");
    const domain::creators::Interval domain_creator_1d{
        {{-2.}}, {{2.}}, {{false}}, {{1}}, {{3}}};
    test_subdomain_operator<
        Poisson::FirstOrderSystem<1, Poisson::Geometry::Euclidean>>(
        domain_creator_1d);

    const domain::creators::Rectangle domain_creator_2d{
        {{-2., 0.}}, {{2., 1.}}, {{false, false}}, {{1, 1}}, {{3, 3}}};
    test_subdomain_operator<
        Poisson::FirstOrderSystem<2, Poisson::Geometry::Euclidean>>(
        domain_creator_2d);

    const domain::creators::Brick domain_creator_3d{{{-2., 0., -1.}},
                                                    {{2., 1., 1.}},
                                                    {{false, false, false}},
                                                    {{1, 1, 1}},
                                                    {{3, 3, 3}}};
    test_subdomain_operator<
        Poisson::FirstOrderSystem<3, Poisson::Geometry::Euclidean>>(
        domain_creator_3d);
  }
  {
    INFO("Rotated elements");
    const domain::creators::RotatedIntervals domain_creator_1d{
        {{-2.}}, {{0.}}, {{2.}}, {{false}}, {{0}}, {{{{3, 3}}}}};
    test_subdomain_operator<
        Poisson::FirstOrderSystem<1, Poisson::Geometry::Euclidean>>(
        domain_creator_1d);

    const domain::creators::RotatedRectangles domain_creator_2d{
        {{-2., 0.}},      {{0., 0.5}}, {{2., 1.}},
        {{false, false}}, {{0, 0}},    {{{{3, 3}}, {{3, 3}}}}};
    test_subdomain_operator<
        Poisson::FirstOrderSystem<2, Poisson::Geometry::Euclidean>>(
        domain_creator_2d);

    const domain::creators::RotatedBricks domain_creator_3d{
        {{-2., 0., -1.}}, {{0., 0.5, 0.}},
        {{2., 1., 1.}},   {{false, false, false}},
        {{1, 1, 1}},      {{{{3, 3}}, {{3, 3}}, {{3, 3}}}}};
    test_subdomain_operator<
        Poisson::FirstOrderSystem<3, Poisson::Geometry::Euclidean>>(
        domain_creator_3d);
  }
  {
    INFO("Refined elements");
    //  |-B0-|--B1---|
    //  [oooo|ooo|ooo]-> xi
    //  ^    ^   ^   ^
    // -2    0   1   2
    const domain::creators::AlignedLattice<1> domain_creator_1d{
        {{{-2., 0., 2.}}},
        {{false}},
        {{0}},
        {{3}},
        {{{{1}}, {{2}}, {{1}}}},  // Refine once in block 1
        {{{{0}}, {{1}}, {{4}}}},  // Increase num points in block 0
        {}};
    test_subdomain_operator<
        Poisson::FirstOrderSystem<1, Poisson::Geometry::Euclidean>>(
        domain_creator_1d);

    //   -2    0   2
    // -2 +----+---+> xi
    //    |oooo|ooo|
    //    |    |ooo|
    //    |    |ooo|
    // -1 |oooo+---+
    //    |    |ooo|
    //    |    |ooo|
    //    |oooo|ooo|
    //  0 +----+---+
    //    |ooo |ooo|
    //    |ooo |ooo|
    //    |ooo |ooo|
    //  2 +----+---+
    //    v eta
    const domain::creators::AlignedLattice<2> domain_creator_2d{
        // Start with 4 unrefined blocks
        {{{-2., 0., 2.}, {-2., 0., 2.}}},
        {{false, false}},
        {{0, 0}},
        {{3, 3}},
        // Refine once in eta in upper-right block in sketch above
        {{{{1, 0}}, {{2, 1}}, {{0, 1}}}},
        // Increase num points in xi in upper-left block in sketch above
        {{{{0, 0}}, {{1, 1}}, {{4, 3}}}},
        {}};
    test_subdomain_operator<
        Poisson::FirstOrderSystem<2, Poisson::Geometry::Euclidean>>(
        domain_creator_2d);

    const domain::creators::AlignedLattice<3> domain_creator_3d{
        {{{-2., 0., 2.}, {-2., 0., 2.}, {-2., 0., 2.}}},
        {{false, false, false}},
        {{0, 0, 0}},
        {{3, 3, 3}},
        {{{{1, 0, 0}}, {{2, 1, 1}}, {{0, 1, 1}}}},
        {{{{0, 0, 0}}, {{1, 1, 1}}, {{4, 3, 2}}}},
        {}};
    test_subdomain_operator<
        Poisson::FirstOrderSystem<3, Poisson::Geometry::Euclidean>>(
        domain_creator_3d);
  }
  {
    INFO("Curved elements");
    const domain::creators::Disk domain_creator_2d{0.5, 2., 1, {{3, 4}}, false};
    test_subdomain_operator<
        Poisson::FirstOrderSystem<2, Poisson::Geometry::Euclidean>>(
        domain_creator_2d);

    const domain::creators::Cylinder domain_creator_3d{
        0.5, 2., 0., 2., false, 0, {{3, 4, 2}}, false};
    test_subdomain_operator<
        Poisson::FirstOrderSystem<3, Poisson::Geometry::Euclidean>>(
        domain_creator_3d);
  }
  {
    INFO("System with fluxes args");
    using system = Elasticity::FirstOrderSystem<3>;
    const domain::creators::Brick domain_creator{{{-2., 0., -1.}},
                                                 {{2., 1., 1.}},
                                                 {{false, false, false}},
                                                 {{1, 1, 1}},
                                                 {{3, 3, 3}}};
    test_subdomain_operator<
        system, tmpl::list<domain::Tags::BoundaryCoordinates<3>>,
        tmpl::list<InitializeConstitutiveRelation<3>,
                   InitializeElasticitySubdomain<3>>,
        tmpl::list<::Elasticity::Tags::ConstitutiveRelationBase>>(
        domain_creator);
  }
  {
      INFO("Test that unmapping args doesn't copy");
      const ConstructionObserver obs0{};
      CHECK(obs0.status == "initial");
      std::unordered_map<size_t, ConstructionObserver> obs_map{};
      obs_map.emplace(std::piecewise_construct, std::make_tuple(0),
                      std::make_tuple());
      CHECK(obs_map.at(0).status == "initial");
      const auto args = std::forward_as_tuple(obs0, obs_map);
      CHECK(get<0>(args).status == "initial");
      CHECK(get<1>(args).at(0).status == "initial");
      CHECK(&obs0 == &get<0>(args));
      CHECK(&obs_map.at(0) == &get<1>(args).at(0));
      const auto unmapped_args =
          elliptic::dg::subdomain_operator::detail::unmap_all(
              args, size_t{0},
              tmpl::list<tmpl::bool_<true>, tmpl::bool_<false>>{});
      static_assert(std::is_same_v<std::decay_t<decltype(unmapped_args)>,
                                   std::tuple<const ConstructionObserver&,
                                              const ConstructionObserver&>>);
      CHECK(get<0>(unmapped_args).status == "initial");
      CHECK(get<1>(unmapped_args).status == "initial");
      CHECK(&get<0>(unmapped_args) == &obs0);
      CHECK(&get<1>(unmapped_args) == &obs_map.at(0));
  }
}
