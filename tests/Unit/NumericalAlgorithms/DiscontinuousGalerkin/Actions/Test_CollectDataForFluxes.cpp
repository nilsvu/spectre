// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Rectangle.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Domain/InterfaceComputeTags.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/CollectDataForFluxes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleMortarData.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeDomain.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeMortars.hpp"
#include "ParallelAlgorithms/Initialization/Actions/AddComputeTags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
constexpr size_t VolumeDim = 2;
constexpr size_t FaceDim = VolumeDim - 1;

using TemporalId = int;

struct TemporalIdTag : db::SimpleTag {
  using type = TemporalId;
};

struct BoundaryData {
  bool is_projected;
  TemporalId temporal_id;
  size_t num_points;
  BoundaryData project_to_mortar(
      const Mesh<FaceDim>& /*face_mesh*/, const Mesh<FaceDim>& /*mortar_mesh*/,
      const std::array<Spectral::MortarSize, FaceDim>& /*mortar_size*/) const
      noexcept {
    return {true, temporal_id, num_points};
  }
};

using MortarData = dg::SimpleMortarData<TemporalId, BoundaryData>;

struct MortarDataTag : db::SimpleTag {
  using type = MortarData;
};

struct DgBoundaryScheme {
  static constexpr size_t volume_dim = VolumeDim;
  using temporal_id_tag = TemporalIdTag;
  using mortar_data_tag = MortarDataTag;
  struct boundary_data_computer {
    using argument_tags = tmpl::list<TemporalIdTag, ::Tags::Mesh<FaceDim>>;
    using volume_tags = tmpl::list<TemporalIdTag>;
    static BoundaryData apply(const TemporalId& temporal_id,
                              const Mesh<FaceDim>& face_mesh) noexcept {
      return {false, temporal_id, face_mesh.number_of_grid_points()};
    }
  };
};

template <typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementIndex<VolumeDim>;
  using const_global_cache_tags = tmpl::list<::Tags::Domain<VolumeDim>>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<db::AddSimpleTags<
                         ::Tags::InitialExtents<VolumeDim>, TemporalIdTag,
                         ::Tags::Next<TemporalIdTag>>>,
                     dg::Actions::InitializeDomain<VolumeDim>,
                     Initialization::Actions::AddComputeTags<tmpl::list<
                         ::Tags::InternalDirections<VolumeDim>,
                         ::Tags::BoundaryDirectionsInterior<VolumeDim>,
                         ::Tags::BoundaryDirectionsExterior<VolumeDim>,
                         ::Tags::InterfaceCompute<
                             ::Tags::InternalDirections<VolumeDim>,
                             ::Tags::Direction<VolumeDim>>,
                         ::Tags::InterfaceCompute<
                             ::Tags::BoundaryDirectionsInterior<VolumeDim>,
                             ::Tags::Direction<VolumeDim>>,
                         ::Tags::InterfaceCompute<
                             ::Tags::BoundaryDirectionsExterior<VolumeDim>,
                             ::Tags::Direction<VolumeDim>>,
                         ::Tags::InterfaceCompute<
                             ::Tags::InternalDirections<VolumeDim>,
                             ::Tags::InterfaceMesh<VolumeDim>>,
                         ::Tags::InterfaceCompute<
                             ::Tags::BoundaryDirectionsInterior<VolumeDim>,
                             ::Tags::InterfaceMesh<VolumeDim>>,
                         ::Tags::InterfaceCompute<
                             ::Tags::BoundaryDirectionsExterior<VolumeDim>,
                             ::Tags::InterfaceMesh<VolumeDim>>>>,
                     dg::Actions::InitializeMortars<DgBoundaryScheme>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<
              dg::Actions::CollectDataForFluxes<
                  DgBoundaryScheme, ::Tags::InternalDirections<VolumeDim>>,
              dg::Actions::CollectDataForFluxes<
                  DgBoundaryScheme,
                  ::Tags::BoundaryDirectionsInterior<VolumeDim>>>>>;
};

struct Metavariables {
  using component_list = tmpl::list<ElementArray<Metavariables>>;
  enum class Phase { Initialization, Testing, Exit };
};

}  // namespace

SPECTRE_TEST_CASE("Unit.DG.Actions.CollectDataForFluxes",
                  "[Unit][NumericalAlgorithms][Actions]") {
  domain::creators::register_derived_with_charm();

  using element_array = ElementArray<Metavariables>;

  // Reference element:
  // ^ eta
  // +-+-+> xi
  // |X| |
  // +-+-+
  // | | |
  // +-+-+
  const ElementId<VolumeDim> self_id{0, {{{1, 0}, {1, 1}}}};
  const ElementId<VolumeDim> east_id{0, {{{1, 1}, {1, 1}}}};
  const auto mortar_id_east =
      std::make_pair(Direction<VolumeDim>::upper_xi(), east_id);
  const auto mortar_id_west =
      std::make_pair(Direction<VolumeDim>::lower_xi(),
                     ElementId<VolumeDim>::external_boundary_id());
  const ElementId<VolumeDim> south_id{0, {{{1, 0}, {1, 0}}}};
  const auto mortar_id_south =
      std::make_pair(Direction<VolumeDim>::lower_eta(), south_id);
  const auto mortar_id_north =
      std::make_pair(Direction<VolumeDim>::upper_eta(),
                     ElementId<VolumeDim>::external_boundary_id());

  const domain::creators::Rectangle domain_creator{
      {{-0.5, 0.}}, {{1.5, 1.}}, {{false, false}}, {{1, 1}}, {{3, 2}}};

  const TemporalId time{1};

  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      {domain_creator.create_domain()}};
  ActionTesting::emplace_component_and_initialize<element_array>(
      &runner, self_id, {domain_creator.initial_extents(), time, time + 1});
  ActionTesting::next_action<element_array>(make_not_null(&runner), self_id);
  ActionTesting::next_action<element_array>(make_not_null(&runner), self_id);
  ActionTesting::next_action<element_array>(make_not_null(&runner), self_id);
  runner.set_phase(Metavariables::Phase::Testing);
  const auto get_tag = [&runner, &self_id](auto tag_v) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<element_array, tag>(runner, self_id);
  };

  const auto check_mortar = [&get_tag, &time ](
      const std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>& mortar_id,
      const size_t num_points) noexcept {
    CAPTURE(mortar_id);
    const auto& all_mortar_data =
        get_tag(::Tags::Mortars<MortarDataTag, VolumeDim>{});
    const auto& boundary_data = all_mortar_data.at(mortar_id).local_data(time);
    CHECK(boundary_data.temporal_id == time);
    CHECK(boundary_data.num_points == num_points);
  };

  // Collect on internal directions
  ActionTesting::next_action<element_array>(make_not_null(&runner), self_id);
  check_mortar(mortar_id_east, 2);
  check_mortar(mortar_id_south, 3);

  // Collect on external directions
  ActionTesting::next_action<element_array>(make_not_null(&runner), self_id);
  check_mortar(mortar_id_west, 2);
  check_mortar(mortar_id_north, 3);
}
