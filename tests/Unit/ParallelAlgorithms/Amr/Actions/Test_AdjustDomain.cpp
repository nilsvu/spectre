// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <deque>
#include <pup.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Domain/Amr/Info.hpp"
#include "Domain/Amr/Tags/Flags.hpp"
#include "Domain/Amr/Tags/NeighborFlags.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Amr/Actions/AdjustDomain.hpp"
#include "ParallelAlgorithms/Amr/Actions/Component.hpp"
#include "ParallelAlgorithms/Amr/Actions/CreateChild.hpp"
#include "ParallelAlgorithms/Amr/Actions/CreateParent.hpp"
#include "ParallelAlgorithms/Amr/Projectors/DefaultInitialize.hpp"
#include "ParallelAlgorithms/Amr/Protocols/AmrMetavariables.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct MockCreateParent {
  template <typename ParallelComponent, typename DataBox,
            typename Metavariables, typename ElementProxy>
  static void apply(
      DataBox& /*box*/, Parallel::GlobalCache<Metavariables>& /*cache*/,
      const int /*array_index*/, ElementProxy /*element_proxy*/,
      ElementId<Metavariables::volume_dim> parent_id,
      const ElementId<Metavariables::volume_dim>& child_id,
      std::deque<ElementId<Metavariables::volume_dim>> sibling_ids_to_collect,
      const std::unordered_map<Parallel::Phase, size_t>&
          child_phase_bookmarks) {
    CHECK(parent_id == ElementId<1>{0, std::array{SegmentId{2, 0}}});
    CHECK(child_id == ElementId<1>{0, std::array{SegmentId{3, 0}}});
    CHECK(sibling_ids_to_collect ==
          std::deque{ElementId<1>{0, std::array{SegmentId{3, 1}}}});
    CHECK(child_phase_bookmarks.empty());
  }
};

struct MockCreateChild {
  template <typename ParallelComponent, typename DataBox,
            typename Metavariables, typename ElementProxy>
  static void apply(
      DataBox& /*box*/, Parallel::GlobalCache<Metavariables>& /*cache*/,
      const int /*array_index*/, ElementProxy /*element_proxy*/,
      ElementId<Metavariables::volume_dim> parent_id,
      std::vector<ElementId<Metavariables::volume_dim>> children_ids,
      const size_t index_of_child_id,
      const std::unordered_map<Parallel::Phase, size_t>&
          parent_phase_bookmarks) {
    CHECK(parent_id == ElementId<1>{0, std::array{SegmentId{1, 1}}});
    if (index_of_child_id == 0) {
      CHECK(children_ids ==
            std::vector{ElementId<1>{0, std::array{SegmentId{2, 2}}},
                        ElementId<1>{0, std::array{SegmentId{2, 3}}}});
    } else {
      CHECK(index_of_child_id == 1);
      CHECK(children_ids ==
            std::vector{ElementId<1>{0, std::array{SegmentId{2, 2}}},
                        ElementId<1>{0, std::array{SegmentId{2, 3}}}});
    }
    CHECK(parent_phase_bookmarks.empty());
  }
};

template <typename Metavariables>
struct ArrayComponent {
  using metavariables = Metavariables;
  static constexpr size_t volume_dim = Metavariables::volume_dim;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<volume_dim>;
  using const_global_cache_tags =
      tmpl::list<logging::Tags::Verbosity<amr::OptionTags::AmrGroup>>;
  using simple_tags = tmpl::list<
      domain::Tags::Element<volume_dim>, domain::Tags::Mesh<volume_dim>,
      domain::Tags::NeighborMesh<volume_dim>, amr::Tags::Info<volume_dim>,
      amr::Tags::NeighborInfo<volume_dim>>;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>>;
};

template <typename Metavariables>
struct SingletonComponent {
  using metavariables = Metavariables;
  using array_index = int;

  using component_being_mocked = amr::Component<Metavariables>;
  static constexpr size_t volume_dim = Metavariables::volume_dim;
  using chare_type = ActionTesting::MockSingletonChare;
  using const_global_cache_tags = tmpl::list<>;
  using simple_tags = tmpl::list<>;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>>;
  using replace_these_simple_actions =
      tmpl::list<amr::Actions::CreateChild, amr::Actions::CreateParent>;
  using with_these_simple_actions =
      tmpl::list<MockCreateChild, MockCreateParent>;
};

struct Metavariables {
  static constexpr size_t volume_dim = 1;
  using component_list = tmpl::list<ArrayComponent<Metavariables>,
                                    SingletonComponent<Metavariables>>;
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}

  struct amr : tt::ConformsTo<::amr::protocols::AmrMetavariables> {
    using projectors = tmpl::list<::amr::projectors::DefaultInitialize<
        Parallel::Tags::GlobalCacheImpl<Metavariables>,
        domain::Tags::NeighborMesh<1>>>;
  };
};

void check_box(const ActionTesting::MockRuntimeSystem<Metavariables>& runner,
               const ElementId<1>& element_id,
               const Element<1>& expected_element, const Mesh<1>& expected_mesh,
               const DirectionalIdMap<1, Mesh<1>>& expected_neighbor_meshes,
               const amr::Info<1>& expected_info,
               const std::unordered_map<ElementId<1>, amr::Info<1>>&
                   expected_neighbor_info) {
  using array_component = ArrayComponent<Metavariables>;
  CHECK(
      ActionTesting::get_databox_tag<array_component, domain::Tags::Element<1>>(
          runner, element_id) == expected_element);
  CHECK(ActionTesting::get_databox_tag<array_component, domain::Tags::Mesh<1>>(
            runner, element_id) == expected_mesh);
  CHECK(ActionTesting::get_databox_tag<array_component, amr::Tags::Info<1>>(
            runner, element_id) == expected_info);
  CHECK(ActionTesting::get_databox_tag<array_component,
                                       amr::Tags::NeighborInfo<1>>(
            runner, element_id) == expected_neighbor_info);
  const auto& neighbor_meshes =
      ActionTesting::get_databox_tag<array_component,
                                     domain::Tags::NeighborMesh<1>>(runner,
                                                                    element_id);
  for (const auto& [direction_id, mesh] : neighbor_meshes) {
    CHECK(mesh == expected_neighbor_meshes.at(direction_id));
  }
}

void test() {
  using array_component = ArrayComponent<Metavariables>;
  using singleton_component = SingletonComponent<Metavariables>;

  const OrientationMap<1> aligned{};

  const ElementId<1> element_1_id{0, std::array{SegmentId{3, 0}}};
  const ElementId<1> element_2_id{0, std::array{SegmentId{3, 1}}};
  const ElementId<1> element_3_id{0, std::array{SegmentId{2, 1}}};
  const ElementId<1> element_4_id{0, std::array{SegmentId{1, 1}}};

  const Element<1> element_1{
      element_1_id,
      DirectionMap<1, Neighbors<1>>{
          {Direction<1>::upper_xi(),
           Neighbors<1>{std::unordered_set{element_2_id}, aligned}}}};
  const Element<1> element_2{
      element_2_id,
      DirectionMap<1, Neighbors<1>>{
          {Direction<1>::lower_xi(),
           Neighbors<1>{std::unordered_set{element_1_id}, aligned}},
          {Direction<1>::upper_xi(),
           Neighbors<1>{std::unordered_set{element_3_id}, aligned}}}};
  const Element<1> element_3{
      element_3_id,
      DirectionMap<1, Neighbors<1>>{
          {Direction<1>::lower_xi(),
           Neighbors<1>{std::unordered_set{element_2_id}, aligned}},
          {Direction<1>::upper_xi(),
           Neighbors<1>{std::unordered_set{element_4_id}, aligned}}}};
  const Element<1> element_4{
      element_4_id,
      DirectionMap<1, Neighbors<1>>{
          {Direction<1>::lower_xi(),
           Neighbors<1>{std::unordered_set{element_3_id}, aligned}}}};

  const Mesh<1> element_1_mesh{std::array{3_st}, Spectral::Basis::Legendre,
                               Spectral::Quadrature::GaussLobatto};
  const Mesh<1> element_2_mesh{std::array{4_st}, Spectral::Basis::Legendre,
                               Spectral::Quadrature::GaussLobatto};
  const Mesh<1> element_3_mesh{std::array{5_st}, Spectral::Basis::Legendre,
                               Spectral::Quadrature::GaussLobatto};
  const Mesh<1> element_4_mesh{std::array{7_st}, Spectral::Basis::Legendre,
                               Spectral::Quadrature::GaussLobatto};
  const Mesh<1> element_3_mesh_post_refinement{
      std::array{6_st}, Spectral::Basis::Legendre,
      Spectral::Quadrature::GaussLobatto};

  amr::Info<1> element_1_info{{amr::Flag::Join}, element_2_mesh};
  amr::Info<1> element_2_info{{amr::Flag::Join}, element_2_mesh};
  amr::Info<1> element_3_info{{amr::Flag::IncreaseResolution},
                              element_3_mesh_post_refinement};
  amr::Info<1> element_4_info{{amr::Flag::Split}, element_4_mesh};

  std::unordered_map<ElementId<1>, amr::Info<1>> element_1_neighbor_info{
      {element_2_id, element_2_info}};
  std::unordered_map<ElementId<1>, amr::Info<1>> element_2_neighbor_info{
      {element_1_id, element_1_info}, {element_3_id, element_3_info}};
  std::unordered_map<ElementId<1>, amr::Info<1>> element_3_neighbor_info{
      {element_2_id, element_2_info}, {element_4_id, element_4_info}};
  std::unordered_map<ElementId<1>, amr::Info<1>> element_4_neighbor_info{
      {element_3_id, element_3_info}};

  using NeighborMeshes = DirectionalIdMap<1, Mesh<1>>;

  ActionTesting::MockRuntimeSystem<Metavariables> runner{{::Verbosity::Debug}};
  ActionTesting::emplace_component_and_initialize<array_component>(
      &runner, element_1_id,
      {element_1, element_1_mesh, NeighborMeshes{}, element_1_info,
       element_1_neighbor_info});
  ActionTesting::emplace_component_and_initialize<array_component>(
      &runner, element_2_id,
      {element_2, element_2_mesh, NeighborMeshes{}, element_2_info,
       element_2_neighbor_info});
  ActionTesting::emplace_component_and_initialize<array_component>(
      &runner, element_3_id,
      {element_3, element_3_mesh, NeighborMeshes{}, element_3_info,
       element_3_neighbor_info});
  ActionTesting::emplace_component_and_initialize<array_component>(
      &runner, element_4_id,
      {element_4, element_4_mesh, NeighborMeshes{}, element_4_info,
       element_4_neighbor_info});
  ActionTesting::emplace_component<singleton_component>(&runner, 0);

  const auto check_for_empty_queues_on_elements =
      [&element_1_id, &element_2_id, &element_3_id, &element_4_id, &runner]() {
        for (const auto& id : std::vector{element_1_id, element_2_id,
                                          element_3_id, element_4_id}) {
          CHECK(ActionTesting::is_simple_action_queue_empty<array_component>(
              runner, id));
        }
      };
  check_for_empty_queues_on_elements();
  CHECK(ActionTesting::number_of_queued_simple_actions<singleton_component>(
            runner, 0) == 0);

  // This should queue CreateParent on the singleton
  ActionTesting::simple_action<array_component, amr::Actions::AdjustDomain>(
      make_not_null(&runner), element_1_id);
  check_for_empty_queues_on_elements();
  CHECK(ActionTesting::number_of_queued_simple_actions<singleton_component>(
            runner, 0) == 1);

  // This should do nothing
  ActionTesting::simple_action<array_component, amr::Actions::AdjustDomain>(
      make_not_null(&runner), element_2_id);
  check_for_empty_queues_on_elements();
  CHECK(ActionTesting::number_of_queued_simple_actions<singleton_component>(
            runner, 0) == 1);

  // This should p-refine element_3
  ActionTesting::simple_action<array_component, amr::Actions::AdjustDomain>(
      make_not_null(&runner), element_3_id);
  check_for_empty_queues_on_elements();
  CHECK(ActionTesting::number_of_queued_simple_actions<singleton_component>(
            runner, 0) == 1);

  // This should queue CreateChild on the singleton
  ActionTesting::simple_action<array_component, amr::Actions::AdjustDomain>(
      make_not_null(&runner), element_4_id);
  check_for_empty_queues_on_elements();
  CHECK(ActionTesting::number_of_queued_simple_actions<singleton_component>(
            runner, 0) == 2);

  // This will invoke the queued actions on the singleton calling the mock
  // actions
  ActionTesting::invoke_queued_simple_action<singleton_component>(
      make_not_null(&runner), 0);
  check_for_empty_queues_on_elements();
  CHECK(ActionTesting::number_of_queued_simple_actions<singleton_component>(
            runner, 0) == 1);
  ActionTesting::invoke_queued_simple_action<singleton_component>(
      make_not_null(&runner), 0);
  check_for_empty_queues_on_elements();
  CHECK(ActionTesting::number_of_queued_simple_actions<singleton_component>(
            runner, 0) == 0);

  check_box(runner, element_1_id, element_1, element_1_mesh, NeighborMeshes{},
            element_1_info, element_1_neighbor_info);
  check_box(runner, element_2_id, element_2, element_2_mesh, NeighborMeshes{},
            element_2_info, element_2_neighbor_info);
  check_box(runner, element_4_id, element_4, element_4_mesh, NeighborMeshes{},
            element_4_info, element_4_neighbor_info);
  const ElementId<1> new_parent_id{
      ElementId<1>{0, std::array{SegmentId{2, 0}}}};
  const ElementId<1> new_lower_child_id{
      ElementId<1>{0, std::array{SegmentId{2, 2}}}};
  const Element<1> element_3_post_refinement{
      element_3_id,
      DirectionMap<1, Neighbors<1>>{
          {Direction<1>::lower_xi(),
           Neighbors<1>{std::unordered_set{new_parent_id}, aligned}},
          {Direction<1>::upper_xi(),
           Neighbors<1>{std::unordered_set{new_lower_child_id}, aligned}}}};
  DirectionalIdMap<1, Mesh<1>> expected_element_3_neighbor_meshes;
  expected_element_3_neighbor_meshes.emplace(std::pair{
      DirectionalId{Direction<1>::lower_xi(), new_parent_id}, element_2_mesh});
  expected_element_3_neighbor_meshes.emplace(
      std::pair{DirectionalId{Direction<1>::upper_xi(), new_lower_child_id},
                element_4_mesh});
  check_box(runner, element_3_id, element_3_post_refinement,
            element_3_mesh_post_refinement, expected_element_3_neighbor_meshes,
            {std::array{amr::Flag::Undefined}, element_3_mesh_post_refinement},
            {});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Amr.Actions.AdjustDomain",
                  "[Unit][ParallelAlgorithms]") {
  test();
}
