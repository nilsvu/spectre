// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <pup.h>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Actions/InitializeFixedSources.hpp"
#include "Elliptic/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeDomain.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

struct ScalarFieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct System {
  using fields_tag = Tags::Variables<tmpl::list<ScalarFieldTag>>;
  using primal_fields = tmpl::list<ScalarFieldTag>;
};

struct Background {
  static tuples::TaggedTuple<Tags::FixedSource<ScalarFieldTag>> variables(
      const tnsr::I<DataVector, 1>& x,
      tmpl::list<Tags::FixedSource<ScalarFieldTag>> /*meta*/) noexcept {
    return {Scalar<DataVector>{get<0>(x)}};
  }
  // NOLINTNEXTLINE
  void pup(PUP::er& /*p*/) noexcept {}
};

template <typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<1>;
  using const_global_cache_tags = tmpl::list<domain::Tags::Domain<1>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<
                         tmpl::list<domain::Tags::InitialRefinementLevels<1>,
                                    domain::Tags::InitialExtents<1>>>,
                     Actions::SetupDataBox, dg::Actions::InitializeDomain<1>>>,

      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<Actions::SetupDataBox,
                     elliptic::Actions::InitializeFixedSources<
                         typename Metavariables::system,
                         elliptic::Tags::Background<Background>>>>>;
};

struct Metavariables {
  using system = System;
  using component_list = tmpl::list<ElementArray<Metavariables>>;
  using const_global_cache_tags =
      tmpl::list<elliptic::Tags::Background<Background>>;
  enum class Phase { Initialization, Testing, Exit };
};

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Actions.InitializeFixedSources",
                  "[Unit][Elliptic][Actions]") {
  domain::creators::register_derived_with_charm();
  // Which element we work with does not matter for this test
  const ElementId<1> element_id{0, {{SegmentId{2, 1}}}};
  const domain::creators::Interval domain_creator{
      {{-0.5}}, {{1.5}}, {{false}}, {{2}}, {{4}}};

  using metavariables = Metavariables;
  using element_array = ElementArray<metavariables>;
  ActionTesting::MockRuntimeSystem<metavariables> runner{
      {std::make_unique<Background<1>>() domain_creator.create_domain()}};
  ActionTesting::emplace_component_and_initialize<element_array>(
      &runner, element_id,
      {domain_creator.initial_refinement_levels(),
       domain_creator.initial_extents()});
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
  }
  ActionTesting::set_phase(make_not_null(&runner),
                           metavariables::Phase::Testing);
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
  }
  const auto get_tag = [&runner, &element_id](auto tag_v) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<element_array, tag>(runner,
                                                              element_id);
  };

  const auto& inertial_coords =
      get_tag(domain::Tags::Coordinates<1, Frame::Inertial>{});
  CHECK(get(get_tag(::Tags::FixedSource<ScalarFieldTag>{})) ==
        get<0>(inertial_coords));
}
