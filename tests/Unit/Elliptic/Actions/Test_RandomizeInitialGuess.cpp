// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Actions/RandomizeInitialGuess.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/TMPL.hpp"

namespace {

struct ScalarFieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct System {
  using primal_fields = tmpl::list<ScalarFieldTag>;
};

template <typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<1>;
  using const_global_cache_tags = tmpl::list<>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<
              tmpl::list<::Tags::Variables<tmpl::list<ScalarFieldTag>>>>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<elliptic::Actions::RandomizeInitialGuess<
              typename Metavariables::system>>>>;
};

struct Metavariables {
  using system = System;
  using component_list = tmpl::list<ElementArray<Metavariables>>;
  using const_global_cache_tags = tmpl::list<>;
  enum class Phase { Initialization, Testing, Exit };
};

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Actions.RandomizeInitialGuess",
                  "[Unit][Elliptic][Actions]") {
  const double amplitude = 1.e-2;
  const DataVector used_for_size{5};

  // Which element we work with does not matter for this test
  const ElementId<1> element_id{0, {{SegmentId{2, 1}}}};
  // Generate some random field data
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<double> dist(-1., 1.);
  const auto fields =
      make_with_random_values<Variables<tmpl::list<ScalarFieldTag>>>(
          make_not_null(&generator), make_not_null(&dist), used_for_size);

  using element_array = ElementArray<Metavariables>;
  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      {typename elliptic::Actions::RandomizeInitialGuess<
          System>::RandomParameters{amplitude, std::nullopt}}};
  ActionTesting::emplace_component_and_initialize<element_array>(
      &runner, element_id, {fields});
  ActionTesting::set_phase(make_not_null(&runner),
                           Metavariables::Phase::Testing);
  ActionTesting::next_action<element_array>(make_not_null(&runner), element_id);
  const auto get_tag = [&runner, &element_id ](auto tag_v) -> const auto& {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<element_array, tag>(runner,
                                                              element_id);
  };

  // Test that the fields have changed, but differ only by the `amplitude`
  const DataVector& fields_randomized = get(get_tag(ScalarFieldTag{}));
  const DataVector& fields_original = get(get<ScalarFieldTag>(fields));
  CHECK(fields_randomized != fields_original);
  for (size_t i = 0; i < fields.size(); ++i) {
    CHECK(abs(fields_randomized[i] - fields_original[i]) <= amplitude);
  }
}
