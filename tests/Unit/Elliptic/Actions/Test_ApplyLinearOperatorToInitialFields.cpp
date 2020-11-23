// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <pup.h>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Actions/ApplyLinearOperatorToInitialFields.hpp"
#include "Elliptic/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace {

struct ScalarFieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};

using fields_tag = Tags::Variables<tmpl::list<ScalarFieldTag>>;
using operator_applied_to_fields_tag =
    db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, fields_tag>;
using operand_tag = db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
using operator_applied_to_operand_tag =
    db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, operand_tag>;

// This action operates on the `operand_tag`. The actions we're testing in this
// file apply it to the `fields_tag` by copying back and forth.
struct ApplyOperator {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<operator_applied_to_operand_tag>(
        make_not_null(&box),
        [](const auto operator_applied_to_operand,
           const auto& operand) noexcept {
          *operator_applied_to_operand = 2. * operand;
        },
        db::get<operand_tag>(box));
    return {std::move(box)};
  }
};

template <typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tags = tmpl::list<>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<
              tmpl::list<fields_tag, operator_applied_to_fields_tag,
                         operand_tag, operator_applied_to_operand_tag>>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<elliptic::Actions::apply_linear_operator_to_initial_fields<
                         ApplyOperator, fields_tag, operand_tag>,
                     Parallel::Actions::TerminatePhase>>>;
};

struct Metavariables {
  using component_list = tmpl::list<ElementArray<Metavariables>>;
  enum class Phase { Initialization, Testing, Exit };
};

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Actions.ApplyLinearOperatorToInitialFields",
                  "[Unit][Elliptic][Actions]") {
  using metavariables = Metavariables;
  using element_array = ElementArray<metavariables>;
  const int element_id = 0;
  ActionTesting::MockRuntimeSystem<metavariables> runner{{}};
  ActionTesting::emplace_component_and_initialize<element_array>(
      &runner, element_id,
      {typename fields_tag::type{5, 1.},
       typename operator_applied_to_fields_tag::type{5},
       typename operand_tag::type{5},
       typename operator_applied_to_operand_tag::type{5}});
  ActionTesting::set_phase(make_not_null(&runner),
                           metavariables::Phase::Testing);
  while (not ActionTesting::get_terminate<element_array>(runner, element_id)) {
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
  }
  const auto get_tag = [&runner, &element_id](auto tag_v) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<element_array, tag>(runner,
                                                              element_id);
  };

  // Test the operator was applied to the fields_tag, despite the operator being
  // implemented for the operand_tag
  CHECK(get(get_tag(LinearSolver::Tags::OperatorAppliedTo<ScalarFieldTag>{})) ==
        2. * get(get_tag(ScalarFieldTag{})));
}
