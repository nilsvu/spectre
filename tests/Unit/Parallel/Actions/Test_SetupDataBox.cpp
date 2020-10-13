// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"  // IWYU pragma: keep
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Label1;
struct Label2;

struct CounterTag : db::SimpleTag {
  using type = size_t;
};

struct VectorTag : db::SimpleTag {
  using type = DataVector;
};

struct SquareVectorTag : db::SimpleTag {
  using type = DataVector;
};

struct SquareVectorCompute : SquareVectorTag, db::ComputeTag {
  using argument_tags = tmpl::list<VectorTag>;
  static void function(const gsl::not_null<DataVector*> result,
                       const DataVector& vector) noexcept {
    *result = square(vector);
  }
  using return_type = DataVector;
  using base = SquareVectorTag;
};


struct InitializationAction {
  using simple_tags = tmpl::list<CounterTag, VectorTag>;
  using compute_tags = tmpl::list<SquareVectorCompute>;
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    CHECK(db::get<CounterTag>(box) == 0_st);
    CHECK(db::get<VectorTag>(box).size() == 0_st);
    CHECK(db::get<SquareVectorTag>(box).size() == 0_st);
    db::mutate_assign<VectorTag>(make_not_null(&box), DataVector{1.2, 3.0});
    CHECK(db::get<SquareVectorTag>(box) == DataVector{1.44, 9.0});
    return {std::move(box)};
  }
};

/// [component]
template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;

  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<Actions::SetupDataBox, InitializationAction>>>;
};
/// [component]

/// [metavariables]
struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;

  enum class Phase { Initialization, Exit };
};
/// [metavariables]
}  // namespace

SPECTRE_TEST_CASE("Unit.Parallel.SetupDataBox", "[Unit][Parallel][Actions]") {
  using component = Component<Metavariables>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  MockRuntimeSystem runner{{}};
  ActionTesting::emplace_component<component>(&runner, 0);
  for (size_t i = 0; i < 2; ++i) {
    runner.next_action<component>(0);
  }
}
