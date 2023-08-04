// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataBox/Tag.hpp"
#include "Framework/ActionTesting.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/Actions/Goto.hpp"
#include "Time/Actions/ChangeStepSize.hpp"
#include "Time/AdaptiveSteppingDiagnostics.hpp"
#include "Time/Slab.hpp"
#include "Time/StepChoosers/Constant.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags/AdaptiveSteppingDiagnostics.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/IsUsingTimeSteppingErrorControl.hpp"
#include "Time/Tags/StepChoosers.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include <pup.h>
// IWYU pragma: no_include <unordered_map>

// IWYU pragma: no_forward_declare ActionTesting::InitializeDataBox

namespace {
// a silly step chooser that just always rejects, to test the step rejection
// control-flow.
struct StepRejector : public StepChooser<StepChooserUse::LtsStep> {
  using argument_tags = tmpl::list<>;
  using compute_tags = tmpl::list<>;
  using PUP::able::register_constructor;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  WRAPPED_PUPable_decl_template(StepRejector);  // NOLINT
#pragma GCC diagnostic pop
  explicit StepRejector(CkMigrateMessage* /*unused*/) {}
  StepRejector() = default;

  std::pair<double, bool> operator()(const double last_step_magnitude) const {
    return {last_step_magnitude, false};
  }

  bool uses_local_data() const override { return false; }

  void pup(PUP::er& /*p*/) override {}
};

PUP::able::PUP_ID StepRejector::my_PUP_ID = 0;

struct Var : db::SimpleTag {
  using type = double;
};

using history_tag = Tags::HistoryEvolvedVariables<Var>;

struct System {
  using variables_tag = Var;
};

struct NoOpLabel {};

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tags = tmpl::list<Tags::TimeStepper<LtsTimeStepper>>;
  using simple_tags =
      tmpl::list<Tags::TimeStepId, Tags::Next<Tags::TimeStepId>, Tags::TimeStep,
                 Tags::Next<Tags::TimeStep>, ::Tags::StepChoosers,
                 Tags::IsUsingTimeSteppingErrorControl,
                 Tags::AdaptiveSteppingDiagnostics, history_tag,
                 typename System::variables_tag>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<Actions::ChangeStepSize<
                         typename Metavariables::step_choosers_to_use>,
                     ::Actions::Label<NoOpLabel>,
                     /*UpdateU action is required to satisfy internal checks of
                       `ChangeStepSize`. It is not used in the test.*/
                     Actions::UpdateU<System>>>>;
};

template <typename StepChoosersToUse = AllStepChoosers>
struct Metavariables {
  using step_choosers_to_use = StepChoosersToUse;
  using system = System;
  static constexpr bool local_time_stepping = true;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                   tmpl::list<StepChoosers::Constant<StepChooserUse::LtsStep>,
                              StepRejector>>>;
  };
  using component_list = tmpl::list<Component<Metavariables>>;
};

template <typename StepChoosersToUse = AllStepChoosers>
void check(const bool time_runs_forward,
           std::unique_ptr<LtsTimeStepper> time_stepper, const Time& time,
           const double request, const TimeDelta& expected_step,
           const bool reject_step) {
  CAPTURE(time);
  CAPTURE(request);

  const TimeDelta initial_step_size =
      (time_runs_forward ? 1 : -1) * time.slab().duration();

  using component = Component<Metavariables<StepChoosersToUse>>;
  using MockRuntimeSystem =
      ActionTesting::MockRuntimeSystem<Metavariables<StepChoosersToUse>>;
  using Constant = StepChoosers::Constant<StepChooserUse::LtsStep>;
  MockRuntimeSystem runner{{std::move(time_stepper)}};

  // Initialize the component
  ActionTesting::emplace_component_and_initialize<component>(
      &runner, 0,
      {TimeStepId(
           time_runs_forward, -1,
           (time_runs_forward ? time.slab().end() : time.slab().start()) -
               initial_step_size),
       TimeStepId(time_runs_forward, 0, time), initial_step_size,
       initial_step_size,
       reject_step
           ? make_vector<std::unique_ptr<StepChooser<StepChooserUse::LtsStep>>>(
                 std::make_unique<Constant>(2. * request),
                 std::make_unique<Constant>(request),
                 std::make_unique<StepRejector>())
           : make_vector<std::unique_ptr<StepChooser<StepChooserUse::LtsStep>>>(
                 std::make_unique<Constant>(2. * request),
                 std::make_unique<Constant>(request),
                 std::make_unique<Constant>(2. * request)),
       false, AdaptiveSteppingDiagnostics{1, 2, 3, 4, 5},
       typename history_tag::type{}, 1.});

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  runner.template next_action<component>(0);
  const auto& box = ActionTesting::get_databox<component>(runner, 0);

  const size_t index =
      ActionTesting::get_next_action_index<component>(runner, 0);
  if (reject_step) {
    // if the step is rejected, it should jump to the UpdateU action
    CHECK(index == 2_st);
    CHECK(db::get<Tags::AdaptiveSteppingDiagnostics>(box) ==
          AdaptiveSteppingDiagnostics{1, 2, 3, 4, 6});
  } else {
    CHECK(index == 1_st);
    CHECK(db::get<Tags::AdaptiveSteppingDiagnostics>(box) ==
          AdaptiveSteppingDiagnostics{1, 2, 3, 4, 5});
  }
  CHECK(db::get<Tags::Next<Tags::TimeStep>>(box) == expected_step);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.Actions.ChangeStepSize", "[Unit][Time][Actions]") {
  register_classes_with_charm<TimeSteppers::AdamsBashforth>();
  register_factory_classes_with_charm<Metavariables<>>();
  const Slab slab(-5., -2.);
  const double slab_length = slab.duration().value();
  for (auto reject_step : {true, false}) {
    check(true, std::make_unique<TimeSteppers::AdamsBashforth>(1),
          slab.start() + slab.duration() / 4, slab_length / 5.,
          slab.duration() / 8, reject_step);
    check(true, std::make_unique<TimeSteppers::AdamsBashforth>(1),
          slab.start() + slab.duration() / 4, slab_length, slab.duration() / 4,
          reject_step);
    check(false, std::make_unique<TimeSteppers::AdamsBashforth>(1),
          slab.end() - slab.duration() / 4, slab_length / 5.,
          -slab.duration() / 8, reject_step);
    check(false, std::make_unique<TimeSteppers::AdamsBashforth>(1),
          slab.end() - slab.duration() / 4, slab_length, -slab.duration() / 4,
          reject_step);
  }
  CHECK_THROWS_WITH(
      ([&slab, &slab_length]() {
        check<tmpl::list<StepChoosers::Constant<StepChooserUse::LtsStep>>>(
            true, std::make_unique<TimeSteppers::AdamsBashforth>(1),
            slab.start() + slab.duration() / 4, slab_length / 5.,
            slab.duration() / 8, true);
      })(),
      Catch::Matchers::ContainsSubstring("is not registered"));
}
