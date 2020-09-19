// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>
#include <string>
#include <utility>

#include "Informer/LogActions.hpp"
#include "Informer/LoggerComponent.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::Events {

template <typename EventRegistrars>
struct PrintTime;

namespace Registrars {
using PrintTime = ::Registration::Registrar<Events::PrintTime>;
}  // namespace Registrars

template <typename EventRegistrars = tmpl::list<Registrars::PrintTime>>
struct PrintTime;

namespace detail {
struct PrintTimeFormatter {
  static std::string apply(const double time) noexcept {
    return "Current time: " + get_output(time);
  }
};
}  // namespace detail

template <typename EventRegistrars>
struct PrintTime : public Event<EventRegistrars> {
  /// \cond
  explicit PrintTime(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(PrintTime);  // NOLINT
  /// \endcond

  using options = tmpl::list<>;
  static constexpr Options::String help =
      "Prints the time of observation in the simulation.";

  PrintTime() = default;

  using argument_tags = tmpl::list<::Tags::Time>;

  template <typename Metavariables, typename ArrayIndex,
            typename ParallelComponent>
  void operator()(const double time,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ArrayIndex& array_index,
                  const ParallelComponent* const /*meta*/) const noexcept {
    Parallel::contribute_to_reduction<
        logging::Actions::Log<detail::PrintTimeFormatter>>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<double, funcl::AssertEqual<>>>{time},
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        Parallel::get_parallel_component<logging::Logger<Metavariables>>(
            cache));
  }
};

/// \cond
template <typename EventRegistrars>
PUP::able::PUP_ID PrintTime<EventRegistrars>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace evolution::Events
