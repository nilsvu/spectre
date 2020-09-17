// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <pup.h>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic {
namespace Triggers {
/// \cond
template <typename IterationId, typename ArraySectionIdTag,
          typename TriggerRegistrars>
class EveryNIterations;
/// \endcond

namespace Registrars {
template <typename IterationId, typename ArraySectionIdTag = void>
using EveryNIterations =
    ::Registration::Registrar<Triggers::EveryNIterations, IterationId,
                              ArraySectionIdTag>;
}  // namespace Registrars

/// \ingroup EventsAndTriggersGroup
/// Trigger every N iterations after a given offset.
template <typename IterationId, typename ArraySectionIdTag = void,
          typename TriggerRegistrars = tmpl::list<
              Registrars::EveryNIterations<IterationId, ArraySectionIdTag>>>
class EveryNIterations : public Trigger<TriggerRegistrars> {
 public:
  /// \cond
  EveryNIterations() = default;
  explicit EveryNIterations(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(EveryNIterations);  // NOLINT
  /// \endcond

  struct N {
    using type = uint64_t;
    static constexpr Options::String help{"How frequently to trigger."};
    static type lower_bound() noexcept { return 1; }
  };
  struct Offset {
    using type = uint64_t;
    static constexpr Options::String help{"First iteration to trigger on."};
    static type default_value() noexcept { return 0; }
  };

  using options = tmpl::list<N, Offset>;
  static constexpr Options::String help{
      "Trigger every N iterations after a given offset."};

  EveryNIterations(const uint64_t interval, const uint64_t offset) noexcept
      : interval_(interval), offset_(offset) {}

  using argument_tags = tmpl::flatten<tmpl::list<
      IterationId,
      tmpl::conditional_t<std::is_same_v<ArraySectionIdTag, void>, tmpl::list<>,
                          Parallel::Tags::SectionBase<ArraySectionIdTag>>>>;

  bool operator()(
      const typename IterationId::type& iteration_id) const noexcept {
    const auto step_number = static_cast<uint64_t>(iteration_id);
    return step_number >= offset_ and (step_number - offset_) % interval_ == 0;
  }

  template <typename ArraySectionProxy>
  bool operator()(const typename IterationId::type& iteration_id,
                  const ArraySectionProxy& section) const noexcept {
    return section and (*this)(iteration_id);
  }

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept {  // NOLINT
    p | interval_;
    p | offset_;
  }

 private:
  uint64_t interval_{0};
  uint64_t offset_{0};
};

/// \cond
template <typename IterationId, typename ArraySectionIdTag,
          typename TriggerRegistrars>
PUP::able::PUP_ID EveryNIterations<IterationId, ArraySectionIdTag,
                                   TriggerRegistrars>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace Triggers
}  // namespace elliptic
