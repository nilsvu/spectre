// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <pup.h>

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

namespace detail {
template <typename LocalArraySectionIdTag>
struct ArraySectionIdOption {
  static std::string name() noexcept {
    return Options::name<LocalArraySectionIdTag>();
  }
  using type = typename LocalArraySectionIdTag::type;
  static constexpr Options::String help{"Array section to trigger on."};
  static type default_value() noexcept {
    return std::numeric_limits<type>::max();
  }
};

template <>
struct ArraySectionIdOption<void> {
  using type = void;
};
}  // namespace detail

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

  using ArraySectionId = tmpl::conditional_t<
      std::is_same_v<ArraySectionIdTag, void>, bool,
      typename detail::ArraySectionIdOption<ArraySectionIdTag>::type>;

  struct N {
    using type = uint64_t;
    static constexpr Options::String help{"How frequently to trigger."};
    static type lower_bound() noexcept { return 1; }
  };
  struct Offset {
    using type = uint64_t;
    static constexpr Options::String help{"First iteration to trigger on."};
  };

  using options = tmpl::flatten<tmpl::list<
      N, Offset,
      tmpl::conditional_t<std::is_same_v<ArraySectionIdTag, void>, tmpl::list<>,
                          detail::ArraySectionIdOption<ArraySectionIdTag>>>>;
  static constexpr Options::String help{
      "Trigger every N iterations after a given offset."};

  EveryNIterations(const uint64_t interval, const uint64_t offset) noexcept
      : interval_(interval), offset_(offset) {}

  EveryNIterations(const uint64_t interval, const uint64_t offset,
                   ArraySectionId array_section_id) noexcept
      : interval_(interval),
        offset_(offset),
        array_section_id_(std::move(array_section_id)) {}

  using argument_tags = tmpl::flatten<tmpl::list<
      IterationId, tmpl::conditional_t<std::is_same_v<ArraySectionIdTag, void>,
                                       tmpl::list<>, ArraySectionIdTag>>>;

  bool operator()(
      const typename IterationId::type& iteration_id) const noexcept {
    const auto step_number = static_cast<uint64_t>(iteration_id);
    return step_number >= offset_ and (step_number - offset_) % interval_ == 0;
  }

  bool operator()(const typename IterationId::type& iteration_id,
                  const ArraySectionId& array_section_id) const noexcept {
    return (array_section_id_ == std::numeric_limits<ArraySectionId>::max() or
            array_section_id == array_section_id_) and
           (*this)(iteration_id);
  }

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept {  // NOLINT
    p | interval_;
    p | offset_;
    p | array_section_id_;
  }

 private:
  uint64_t interval_{0};
  uint64_t offset_{0};
  ArraySectionId array_section_id_ = std::numeric_limits<ArraySectionId>::max();
};

/// \cond
template <typename IterationId, typename ArraySectionIdTag,
          typename TriggerRegistrars>
PUP::able::PUP_ID EveryNIterations<IterationId, ArraySectionIdTag,
                                   TriggerRegistrars>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace Triggers
}  // namespace elliptic
