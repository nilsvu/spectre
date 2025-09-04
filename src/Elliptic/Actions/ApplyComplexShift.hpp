// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <complex>
#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Variables.hpp"
#include "Options/String.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic {

namespace OptionTags {
template <typename OptionsGroup>
struct ComplexShift {
  using type = double;
  static constexpr Options::String help =
      "Subtracts i alpha from the operator.";
  using group = OptionsGroup;
};
}  // namespace OptionTags

namespace Tags {
template <typename OptionsGroup>
struct ComplexShift : db::SimpleTag {
  using type = double;
  static constexpr bool pass_metavariables = false;
  using option_tags = tmpl::list<OptionTags::ComplexShift<OptionsGroup>>;
  static type create_from_options(const type value) { return value; };
  static std::string name() {
    return "ComplexShift(" + pretty_type::name<OptionsGroup>() + ")";
  }
};
}  // namespace Tags

namespace Actions {

/*!
 * \brief Initialize the dynamic fields of the elliptic system, i.e. those we
 * solve for.
 *
 * Uses:
 * - System:
 *   - `primal_fields`
 * - DataBox:
 *   - `InitialGuessTag`
 *   - `Tags::Coordinates<Dim, Frame::Inertial>`
 *
 * DataBox:
 * - Adds:
 *   - `primal_fields`
 */
template <typename OperandTag, typename OperatorAppliedToOperandTag,
          typename OptionsGroup>
struct ApplyComplexShift {
  using const_global_cache_tags =
      tmpl::list<elliptic::Tags::ComplexShift<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*element_id*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    db::mutate<OperatorAppliedToOperandTag>(
        [](const auto operator_applied_to_operand, const auto& operand,
           const double complex_shift) {
          *operator_applied_to_operand -=
              std::complex<double>(0.0, complex_shift) * operand;
        },
        make_not_null(&box), db::get<OperandTag>(box),
        db::get<elliptic::Tags::ComplexShift<OptionsGroup>>(box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

}  // namespace Actions
}  // namespace elliptic
