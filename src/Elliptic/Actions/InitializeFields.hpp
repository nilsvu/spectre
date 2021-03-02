// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Utilities/TMPL.hpp"

#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Parallel/Printf.hpp"

namespace elliptic::Actions {

namespace detail::Tags {
struct AnalyticShiftFactor : db::SimpleTag {
  using type = double;
  static constexpr Options::String help =
      "Multiply the initial analytic shift by this factor";
  using option_tags = tmpl::list<AnalyticShiftFactor>;
  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double value) noexcept {
    return value;
  }
};
}  // namespace detail::Tags

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
 *
 * \note This action relies on the `SetupDataBox` aggregated initialization
 * mechanism, so `Actions::SetupDataBox` must be present in the `Initialization`
 * phase action list prior to this action.
 */
template <typename System, typename InitialGuessTag>
struct InitializeFields {
 private:
  using system = System;
  using fields_tag = ::Tags::Variables<typename system::primal_fields>;

 public:
  using simple_tags = tmpl::list<fields_tag>;
  using compute_tags = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<detail::Tags::AnalyticShiftFactor>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& inertial_coords =
        get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(box);
    const auto& initial_guess = db::get<InitialGuessTag>(box);
    auto initial_fields = variables_from_tagged_tuple(initial_guess.variables(
        inertial_coords, typename fields_tag::tags_list{}));
    Initialization::mutate_assign<simple_tags>(make_not_null(&box),
                                               std::move(initial_fields));
    db::mutate<Xcts::Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>(
        make_not_null(&box),
        [](const auto shift_excess, const auto& solution,
           const double analytic_shift_factor) {
          *shift_excess = get<::Tags::Analytic<
              Xcts::Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>>(
              *solution);
          for (size_t i = 0; i < 3; ++i) {
            shift_excess->get(i) *= analytic_shift_factor;
          }
        },
        db::get<::Tags::AnalyticSolutionsBase>(box),
        db::get<detail::Tags::AnalyticShiftFactor>(box));
    return {std::move(box)};
  }
};

}  // namespace elliptic::Actions
