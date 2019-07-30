// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/NonlinearSolver/Tags.hpp"

namespace NonlinearSolver {
namespace observe_detail {

using reduction_data = Parallel::ReductionData<
    // Iteration
    Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
    // Residual
    Parallel::ReductionDatum<double, funcl::AssertEqual<>>>;

struct ObservationType {};

struct Registration {
  template <typename ParallelComponent, typename DbTagsList,
            typename ArrayIndex>
  static std::pair<observers::TypeOfObservation, observers::ObservationId>
  register_info(const db::DataBox<DbTagsList>& /*box*/,
                const ArrayIndex& /*array_index*/) noexcept {
    observers::ObservationId fake_initial_observation_id{0., ObservationType{}};
    return {
        observers::TypeOfObservation::Reduction,
        std::move(fake_initial_observation_id)  // NOLINT
    };
  }
};

/*!
 * \brief Contributes data from the residual monitor to the reduction observer
 *
 * With:
 * - `residual_magnitude_tag` = `db::add_tag_prefix<
 * LinearSolver::Tags::Magnitude, db::add_tag_prefix<
 * NonlinearSolver::Tags::Residual, nonlinear_fields_tag>>`
 *
 * Uses:
 * - System:
 *   - `nonlinear_fields_tag`
 * - DataBox:
 *   - `NonlinearSolver::Tags::IterationId`
 *   - `residual_magnitude_tag`
 */
template <typename FieldsTag, typename DbTagsList, typename Metavariables>
void contribute_to_reduction_observer(
    db::DataBox<DbTagsList>& box,
    Parallel::ConstGlobalCache<Metavariables>& cache) noexcept {
  using fields_tag = FieldsTag;
  using residual_magnitude_tag = db::add_tag_prefix<
      LinearSolver::Tags::Magnitude,
      db::add_tag_prefix<NonlinearSolver::Tags::Residual, fields_tag>>;

  const auto observation_id = observers::ObservationId(
      get<NonlinearSolver::Tags::IterationId>(box), ObservationType{});
  auto& reduction_writer = Parallel::get_parallel_component<
      observers::ObserverWriter<Metavariables>>(cache);
  Parallel::threaded_action<observers::ThreadedActions::WriteReductionData>(
      // Node 0 is always the writer, so directly call the component on that
      // node
      reduction_writer[0], observation_id,
      // When multiple nonlinear solves are performed, e.g. for AMR, we'll need
      // to write into separate subgroups, e.g.:
      // `/nonlinear_residuals/<amr_iteration_id>`
      std::string{"/nonlinear_residuals"},
      std::vector<std::string>{"Iteration", "Residual"},
      reduction_data{get<NonlinearSolver::Tags::IterationId>(box),
                     get<residual_magnitude_tag>(box)});
}

}  // namespace observe_detail
}  // namespace NonlinearSolver
