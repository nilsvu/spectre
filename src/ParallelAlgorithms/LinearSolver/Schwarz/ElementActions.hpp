// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "NumericalAlgorithms/LinearSolver/Gmres.hpp"
#include "NumericalAlgorithms/LinearSolver/InnerProduct.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/SubdomainData.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/SubdomainHelpers.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Tags.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"

#include "Parallel/Printf.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace LinearSolver {
namespace schwarz_detail {

template <typename FieldsTag, typename OptionsGroup>
struct PrepareSolve {
 private:
  using fields_tag = FieldsTag;

 public:
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    Parallel::printf("%s Prepare Schwarz solve\n", array_index);
    db::mutate<LinearSolver::Tags::IterationId<OptionsGroup>,
               LinearSolver::Tags::HasConverged<OptionsGroup>>(
        make_not_null(&box),
        [](const gsl::not_null<size_t*> iteration_id,
           const gsl::not_null<Convergence::HasConverged*> has_converged,
           const size_t& max_iterations) noexcept {
          *iteration_id = std::numeric_limits<size_t>::max();
          *has_converged =
              Convergence::HasConverged{{max_iterations, 0., 0.}, 0, 1., 1.};
        },
        get<LinearSolver::Tags::Iterations<OptionsGroup>>(box));
    return std::forward_as_tuple(std::move(box));
  }
};

template <typename FieldsTag, typename OptionsGroup>
struct PrepareStep {
 private:
  using fields_tag = FieldsTag;

 public:
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    Parallel::printf(
        "%s Prepare Schwarz step %zu\n", array_index,
        get<LinearSolver::Tags::IterationId<OptionsGroup>>(box) + 1);
    db::mutate<LinearSolver::Tags::IterationId<OptionsGroup>,
               LinearSolver::Tags::HasConverged<OptionsGroup>>(
        make_not_null(&box),
        [](const gsl::not_null<size_t*> iteration_id,
           const gsl::not_null<Convergence::HasConverged*> has_converged,
           const size_t& max_iterations) noexcept {
          (*iteration_id)++;
          // Is this needed?
          *has_converged = Convergence::HasConverged{
              {max_iterations, 0., 0.}, *iteration_id, 1., 1.};
        },
        get<LinearSolver::Tags::Iterations<OptionsGroup>>(box));
    return std::forward_as_tuple(std::move(box));
  }
};

template <typename FieldsTag, typename OptionsGroup, typename SubdomainOperator,
          typename WeightingOperator>
struct PerformStep {
 private:
  using fields_tag = FieldsTag;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>;
  static constexpr size_t volume_dim = SubdomainOperator::volume_dim;
  using SubdomainDataType = typename SubdomainOperator::SubdomainDataType;

 public:
  using const_global_cache_tags =
      tmpl::list<LinearSolver::Tags::Iterations<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ElementId<volume_dim>& element_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    Parallel::printf("%s Perform Schwarz step %zu\n", element_index,
                     get<LinearSolver::Tags::IterationId<OptionsGroup>>(box));
    const auto& res = get<residual_tag>(box);
    Parallel::printf("%s Residual: %e\n", element_index,
                     sqrt(inner_product(res, res)));

    // Gather residual and overlap from neighbors
    // TODO: Avoid copying
    const SubdomainDataType residual_subdomain{
        typename SubdomainDataType::element_data_type(get<residual_tag>(box)),
        get<Tags::SubdomainBoundaryData<SubdomainOperator, OptionsGroup>>(box)};

    Parallel::printf("%s  Initial fields: %s\n", element_index,
                     get<fields_tag>(box));
    Parallel::printf("%s  Residual (central): %s\n", element_index,
                     residual_subdomain.element_data);
    Parallel::printf("%s  Overlap with: %d elements\n", element_index,
                     residual_subdomain.boundary_data.size());

    const auto& subdomain_solver =
        get<Tags::SubdomainSolverBase<OptionsGroup>>(box);
    auto subdomain_solve_result = subdomain_solver(
        [&box](const SubdomainDataType& arg) noexcept {
          return db::apply<SubdomainOperator>(box, arg);
        },
        residual_subdomain,
        // Using the residual as initial guess so not iterating the
        // subdomain solver at all is the identity operation
        residual_subdomain);
    const auto& subdomain_solve_has_converged = subdomain_solve_result.first;
    auto& subdomain_solution = subdomain_solve_result.second;
    if (not subdomain_solve_has_converged or
        subdomain_solve_has_converged.reason() ==
            Convergence::Reason::MaxIterations) {
      Parallel::printf(
          "WARNING: Subdomain solver on element %s did not converge in %zu "
          "iterations.\n",
          element_index, subdomain_solve_has_converged.num_iterations());
    } else {
      Parallel::printf("%s Subdomain solver converged in %zu iterations.\n",
                       element_index,
                       subdomain_solve_has_converged.num_iterations());
    }

    Parallel::printf("%s  Subdomain solution (central): %s\n", element_index,
                     subdomain_solution.element_data);

    // Weighting
    db::apply<WeightingOperator>(box, make_not_null(&subdomain_solution));

    db::mutate<Tags::SubdomainBoundaryData<SubdomainOperator, OptionsGroup>>(
        make_not_null(&box),
        [&subdomain_solution](
            const gsl::not_null<typename SubdomainDataType::BoundaryDataType*>
                subdomain_boundary_data) {
          *subdomain_boundary_data =
              std::move(subdomain_solution.boundary_data);
        });

    Parallel::printf("%s  Subdomain solution WEIGHTED (central): %s\n",
                     element_index, subdomain_solution.element_data);

    // Apply solution to central element
    db::mutate<fields_tag>(
        make_not_null(&box),
        [&subdomain_solution](
            const gsl::not_null<db::item_type<fields_tag>*> fields) noexcept {
          *fields += subdomain_solution.element_data;
        });

    Parallel::printf("%s  Current solution: %s\n", element_index,
                     get<fields_tag>(box));

    db::mutate<LinearSolver::Tags::HasConverged<OptionsGroup>>(
        make_not_null(&box),
        [](const gsl::not_null<Convergence::HasConverged*> has_converged,
           const size_t& max_iterations, const size_t& iteration_id) noexcept {
          // Run the solver for a set number of iterations
          *has_converged = Convergence::HasConverged{
              {max_iterations, 0., 0.}, iteration_id + 1, 1., 1.};
        },
        get<LinearSolver::Tags::Iterations<OptionsGroup>>(box),
        get<LinearSolver::Tags::IterationId<OptionsGroup>>(box));
    return std::forward_as_tuple(std::move(box));
  }
};

}  // namespace schwarz_detail
}  // namespace LinearSolver
