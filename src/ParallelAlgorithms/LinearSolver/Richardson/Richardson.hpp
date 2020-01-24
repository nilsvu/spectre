// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "IO/Observer/Helpers.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/TMPL.hpp"

#include "Parallel/Printf.hpp"

namespace LinearSolver {

namespace richardson_detail {
struct DoNothing {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    return {std::move(box), false};
  }
};
}  // namespace richardson_detail

/*!
 * \ingroup LinearSolverGroup
 * \brief A simple Richardson scheme for solving a system of linear equations
 * \f$Ax=b\f$
 *
 * \warning This linear solver is useful only for basic preconditioning of
 * another linear solve or for testing purposes. See `ConjugateGradient` or
 * `Gmres` for more useful general-purpose linear solvers.
 *
 * In each step the solution is updated from its initial state \f$x_0\f$ as
 *
 * \f[
 * x_{k+1} = x_k + \omega \left(b - Ax\right)
 * \f]
 *
 * where \f$omega\f$ is a _relaxation parameter_ that weights the residuum.
 *
 * The scheme converges if the spectral radius (i.e. the largest absolute
 * eigenvalue) of the iteration operator \f$G=1-\omega A\f$ is smaller than one.
 * For symmetric positive definite (SPD) matrices \f$A\f$ with largest
 * eigenvalue \f$\lambda_\mathrm{max}\f$ and smallest eigenvalue
 * \f$\lambda_\mathrm{min}\f$ choose
 *
 * \f[
 * \omega_mathrm{SPD,optimal} = \frac{2}{\lambda_\mathrm{max} +
 * \lambda_\mathrm{min}}
 * \f]
 *
 * for optimal convergence.
 */
template <typename FieldsTag, typename OptionsGroup,
          typename SourceTag =
              db::add_tag_prefix<::Tags::FixedSource, FieldsTag>>
struct Richardson {
 public:
  using fields_tag = FieldsTag;
  using operand_tag = fields_tag;
  using source_tag = SourceTag;
  using options_group = OptionsGroup;
  using component_list = tmpl::list<>;
  using observed_reduction_data_tags =
      observers::make_reduction_data_tags<tmpl::list<>>;

 private:
  using operator_applied_to_fields_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, fields_tag>;

  struct RelaxationParameter : db::SimpleTag {
    using type = double;
    using group = OptionsGroup;
    static constexpr OptionString help =
        "The weight for the residuum in the scheme";
    using option_tags = tmpl::list<RelaxationParameter>;
    template <typename Metavariables>
    static type create_from_options(const type& value) noexcept {
      return value;
    }
  };
  struct Iterations : db::SimpleTag {
    using type = size_t;
    using group = OptionsGroup;
    static constexpr OptionString help = "Number of iterations to run";
    using option_tags = tmpl::list<Iterations>;
    template <typename Metavariables>
    static type create_from_options(const type& value) noexcept {
      return value;
    }
  };

 public:
  struct initialize_element {
    template <typename DbTagsList, typename... InboxTags,
              typename Metavariables, typename ArrayIndex, typename ActionList,
              typename ParallelComponent>
    static auto apply(db::DataBox<DbTagsList>& box,
                      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                      Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                      const ArrayIndex& /*array_index*/,
                      const ActionList /*meta*/,
                      const ParallelComponent* const /*meta*/) noexcept {
      Parallel::printf("  Init richardson\n");
      using compute_tags = db::AddComputeTags<>;
      return std::make_tuple(
          ::Initialization::merge_into_databox<
              initialize_element,
              db::AddSimpleTags<LinearSolver::Tags::IterationId<OptionsGroup>,
                                LinearSolver::Tags::HasConverged<OptionsGroup>>,
              compute_tags>(std::move(box), std::numeric_limits<size_t>::max(),
                            Convergence::HasConverged{}));
    }
  };

  struct prepare_solve {
    using const_global_cache_tags = tmpl::list<Iterations>;

    template <typename DbTagsList, typename... InboxTags,
              typename Metavariables, typename ArrayIndex, typename ActionList,
              typename ParallelComponent>
    static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
        db::DataBox<DbTagsList>& box,
        const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
        Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
        const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
        const ParallelComponent* const /*meta*/) noexcept {
      Parallel::printf("  Prepare richardson\n");
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
          get<Iterations>(box));
      return {std::move(box), false};
    }
  };

  struct prepare_step {
    template <typename DbTagsList, typename... InboxTags,
              typename Metavariables, typename ArrayIndex, typename ActionList,
              typename ParallelComponent>
    static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
        db::DataBox<DbTagsList>& box,
        const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
        Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
        const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
        const ParallelComponent* const /*meta*/) noexcept {
      Parallel::printf(
          "  Prep richardson step %zu\n",
          get<LinearSolver::Tags::IterationId<OptionsGroup>>(box) + 1);
      db::mutate<LinearSolver::Tags::IterationId<OptionsGroup>>(
          make_not_null(&box),
          [](const gsl::not_null<size_t*> iteration_id) noexcept {
            (*iteration_id)++;
          });
      return {std::move(box), false};
    }
  };

  struct perform_step {
    using const_global_cache_tags = tmpl::list<RelaxationParameter, Iterations>;

    template <typename DbTagsList, typename... InboxTags,
              typename Metavariables, typename ArrayIndex, typename ActionList,
              typename ParallelComponent>
    static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
        db::DataBox<DbTagsList>& box,
        const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
        const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
        const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
        const ParallelComponent* const /*meta*/) noexcept {
      Parallel::printf("  Perform richardson step\n");
      {
        const auto r =
            get<source_tag>(box) - get<operator_applied_to_fields_tag>(box);
        const double res = inner_product(r, r);
        Parallel::printf("    Starting residual: %e\n", res);
      }
      db::mutate<fields_tag, LinearSolver::Tags::HasConverged<OptionsGroup>>(
          make_not_null(&box),
          [](const gsl::not_null<db::item_type<fields_tag>*> fields,
             const gsl::not_null<Convergence::HasConverged*> has_converged,
             const size_t& iteration_id,
             const db::item_type<operator_applied_to_fields_tag>&
                 operator_applied_to_fields,
             const db::item_type<source_tag>& source,
             const double& relaxation_parameter,
             const size_t& max_iterations) noexcept {
            *fields +=
                relaxation_parameter * (source - operator_applied_to_fields);
            *has_converged = Convergence::HasConverged{
                {max_iterations, 0., 0.}, iteration_id + 1, 1., 1.};
          },
          get<LinearSolver::Tags::IterationId<OptionsGroup>>(box),
          get<operator_applied_to_fields_tag>(box), get<source_tag>(box),
          get<RelaxationParameter>(box), get<Iterations>(box));
      return {std::move(box), false};
    }
  };
};
}  // namespace LinearSolver
