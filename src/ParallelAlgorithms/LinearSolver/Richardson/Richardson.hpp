// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "NumericalAlgorithms/LinearSolver/InnerProduct.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace LinearSolver {

namespace richardson_detail {

namespace OptionTags {
template <typename OptionsGroup>
struct RelaxationParameter {
  using type = double;
  using group = OptionsGroup;
  static constexpr OptionString help =
      "The weight for the residuum in the scheme";
};
}  // namespace OptionTags

namespace Tags {
template <typename OptionsGroup>
struct RelaxationParameter : db::SimpleTag {
  using type = double;
  static constexpr bool pass_metavariables = false;
  using option_tags = tmpl::list<OptionTags::RelaxationParameter<OptionsGroup>>;
  static double create_from_options(const double value) noexcept {
    return value;
  }
};
}  // namespace Tags

using reduction_data = Parallel::ReductionData<
    // Iteration
    Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
    // Residual
    Parallel::ReductionDatum<double, funcl::Plus<>, funcl::Sqrt<>>>;

template <typename OptionsGroup, typename Metavariables>
void contribute_to_residual_observation(
    const size_t iteration_id, const double residual_magnitude_square,
    Parallel::ConstGlobalCache<Metavariables>& cache) noexcept {
  auto& local_observer =
      *Parallel::get_parallel_component<observers::Observer<Metavariables>>(
           cache)
           .ckLocalBranch();
  Parallel::simple_action<observers::Actions::ContributeReductionData>(
      local_observer,
      observers::ObservationId(
          iteration_id, typename Metavariables::element_observation_type{}),
      std::string{"/" + option_name<OptionsGroup>() + "Residuals"},
      std::vector<std::string>{"Iteration", "Residual"},
      reduction_data{iteration_id, residual_magnitude_square});
}

template <typename FieldsTag, typename OptionsGroup, typename SourceTag>
struct InitializeElement {
  using operator_applied_to_fields_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, FieldsTag>;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, FieldsTag>;
  using residual_magnitude_square_tag =
      db::add_tag_prefix<LinearSolver::Tags::MagnitudeSquare, residual_tag>;
  using initial_residual_magnitude_tag = db::add_tag_prefix<
      LinearSolver::Tags::Initial,
      db::add_tag_prefix<LinearSolver::Tags::Magnitude, residual_tag>>;

  using const_global_cache_tags =
      tmpl::list<LinearSolver::Tags::ConvergenceCriteria<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    return std::make_tuple(
        ::Initialization::merge_into_databox<
            InitializeElement,
            db::AddSimpleTags<LinearSolver::Tags::IterationId<OptionsGroup>,
                              operator_applied_to_fields_tag,
                              residual_magnitude_square_tag,
                              initial_residual_magnitude_tag>,
            db::AddComputeTags<
                LinearSolver::Tags::ResidualCompute<FieldsTag, SourceTag>,
                LinearSolver::Tags::MagnitudeCompute<
                    residual_magnitude_square_tag>,
                LinearSolver::Tags::HasConvergedCompute<FieldsTag,
                                                        OptionsGroup>>>(
            std::move(box), std::numeric_limits<size_t>::max(),
            db::item_type<operator_applied_to_fields_tag>{},
            std::numeric_limits<double>::signaling_NaN(),
            std::numeric_limits<double>::signaling_NaN()));
  }
};

template <typename FieldsTag, typename OptionsGroup, typename SourceTag>
struct PrepareSolve {
  using operator_applied_to_fields_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, FieldsTag>;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, FieldsTag>;
  using residual_magnitude_square_tag =
      db::add_tag_prefix<LinearSolver::Tags::MagnitudeSquare, residual_tag>;
  using initial_residual_magnitude_tag = db::add_tag_prefix<
      LinearSolver::Tags::Initial,
      db::add_tag_prefix<LinearSolver::Tags::Magnitude, residual_tag>>;

  using const_global_cache_tags =
      tmpl::list<LinearSolver::Tags::ConvergenceCriteria<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<LinearSolver::Tags::IterationId<OptionsGroup>,
               residual_magnitude_square_tag, initial_residual_magnitude_tag>(
        make_not_null(&box),
        [](const gsl::not_null<size_t*> iteration_id,
           const gsl::not_null<double*> residual_magnitude_square,
           const gsl::not_null<double*> initial_residual_magnitude,
           const db::const_item_type<residual_tag>& residual) noexcept {
          *iteration_id = std::numeric_limits<size_t>::max();
          *residual_magnitude_square = inner_product(residual, residual);
          *initial_residual_magnitude = sqrt(*residual_magnitude_square);
        },
        get<residual_tag>(box));
    // Observe the initial residual
    contribute_to_residual_observation<OptionsGroup>(
        0, get<residual_magnitude_square_tag>(box), cache);
    return {std::move(box)};
  }
};

template <typename FieldsTag, typename OptionsGroup, typename SourceTag>
struct PrepareStep {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<LinearSolver::Tags::IterationId<OptionsGroup>>(
        make_not_null(&box),
        [](const gsl::not_null<size_t*> iteration_id) noexcept {
          (*iteration_id)++;
        });
    return {std::move(box)};
  }
};

template <typename FieldsTag, typename OptionsGroup, typename SourceTag>
struct PerformStep {
  using operator_applied_to_fields_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, FieldsTag>;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, FieldsTag>;
  using residual_magnitude_square_tag =
      db::add_tag_prefix<LinearSolver::Tags::MagnitudeSquare, residual_tag>;
  using initial_residual_magnitude_tag = db::add_tag_prefix<
      LinearSolver::Tags::Initial,
      db::add_tag_prefix<LinearSolver::Tags::Magnitude, residual_tag>>;

  using const_global_cache_tags =
      tmpl::list<richardson_detail::Tags::RelaxationParameter<OptionsGroup>,
                 LinearSolver::Tags::ConvergenceCriteria<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    // First, update and observe residual magnitude
    db::mutate<residual_magnitude_square_tag>(
        make_not_null(&box),
        [](const gsl::not_null<double*> residual_magnitude_square,
           const db::const_item_type<residual_tag>& residual) noexcept {
          *residual_magnitude_square = inner_product(residual, residual);
        },
        get<residual_tag>(box));
    // The initial residual magnitude is observed in `PrepareSolve`
    if (LIKELY(get<LinearSolver::Tags::IterationId<OptionsGroup>>(box) > 0)) {
      contribute_to_residual_observation<OptionsGroup>(
          get<LinearSolver::Tags::IterationId<OptionsGroup>>(box),
          get<residual_magnitude_square_tag>(box), cache);
    }

    // Second, update the solution fields according to Richardson scheme
    if (not get<LinearSolver::Tags::HasConverged<OptionsGroup>>(box)) {
      db::mutate<FieldsTag>(
          make_not_null(&box),
          [](const gsl::not_null<db::item_type<FieldsTag>*> fields,
             const db::const_item_type<residual_tag>& residual,
             const double relaxation_parameter) noexcept {
            *fields += relaxation_parameter * residual;
          },
          get<residual_tag>(box),
          get<richardson_detail::Tags::RelaxationParameter<OptionsGroup>>(box));
    }
    return {std::move(box)};
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
 *
 * \note The convergence criteria for this linear solver are evaluated on each
 * element individually with no global synchronization, i.e. the algorithm on
 * an element terminates once the residual on that element falls below the
 * prescribed tolerance. Therefore, the residual over all elements once the
 * algorithm is complete is a factor \f$\sqrt{N_\mathrm{elements}}\f$ larger
 * than the prescribed tolerance.
 */
template <typename FieldsTag, typename OptionsGroup,
          typename SourceTag =
              db::add_tag_prefix<::Tags::FixedSource, FieldsTag>>
struct Richardson {
  using fields_tag = FieldsTag;
  using options_group = OptionsGroup;
  using source_tag = SourceTag;
  using operand_tag = fields_tag;
  using component_list = tmpl::list<>;
  using observed_reduction_data_tags = observers::make_reduction_data_tags<
      tmpl::list<richardson_detail::reduction_data>>;
  using initialize_element =
      richardson_detail::InitializeElement<FieldsTag, OptionsGroup, SourceTag>;
  using prepare_solve =
      richardson_detail::PrepareSolve<FieldsTag, OptionsGroup, SourceTag>;
  using prepare_step =
      richardson_detail::PrepareStep<FieldsTag, OptionsGroup, SourceTag>;
  using perform_step =
      richardson_detail::PerformStep<FieldsTag, OptionsGroup, SourceTag>;
};
}  // namespace LinearSolver
