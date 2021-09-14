// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "IO/Observer/Helpers.hpp"
#include "ParallelAlgorithms/LinearSolver/AsynchronousSolvers/ElementActions.hpp"
#include "ParallelAlgorithms/LinearSolver/Richardson/Tags.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
namespace Parallel {
template <typename Metavariables>
struct GlobalCache;
}  // namespace Parallel
/// \endcond

/// Items related to the %Richardson linear solver
///
/// \see `LinearSolver::Richardson::Richardson`
namespace LinearSolver::Richardson {

namespace detail {

template <typename FieldsTag, typename OptionsGroup, typename SourceTag>
struct UpdateFields {
 private:
  using residual_tag = LinearSolver::Tags::Residual<FieldsTag>;

 public:
  using const_global_cache_tags =
      tmpl::list<Tags::RelaxationParameter<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    // Update the solution fields according to the Richardson scheme
    db::mutate<FieldsTag>(
        make_not_null(&box),
        [](const auto fields, const auto& residual,
           const double relaxation_parameter) noexcept {
          *fields += relaxation_parameter * residual;
        },
        get<residual_tag>(box),
        get<Tags::RelaxationParameter<OptionsGroup>>(box));
    return {std::move(box)};
  }
};

}  // namespace detail

/*!
 * \ingroup LinearSolverGroup
 * \brief A simple %Richardson scheme for solving a system of linear equations
 * \f$Ax=b\f$
 *
 * \warning This linear solver is useful only for basic preconditioning of
 * another linear solver or for testing purposes. See
 * `LinearSolver::cg::ConjugateGradient` or `LinearSolver::gmres::Gmres` for
 * more useful general-purpose linear solvers.
 *
 * In each step the solution is updated from its initial state \f$x_0\f$ as
 *
 * \f[
 * x_{k+1} = x_k + \omega \left(b - Ax\right)
 * \f]
 *
 * where \f$\omega\f$ is a _relaxation parameter_ that weights the residual.
 *
 * The scheme converges if the spectral radius (i.e. the largest absolute
 * eigenvalue) of the iteration operator \f$G=1-\omega A\f$ is smaller than one.
 * For symmetric positive definite (SPD) matrices \f$A\f$ with largest
 * eigenvalue \f$\lambda_\mathrm{max}\f$ and smallest eigenvalue
 * \f$\lambda_\mathrm{min}\f$ choose
 *
 * \f[
 * \omega_\mathrm{SPD,optimal} = \frac{2}{\lambda_\mathrm{max} +
 * \lambda_\mathrm{min}}
 * \f]
 *
 * for optimal convergence.
 *
 * \par Array sections
 * This linear solver requires no synchronization between elements, so it runs
 * on all elements in the array parallel component. Partitioning of the elements
 * in sections is only relevant for observing residual norms. Pass the section
 * ID tag for the `ArraySectionIdTag` template parameter if residual norms
 * should be computed over a section. Pass `void` (default) to compute residual
 * norms over all elements in the array.
 */
template <typename FieldsTag, typename OptionsGroup,
          typename SourceTag = ::Tags::FixedSource<FieldsTag>,
          typename ArraySectionIdTag = void>
struct Richardson {
  using fields_tag = FieldsTag;
  using options_group = OptionsGroup;
  using source_tag = SourceTag;
  using operand_tag = fields_tag;
  using component_list = tmpl::list<>;
  using observed_reduction_data_tags = observers::make_reduction_data_tags<
      tmpl::list<async_solvers::reduction_data>>;
  using initialize_element =
      async_solvers::InitializeElement<FieldsTag, OptionsGroup, SourceTag>;
  using register_element =
      async_solvers::RegisterElement<FieldsTag, OptionsGroup, SourceTag,
                                     ArraySectionIdTag>;
  template <typename ApplyOperatorActions, typename Label = OptionsGroup>
  using solve =
      tmpl::list<async_solvers::PrepareSolve<FieldsTag, OptionsGroup, SourceTag,
                                             Label, ArraySectionIdTag>,
                 detail::UpdateFields<FieldsTag, OptionsGroup, SourceTag>,
                 ApplyOperatorActions,
                 async_solvers::CompleteStep<FieldsTag, OptionsGroup, SourceTag,
                                             Label, ArraySectionIdTag>>;
};
}  // namespace LinearSolver::Richardson
