// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DenseMatrix.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Helpers/ParallelAlgorithms/LinearSolver/DistributedLinearSolverAlgorithmTestHelpers.hpp"
#include "Options/Options.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/Tags/Section.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Tags.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace helpers_distributed = DistributedLinearSolverAlgorithmTestHelpers;

namespace TestHelpers::LinearSolver::multigrid {

namespace OptionTags {
struct LinearOperator {
  static constexpr Options::String help = "The linear operator A to invert.";
  using type =
      std::vector<std::vector<DenseMatrix<double, blaze::columnMajor>>>;
};
}  // namespace OptionTags

struct LinearOperator : db::SimpleTag {
  using type =
      std::vector<std::vector<DenseMatrix<double, blaze::columnMajor>>>;
  using option_tags = tmpl::list<OptionTags::LinearOperator>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& linear_operator) noexcept {
    return linear_operator;
  }
};

using fields_tag = helpers_distributed::fields_tag;
using sources_tag = helpers_distributed::sources_tag;

struct InitializeElement {
  using const_global_cache_tags = tmpl::list<helpers_distributed::Source>;
  using simple_tags = tmpl::list<fields_tag, sources_tag>;
  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<1>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const size_t element_index = helpers_distributed::get_index(element_id);
    const int mg_lev = element_id.segment_ids()[0].refinement_level();
    const auto& source =
        mg_lev == 1 ? typename sources_tag::type(gsl::at(
                          get<helpers_distributed::Source>(box), element_index))
                    : typename sources_tag::type{};
    const size_t num_points =
        db::get<::domain::Tags::Mesh<1>>(box).number_of_grid_points();
    auto initial_fields = typename fields_tag::type{num_points, 0.};

    Initialization::mutate_assign<simple_tags>(
        make_not_null(&box), std::move(initial_fields), std::move(source));
    return {std::move(box)};
  }
};

template <typename OperandTag>
struct CollectOperatorAction;

template <typename OperandTag>
struct ComputeOperatorAction {
  using const_global_cache_tags = tmpl::list<LinearOperator>;
  using local_operator_applied_to_operand_tag =
      db::add_tag_prefix<::LinearSolver::Tags::OperatorAppliedTo, OperandTag>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<1>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const size_t mg_lev =
        db::get<::LinearSolver::multigrid::Tags::MultigridLevel>(box);
    const size_t element_index = helpers_distributed::get_index(element_id);
    const auto& operator_matrices = get<LinearOperator>(box)[mg_lev];
    const auto number_of_elements = operator_matrices.size();
    const auto& linear_operator = gsl::at(operator_matrices, element_index);
    const auto number_of_grid_points = linear_operator.columns();
    const auto& operand = get<OperandTag>(box);

    typename OperandTag::type operator_applied_to_operand{
        number_of_grid_points * number_of_elements};
    dgemv_('N', linear_operator.rows(), linear_operator.columns(), 1,
           linear_operator.data(), linear_operator.spacing(), operand.data(), 1,
           0, operator_applied_to_operand.data(), 1);

    auto& section = *db::get<Parallel::Tags::Section<
        ParallelComponent, ::LinearSolver::multigrid::Tags::MultigridLevel>>(
        box);
    Parallel::contribute_to_reduction<CollectOperatorAction<OperandTag>>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<typename OperandTag::type, funcl::Plus<>>,
            Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>>{
            operator_applied_to_operand, mg_lev},
        section.proxy()[element_id], section.proxy(), section);

    // Terminate algorithm for now. The reduction will be broadcast to the
    // next action which is responsible for restarting the algorithm.
    return {std::move(box), true};
  }
};

template <typename OperandTag>
struct CollectOperatorAction {
  using local_operator_applied_to_operand_tag =
      db::add_tag_prefix<::LinearSolver::Tags::OperatorAppliedTo, OperandTag>;

  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ScalarFieldOperandTag,
            Requires<tmpl::list_contains_v<
                DbTagsList, local_operator_applied_to_operand_tag>> = nullptr>
  static void apply(
      db::DataBox<DbTagsList>& box, Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<1>& element_id,
      const Variables<tmpl::list<ScalarFieldOperandTag>>& Ap_global_data,
      const size_t broadcasting_mg_lev) noexcept {
    // FIXME: We're receiving broadcasts also from reductions over other
    // sections for some reason
    const size_t mg_lev =
        db::get<::LinearSolver::multigrid::Tags::MultigridLevel>(box);
    if (mg_lev != broadcasting_mg_lev) {
      // Parallel::printf("Received broadcast on other MG level, ignoring.\n");
      return;
    }
    const size_t element_index = helpers_distributed::get_index(element_id);
    // This could be generalized to work on the Variables instead of the
    // Scalar, but it's only for the purpose of this test.
    const auto number_of_grid_points =
        get<LinearOperator>(box)[mg_lev][0].columns();
    const auto& Ap_global = get<ScalarFieldOperandTag>(Ap_global_data).get();
    DataVector Ap_local{number_of_grid_points};
    std::copy(Ap_global.begin() +
                  static_cast<int>(element_index * number_of_grid_points),
              Ap_global.begin() +
                  static_cast<int>((element_index + 1) * number_of_grid_points),
              Ap_local.begin());
    db::mutate<local_operator_applied_to_operand_tag>(
        make_not_null(&box),
        [&Ap_local, &number_of_grid_points](auto Ap) noexcept {
          *Ap = typename local_operator_applied_to_operand_tag::type{
              number_of_grid_points};
          get(get<
              ::LinearSolver::Tags::OperatorAppliedTo<ScalarFieldOperandTag>>(
              *Ap)) = Ap_local;
        });
    // Proceed with algorithm
    Parallel::get_parallel_component<ParallelComponent>(cache)[element_id]
        .perform_algorithm(true);
  }
};

}  // namespace TestHelpers::LinearSolver::multigrid
