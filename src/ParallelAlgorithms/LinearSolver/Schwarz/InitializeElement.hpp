// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Domain/ElementIndex.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace LinearSolver {
namespace schwarz_detail {

template <typename FieldsTag, typename OptionsGroup, typename SubdomainOperator,
          typename SourceTag>
struct InitializeElement {
 private:
  using fields_tag = FieldsTag;
  using source_tag = SourceTag;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>;
  using SubdomainDataType = typename SubdomainOperator::SubdomainDataType;
  using subdomain_boundary_data_tag =
      Tags::SubdomainBoundaryData<FieldsTag, OptionsGroup, SubdomainOperator>;
  using subdomain_solver_tag =
      Tags::SubdomainSolver<LinearSolver::Serial::Gmres<SubdomainDataType>,
                            OptionsGroup>;

 public:
  using initialization_tags = tmpl::list<subdomain_solver_tag>;
  using initialization_tags_to_keep = tmpl::list<subdomain_solver_tag>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ElementIndex<Dim>& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using compute_tags = db::AddComputeTags<
        LinearSolver::Tags::ResidualCompute<fields_tag, source_tag>,
        ::Tags::NextCompute<LinearSolver::Tags::IterationId<OptionsGroup>>>;
    return std::make_tuple(
        ::Initialization::merge_into_databox<
            InitializeElement,
            db::AddSimpleTags<LinearSolver::Tags::IterationId<OptionsGroup>,
                              LinearSolver::Tags::HasConverged<OptionsGroup>,
                              subdomain_boundary_data_tag>,
            compute_tags>(std::move(box),
                          // The `PrepareSolve` action populates these tags with
                          // initial values
                          std::numeric_limits<size_t>::max(),
                          Convergence::HasConverged{},
                          db::item_type<subdomain_boundary_data_tag>{}));
  }
};

}  // namespace schwarz_detail
}  // namespace LinearSolver
