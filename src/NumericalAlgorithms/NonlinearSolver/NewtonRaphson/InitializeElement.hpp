// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "NumericalAlgorithms/LinearSolver/InnerProduct.hpp"
#include "NumericalAlgorithms/NonlinearSolver/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "Utilities/MakeWithValue.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
namespace LinearSolver {
namespace newton_raphson_detail {
template <typename Metavariables>
struct ResidualMonitor;
template <typename BroadcastTarget>
struct InitializeResidual;
}  // namespace newton_raphson_detail
}  // namespace LinearSolver
/// \endcond

namespace NonlinearSolver {
namespace newton_raphson_detail {

template <typename Metavariables>
struct InitializeElement {
 private:
  using fields_tag = typename Metavariables::system::nonlinear_fields_tag;
  using nonlinear_source_tag = db::add_tag_prefix<::Tags::Source, fields_tag>;
  using nonlinear_operator_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::OperatorAppliedTo, fields_tag>;
  using correction_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::Correction, fields_tag>;
  using linear_source_tag = db::add_tag_prefix<::Tags::Source, correction_tag>;
  using linear_operator_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, correction_tag>;

 public:
  using simple_tags = db::AddSimpleTags<linear_source_tag, linear_operator_tag,
                                        NonlinearSolver::Tags::HasConverged>;
  using compute_tags = db::AddComputeTags<>;

  template <typename TagsList, typename ArrayIndex, typename ParallelComponent>
  static auto initialize(db::DataBox<TagsList>&& box,
                         const Parallel::ConstGlobalCache<Metavariables>& cache,
                         const ArrayIndex& array_index,
                         const ParallelComponent* const /*meta*/) noexcept {
    // Compute nonlinear residual. It sources the linear solve for the
    // correction, so directly store as such in the DataBox.
    auto linear_source = db::item_type<linear_source_tag>(
        get<nonlinear_source_tag>(box) - get<nonlinear_operator_tag>(box));

    // Always start with a zero initial guess for the correction
    db::mutate<correction_tag>(
        make_not_null(&box), [&linear_source](const gsl::not_null<
                                              db::item_type<correction_tag>*>
                                                  correction) noexcept {
          *correction =
              make_with_value<db::item_type<correction_tag>>(linear_source, 0.);
        });

    // Since the correction is zero, so is the linear operator applied to it
    auto linear_operator =
        make_with_value<db::item_type<linear_operator_tag>>(linear_source, 0.);

    // Perform global reduction to compute initial residual magnitude square for
    // residual monitor
    Parallel::contribute_to_reduction<
        newton_raphson_detail::InitializeResidual<ParallelComponent>>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<double, funcl::Plus<>, funcl::Sqrt<>>>{
            LinearSolver::inner_product(linear_source, linear_source)},
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        Parallel::get_parallel_component<ResidualMonitor<Metavariables>>(
            cache));

    return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
        std::move(box), std::move(linear_source), std::move(linear_operator),
        db::item_type<NonlinearSolver::Tags::HasConverged>{});
  }
};

}  // namespace newton_raphson_detail
}  // namespace NonlinearSolver
