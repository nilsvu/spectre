// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

#include "DataStructures/Matrix.hpp"
#include "Domain/ElementId.hpp"
#include "ErrorHandling/Assert.hpp"
#include "IO/Observer/Helpers.hpp"
#include "NumericalAlgorithms/LinearOperators/ApplyMatrices.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/ElementActions.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Tags.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/TMPL.hpp"

#include "Parallel/Printf.hpp"
#include "Utilities/GetOutput.hpp"

/// Items related to the multigrid linear solver
///
/// \see `LinearSolver::multigrid::Multigrid`
namespace LinearSolver::multigrid {

struct VcycleDownLabel {};
struct VcycleUpLabel {};

template <typename Metavariables, typename FieldsTag, typename OptionsGroup,
          typename SourceTag =
              db::add_tag_prefix<::Tags::FixedSource, FieldsTag>>
struct Multigrid {
  using fields_tag = FieldsTag;
  using options_group = OptionsGroup;
  using source_tag = SourceTag;

  using operand_tag = FieldsTag;

  using smooth_source_tag = source_tag;
  using smooth_fields_tag = fields_tag;

  using component_list = tmpl::list<>;

  using observed_reduction_data_tags = tmpl::list<>;

  using initialize_element =
      detail::InitializeElement<Metavariables::volume_dim, FieldsTag,
                                OptionsGroup, SourceTag>;

  using register_element = tmpl::list<>;

  using prepare_solve =
      detail::PrepareSolve<FieldsTag, OptionsGroup, SourceTag>;

  // - Wait for interpolated residual from finer grid
  using prepare_step_down =
      detail::RestrictResidualFromChildren<Metavariables::volume_dim, FieldsTag,
                                           OptionsGroup, SourceTag>;

  // - Run smoother
  // - Apply operator

  // - Interpolate to coarser grid
  using perform_step_down = tmpl::list<
      detail::SkipPostsmoothingAtBottom<FieldsTag, OptionsGroup, SourceTag>,
      detail::SendResidualToParent<FieldsTag, OptionsGroup, SourceTag>>;

  // - Wait for interpolated correction from coarser grid
  using prepare_step_up = detail::ProlongateCorrectionFromParent<
      Metavariables::volume_dim, FieldsTag, OptionsGroup, SourceTag>;

  // - Run smoother
  // - Apply operator

  // - Interpolate to finer grid
  using perform_step_up = tmpl::list<
      detail::SendCorrectionToChildren<FieldsTag, OptionsGroup, SourceTag>,
      detail::CompleteStep<FieldsTag, OptionsGroup, SourceTag>>;
};

}  // namespace LinearSolver::multigrid
