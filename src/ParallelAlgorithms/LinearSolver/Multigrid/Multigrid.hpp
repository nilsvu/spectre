// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/Matrix.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "ErrorHandling/Assert.hpp"
#include "IO/Observer/Helpers.hpp"
#include "Informer/LoggerComponent.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "ParallelAlgorithms/LinearSolver/AsynchronousSolvers/ElementActions.hpp"
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

/*!
 * \brief Multigrid solver
 *
 * - At the end of the `PreSmootherActions` and the `PostSmootherActions` the
 * operator applied to the fields must be up-to-date.
 */
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

  using component_list = tmpl::list<logging::Logger<Metavariables>>;

  using observed_reduction_data_tags = observers::make_reduction_data_tags<
      tmpl::list<async_solvers::reduction_data>>;

  using initialize_element = tmpl::list<
      async_solvers::InitializeElement<FieldsTag, OptionsGroup, SourceTag>,
      detail::InitializeElement<Metavariables::volume_dim, FieldsTag,
                                OptionsGroup, SourceTag>>;

  using register_element =
      async_solvers::RegisterElement<FieldsTag, OptionsGroup, SourceTag,
                                     Tags::IsFinestLevel>;

  template <typename PreSmootherActions, typename PostSmootherActions,
            typename Label = OptionsGroup>
  using solve = tmpl::list<
      detail::PrepareSolve<FieldsTag, OptionsGroup, SourceTag>,
      async_solvers::PrepareSolve<FieldsTag, OptionsGroup, SourceTag, Label,
                                  Tags::IsFinestLevel, false>,
      detail::RestrictResidualFromChildren<Metavariables::volume_dim, FieldsTag,
                                           OptionsGroup, SourceTag>,
      PreSmootherActions,
      detail::SkipPostsmoothingAtBottom<FieldsTag, OptionsGroup, SourceTag>,
      detail::SendResidualToParent<FieldsTag, OptionsGroup, SourceTag>,
      detail::ProlongateCorrectionFromParent<
          Metavariables::volume_dim, FieldsTag, OptionsGroup, SourceTag>,
      PostSmootherActions,
      detail::SendCorrectionToChildren<FieldsTag, OptionsGroup, SourceTag>,
      async_solvers::CompleteStep<FieldsTag, OptionsGroup, SourceTag, Label,
                                  Tags::IsFinestLevel, false>>;
};

}  // namespace LinearSolver::multigrid
