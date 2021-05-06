// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "IO/Observer/Actions/RegisterWithObservers.hpp"
#include "IO/Observer/Helpers.hpp"
#include "ParallelAlgorithms/LinearSolver/AsynchronousSolvers/ElementActions.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/ElementActions.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/ObserveVolumeData.hpp"
#include "Utilities/TMPL.hpp"

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
template <size_t Dim, typename FieldsTag, typename OptionsGroup,
          typename ResidualIsMassiveTag,
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

  using observed_reduction_data_tags = observers::make_reduction_data_tags<
      tmpl::list<async_solvers::reduction_data>>;

  using initialize_element = tmpl::list<
      async_solvers::InitializeElement<FieldsTag, OptionsGroup, SourceTag>,
      detail::InitializeElement<Dim, FieldsTag, OptionsGroup, SourceTag>>;

  using register_element =
      tmpl::list<async_solvers::RegisterElement<FieldsTag, OptionsGroup,
                                                SourceTag, Tags::IsFinestGrid>,
                 observers::Actions::RegisterWithObservers<
                     detail::RegisterWithVolumeObserver<OptionsGroup>>>;

  template <typename PreSmootherActions, typename PostSmootherActions,
            typename Label = OptionsGroup>
  using solve = tmpl::list<
      detail::PrepareSolve<FieldsTag, OptionsGroup, SourceTag>,
      async_solvers::PrepareSolve<FieldsTag, OptionsGroup, SourceTag, Label,
                                  Tags::IsFinestGrid, false>,
      detail::ReceiveResidualFromFinerGrid<Dim, FieldsTag, OptionsGroup,
                                           SourceTag>,
      detail::PreparePreSmoothing<FieldsTag, OptionsGroup, SourceTag>,
      PreSmootherActions,
      detail::SkipPostsmoothingAtBottom<FieldsTag, OptionsGroup, SourceTag>,
      detail::SendResidualToCoarserGrid<FieldsTag, OptionsGroup,
                                        ResidualIsMassiveTag, SourceTag>,
      detail::ReceiveCorrectionFromCoarserGrid<Dim, FieldsTag, OptionsGroup,
                                               SourceTag>,
      PostSmootherActions,
      detail::SendCorrectionToFinerGrid<FieldsTag, OptionsGroup, SourceTag>,
      detail::ObserveVolumeData<FieldsTag, OptionsGroup, SourceTag>,
      async_solvers::CompleteStep<FieldsTag, OptionsGroup, SourceTag, Label,
                                  Tags::IsFinestGrid, false>>;
};

}  // namespace LinearSolver::multigrid
