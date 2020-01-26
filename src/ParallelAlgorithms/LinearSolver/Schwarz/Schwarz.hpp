// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Prefixes.hpp"
#include "IO/Observer/Helpers.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/ElementActions.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/InitializeElement.hpp"
#include "Utilities/TMPL.hpp"

namespace LinearSolver {

/*!
 * \ingroup LinearSolverGroup
 * \brief A Schwarz-type solver for linear systems of equations \f$Ax=b\f$.
 *
 * Solving Ax=b approximately by inverting A element-wise with some overlap:
 * 1. Gather source data with overlap from neighbors
 * 2. Iteratively solve Ax=b restricted to subdomain by applying operator to
 * fields (without communication)
 *   - Possible libraries: Eigen, PETSc
 * 3. Weight result with function and sum over subdomains
 */
template <typename Metavariables, typename FieldsTag, typename OptionsGroup,
          typename SourceTag =
              db::add_tag_prefix<::Tags::FixedSource, FieldsTag>>
struct Schwarz {
  using fields_tag = FieldsTag;
  using source_tag = SourceTag;
  using options_group = OptionsGroup;

  using component_list = tmpl::list<>;
  using observed_reduction_data_tags = tmpl::list<>;

  using initialize_element =
      schwarz_detail::InitializeElement<FieldsTag, OptionsGroup>;

  using prepare_solve = schwarz_detail::PrepareSolve<FieldsTag, OptionsGroup>;

  using prepare_step = schwarz_detail::PrepareStep<FieldsTag, OptionsGroup>;

  using perform_step = schwarz_detail::PerformStep<FieldsTag, OptionsGroup>;
};

}  // namespace LinearSolver
