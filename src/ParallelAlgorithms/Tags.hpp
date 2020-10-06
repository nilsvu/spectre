// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "NumericalAlgorithms/Convergence/Criteria.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "Options/Options.hpp"
#include "Utilities/PrettyType.hpp"

namespace Parallel {
namespace OptionTags {

template <typename OptionsGroup>
struct ConvergenceCriteria {
  static constexpr Options::String help =
      "Determine convergence of the algorithm";
  using type = Convergence::Criteria;
  using group = OptionsGroup;
};

template <typename OptionsGroup>
struct Iterations {
  static constexpr Options::String help =
      "Number of iterations to run the algorithm";
  using type = size_t;
  using group = OptionsGroup;
};

}  // namespace OptionTags

namespace Tags {

/// `Convergence::Criteria` that determine the algorithm has converged
template <typename OptionsGroup>
struct ConvergenceCriteria : db::SimpleTag {
  static std::string name() noexcept {
    return "ConvergenceCriteria(" + Options::name<OptionsGroup>() + ")";
  }
  using type = Convergence::Criteria;

  using option_tags = tmpl::list<OptionTags::ConvergenceCriteria<OptionsGroup>>;
  static constexpr bool pass_metavariables = false;
  static Convergence::Criteria create_from_options(
      const Convergence::Criteria& convergence_criteria) noexcept {
    return convergence_criteria;
  }
};

/// A fixed number of iterations to run the parallel algorithm
template <typename OptionsGroup>
struct Iterations : db::SimpleTag {
  static std::string name() noexcept {
    return "Iterations(" + Options::name<OptionsGroup>() + ")";
  }
  using type = size_t;

  static constexpr bool pass_metavariables = false;
  using option_tags = tmpl::list<OptionTags::Iterations<OptionsGroup>>;
  static size_t create_from_options(const size_t max_iterations) noexcept {
    return max_iterations;
  }
};

/*!
 * \brief Holds an `IterationId` that identifies a step in a parallel algorithm
 */
template <typename Label>
struct IterationId : db::SimpleTag {
  static std::string name() noexcept {
    return "IterationId(" + pretty_type::short_name<Label>() + ")";
  }
  using type = size_t;
};

/*!
 * \brief Holds a `Convergence::HasConverged` flag that signals the parallel
 * algorithm has converged, along with the reason for convergence.
 */
template <typename Label>
struct HasConverged : db::SimpleTag {
  static std::string name() noexcept {
    return "HasConverged(" + pretty_type::short_name<Label>() + ")";
  }
  using type = Convergence::HasConverged;
};

}  // namespace Tags
}  // namespace Parallel
