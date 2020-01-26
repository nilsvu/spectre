// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"

namespace LinearSolver {
namespace schwarz_detail {

namespace OptionTags {

template <typename SolverType, typename OptionsGroup>
struct SubdomainSolver {
  using type = SolverType;
  using group = OptionsGroup;
  static constexpr OptionString help =
      "Options for the linear solver on subdomains";
};

}  // namespace OptionTags

namespace Tags {

template <typename OptionsGroup>
struct SubdomainSolverBase : db::BaseTag {};

template <typename SolverType, typename OptionsGroup>
struct SubdomainSolver : SubdomainSolverBase<OptionsGroup>, db::SimpleTag {
  using type = SolverType;
};

template <typename SolverType, typename OptionsGroup>
struct SubdomainSolverFromOptions : SubdomainSolver<SolverType, OptionsGroup> {
  static std::string name() noexcept {
    return option_name<OptionsGroup>() + "(SubdomainSolver)";
  }
  using type = SolverType;
  using option_tags =
      tmpl::list<OptionTags::SubdomainSolver<SolverType, OptionsGroup>>;

  template <typename Metavariables>
  static type create_from_options(const type& value) noexcept {
    return value;
  }
};

}  // namespace Tags
}  // namespace schwarz_detail
}  // namespace LinearSolver
