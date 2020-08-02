// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
class DataVector;
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace elliptic {

template <typename SolutionType>
class InitialGuessFromSolution : public SolutionType {
 public:
  static constexpr size_t volume_dim = SolutionType::volume_dim;

  using SolutionType::SolutionType;

  using options = typename SolutionType::options;
  static constexpr Options::String help = SolutionType::help;
  static std::string name() noexcept { return Options::name<SolutionType>(); }

  template <typename... Tags>
  tuples::TaggedTuple<::Tags::Initial<Tags>...> variables(
      const tnsr::I<DataVector, 3>& x,
      tmpl::list<::Tags::Initial<Tags>...> /*meta*/) const noexcept {
    auto vars = SolutionType::variables(x, tmpl::list<Tags...>{});
    return {std::move(get<Tags>(vars))...};
  }

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept { SolutionType::pup(p); }  // NOLINT
};

template <typename SolutionType>
inline constexpr bool operator==(
    const InitialGuessFromSolution<SolutionType>& lhs,
    const InitialGuessFromSolution<SolutionType>& rhs) noexcept {
  return dynamic_cast<const SolutionType&>(lhs) ==
         dynamic_cast<const SolutionType&>(rhs);
}

template <typename SolutionType>
inline constexpr bool operator!=(
    const InitialGuessFromSolution<SolutionType>& lhs,
    const InitialGuessFromSolution<SolutionType>& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace elliptic
