// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "Options/Options.hpp"
#include "Utilities/PrettyType.hpp"

/// Functionality related to solving elliptic partial differential equations
namespace elliptic {

namespace OptionTags {

template <typename BackgroundType>
struct Background {
  static std::string name() noexcept { return Options::name<BackgroundType>(); }
  static constexpr Options::String help = "Options for the background";
  using type = BackgroundType;
};

struct InitialGuessGroup {
  static std::string name() noexcept { return "InitialGuess"; }
  static constexpr Options::String help =
      "Initial guess for the elliptic solve";
};

template <typename InitialGuessType>
struct InitialGuess {
  static std::string name() noexcept {
    return Options::name<InitialGuessType>();
  }
  static constexpr Options::String help = "Options for the initial guess";
  using group = InitialGuessGroup;
  using type = InitialGuessType;
};

}  // namespace OptionTags

namespace Tags {

/*!
 * \brief Holds an object that computes the principal part of the elliptic PDEs.
 *
 * \details The `FluxesComputerType` must have an `apply` function that computes
 * fluxes from the system fields.
 *
 * When placed in the cache, the `FluxesComputerType` is default-constructed.
 * Provide tags that derive from this tag to construct it differently, e.g. to
 * construct it from problem-specific options or retrieve it from an analytic
 * solution.
 */
template <typename FluxesComputerType>
struct FluxesComputer : db::SimpleTag {
  using type = FluxesComputerType;
  static std::string name() noexcept {
    return pretty_type::short_name<FluxesComputerType>();
  }
  using option_tags = tmpl::list<>;

  static constexpr bool pass_metavariables = false;
  static FluxesComputerType create_from_options() {
    return FluxesComputerType{};
  }
};

/// The variable-independent part of the elliptic equations, e.g. the
/// fixed-sources \f$f(x)\f$ in a Poisson equation \f$-\Delta u=f(x)\f$, the
/// matter-density in a TOV-solve or the conformal metric in an XCTS solve.
template <typename BackgroundType>
struct Background : db::SimpleTag {
  using type = BackgroundType;
  using option_tags = tmpl::list<OptionTags::Background<BackgroundType>>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& value) noexcept { return value; }
};

/// The initial guess for the elliptic solve.
template <typename InitialGuessType>
struct InitialGuess : db::SimpleTag {
  using type = InitialGuessType;
  using option_tags = tmpl::list<OptionTags::InitialGuess<InitialGuessType>>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& value) noexcept { return value; }
};

}  // namespace Tags
}  // namespace elliptic
