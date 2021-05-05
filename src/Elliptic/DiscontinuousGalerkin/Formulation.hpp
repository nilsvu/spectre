// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <iosfwd>

/// \cond
namespace Options {
class Option;
template <typename T>
struct create_from_yaml;
}  // namespace Options
/// \endcond

namespace elliptic::dg {

/// The DG formulation
enum class Formulation {
  /// Both auxiliary and primal equations use the "strong" formulation
  Strong,
  /// The auxiliary equations use the "strong" formulation and the primal
  /// equations use the "weak" formulation. This helps make the DG operator
  /// symmetric.
  StrongWeak
};

std::ostream& operator<<(std::ostream& os, Formulation t) noexcept;
}  // namespace elliptic::dg

/// \cond
template <>
struct Options::create_from_yaml<elliptic::dg::Formulation> {
  template <typename Metavariables>
  static elliptic::dg::Formulation create(const Options::Option& options) {
    return create<void>(options);
  }
};

template <>
elliptic::dg::Formulation
Options::create_from_yaml<elliptic::dg::Formulation>::create<void>(
    const Options::Option& options);
/// \endcond
