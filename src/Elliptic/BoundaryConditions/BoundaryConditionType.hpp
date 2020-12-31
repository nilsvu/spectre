// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <ostream>

/// \cond
namespace Options {
class Option;
template <typename T>
struct create_from_yaml;
}  // namespace Options
/// \endcond

namespace elliptic {

enum class BoundaryConditionType { Dirichlet, Neumann };

std::ostream& operator<<(
    std::ostream& os,
    const BoundaryConditionType boundary_condition_type) noexcept;

}  // namespace elliptic

/// \cond
template <>
struct Options::create_from_yaml<elliptic::BoundaryConditionType> {
  template <typename Metavariables>
  static elliptic::BoundaryConditionType create(
      const Options::Option& options) {
    return create<void>(options);
  }
};

template <>
elliptic::BoundaryConditionType
Options::create_from_yaml<elliptic::BoundaryConditionType>::create<void>(
    const Options::Option& options);
/// \endcond
