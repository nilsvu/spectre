// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/DiscontinuousGalerkin/Formulation.hpp"

#include <ostream>
#include <string>

#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace elliptic::dg {
std::ostream& operator<<(std::ostream& os, const Formulation t) noexcept {
  switch (t) {
    case Formulation::Strong:
      return os << "Strong";
    case Formulation::StrongWeak:
      return os << "StrongWeak";
    default:
      ERROR("Unknown DG formulation.");
  }
}
}  // namespace elliptic::dg

template <>
elliptic::dg::Formulation
Options::create_from_yaml<elliptic::dg::Formulation>::create<void>(
    const Options::Option& options) {
  const auto type_read = options.parse_as<std::string>();
  if ("Strong" == type_read) {
    return elliptic::dg::Formulation::Strong;
  } else if ("StrongWeak" == type_read) {
    return elliptic::dg::Formulation::StrongWeak;
  }
  PARSE_ERROR(options.context(),
              "Failed to convert '"
                  << type_read
                  << "' to elliptic::dg::Formulation. Must be one "
                     "of 'Strong' or 'StrongWeak'.");
}
