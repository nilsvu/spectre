// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

namespace elliptic {
/// \ref protocols "Protocols" related to elliptic systems
namespace protocols {

struct AnalyticSolution {
  template <typename ConformingType>
  struct test {
    // Only using this protocol to "mark" classes as analytic solutions for now,
    // without checking they fulfill a particular compile-time interface. We
    // plan to make the analytic solutions factory-creatable anyway, so this
    // protocol will convert to an abstract base class.
  };
};

}  // namespace protocols
}  // namespace elliptic
