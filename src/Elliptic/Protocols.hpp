// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

namespace elliptic {
/// \ref protocols related to elliptic systems
namespace protocols {

struct AnalyticSolution {
  template <typename ConformingType>
  struct test {};
};

}  // namespace protocols
}  // namespace elliptic
