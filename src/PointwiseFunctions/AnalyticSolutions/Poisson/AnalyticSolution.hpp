// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Parallel/CharmPupable.hpp"
#include "Utilities/FakeVirtual.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
// Empty base class for marking analytic solutions.
struct MarkAsAnalyticSolution {};
/// \endcond

namespace Poisson {
namespace Solutions {

namespace Registrars {}

template <size_t Dim, typename LocalRegistrars>
class Solution : public PUP::able {
 protected:
  /// \cond
  Solution() = default;
  Solution(const Solution&) = default;
  Solution(Solution&&) = default;
  Solution& operator=(const Solution&) = default;
  Solution& operator=(Solution&&) = default;
  /// \endcond

 public:
  ~Solution() override = default;

  WRAPPED_PUPable_abstract(Solution);  // NOLINT

  using creatable_classes = Registration::registrants<LocalRegistrars>;

  template <typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataVector, Dim, Frame::Inertial>& x,
      tmpl::list<Tags...> /*meta*/) const noexcept {
    return call_with_dynamic_type<tuples::TaggedTuple<Tags...>,
                                  creatable_classes>(
        this, [&x](const auto* const solution) noexcept {
          return solution->variables_impl(x, tmpl::list<Tags...>{});
        });
  }
};

}  // namespace Solutions
}  // namespace Poisson
