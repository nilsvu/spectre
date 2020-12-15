// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/Tensor.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/FakeVirtual.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
// Empty base class for marking analytic data.
struct MarkAsAnalyticData {};
/// \endcond

/// Provides analytic tensor data as a function of the spatial coordinates
template <size_t Dim, typename Registrars>
class AnalyticData : public PUP::able {
 public:
  static constexpr size_t volume_dim = Dim;
  using registrars = Registrars;

 protected:
  /// \cond
  AnalyticData() = default;
  AnalyticData(const AnalyticData&) = default;
  AnalyticData(AnalyticData&&) = default;
  AnalyticData& operator=(const AnalyticData&) = default;
  AnalyticData& operator=(AnalyticData&&) = default;
  /// \endcond

 public:
  ~AnalyticData() override = default;

  explicit AnalyticData(CkMigrateMessage* m) noexcept : PUP::able(m) {}
  WRAPPED_PUPable_abstract(AnalyticData);  // NOLINT

  using creatable_classes = Registration::registrants<registrars>;

  /// Retrieve a collection of tensor fields at spatial coordinate(s) `x`
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataType, Dim>& x,
                                         tmpl::list<Tags...> /*meta*/) const
      noexcept {
    return call_with_dynamic_type<tuples::TaggedTuple<Tags...>,
                                  creatable_classes>(
        this, [&x](auto* const derived) noexcept {
          return derived->variables(x, tmpl::list<Tags...>{});
        });
  }
};
