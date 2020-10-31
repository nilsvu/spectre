// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Protocols.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"
#include "Options/Options.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
class DataVector;
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Elasticity::Solutions {
template <size_t Dim>
class Zero : public tt::ConformsTo<elliptic::protocols::AnalyticSolution> {
 public:
  static constexpr size_t volume_dim = Dim;

  using options = tmpl::list<>;
  static constexpr Options::String help{
      "The trivial solution, useful as initial guess."};

  Zero() = default;
  Zero(const Zero&) noexcept = default;
  Zero& operator=(const Zero&) noexcept = default;
  Zero(Zero&&) noexcept = default;
  Zero& operator=(Zero&&) noexcept = default;
  ~Zero() noexcept = default;

  // @{
  /// Retrieve variable at coordinates `x`
  auto variables(const tnsr::I<DataVector, Dim>& x,
                 tmpl::list<Tags::Displacement<Dim>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<Tags::Displacement<Dim>> {
    return make_with_value<tnsr::I<DataVector, Dim>>(x, 0.);
  }

  auto variables(const tnsr::I<DataVector, Dim>& x,
                 tmpl::list<Tags::Strain<Dim>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<Tags::Strain<Dim>> {
    return make_with_value<tnsr::ii<DataVector, Dim>>(x, 0.);
  }

  static auto variables(
      const tnsr::I<DataVector, Dim>& x,
      tmpl::list<
          ::Tags::FixedSource<Tags::Displacement<Dim>>> /*meta*/) noexcept
      -> tuples::TaggedTuple<::Tags::FixedSource<Tags::Displacement<Dim>>> {
    return make_with_value<tnsr::I<DataVector, Dim>>(x, 0.);
  }
  // @}

  /// Retrieve a collection of variables at coordinates `x`
  template <typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataVector, Dim>& x,
                                         tmpl::list<Tags...> /*meta*/) const
      noexcept {
    static_assert(sizeof...(Tags) > 1, "The requested tag is not implemented.");
    return {tuples::get<Tags>(variables(x, tmpl::list<Tags>{}))...};
  }

  // clang-tidy: no pass by reference
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

template <size_t Dim>
bool operator==(const Zero<Dim>& /*lhs*/, const Zero<Dim>& /*rhs*/) noexcept {
  return true;
}

template <size_t Dim>
bool operator!=(const Zero<Dim>& /*lhs*/, const Zero<Dim>& /*rhs*/) noexcept {
  return false;
}

}  // namespace Elasticity::Solutions
