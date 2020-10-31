// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Protocols.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
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

namespace Poisson {
namespace Solutions {

template <size_t Dim>
class Zero : public tt::ConformsTo<elliptic::protocols::AnalyticSolution> {
 public:
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
  auto variables(const tnsr::I<DataVector, Dim, Frame::Inertial>& x,
                 tmpl::list<Tags::Field> /*meta*/) const noexcept
      -> tuples::TaggedTuple<Tags::Field> {
    return make_with_value<Scalar<DataVector>>(x, 0.);
  }

  auto variables(const tnsr::I<DataVector, Dim, Frame::Inertial>& x,
                 tmpl::list<::Tags::deriv<Tags::Field, tmpl::size_t<Dim>,
                                          Frame::Inertial>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<
          ::Tags::deriv<Tags::Field, tmpl::size_t<Dim>, Frame::Inertial>> {
    return make_with_value<tnsr::i<DataVector, Dim>>(x, 0.);
  }

  auto variables(const tnsr::I<DataVector, Dim, Frame::Inertial>& x,
                 tmpl::list<::Tags::FixedSource<Tags::Field>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<::Tags::FixedSource<Tags::Field>> {
    return make_with_value<Scalar<DataVector>>(x, 0.);
  }
  // @}

  /// Retrieve a collection of variables at coordinates `x`
  template <typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataVector, Dim, Frame::Inertial>& x,
      tmpl::list<Tags...> /*meta*/) const noexcept {
    static_assert(sizeof...(Tags) > 1, "The requested tag is not implemented");
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

}  // namespace Solutions
}  // namespace Poisson
