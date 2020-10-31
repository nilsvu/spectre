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
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
class DataVector;
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Poisson::Solutions {

/// The trivial solution \f$u=0\f$ of a Poisson equation. Useful as initial
/// guess.
template <size_t Dim>
class Zero : public tt::ConformsTo<elliptic::protocols::AnalyticSolution> {
 public:
  using supported_tags =
      tmpl::list<Tags::Field,
                 ::Tags::deriv<Tags::Field, tmpl::size_t<Dim>, Frame::Inertial>,
                 ::Tags::FixedSource<Tags::Field>>;

  using options = tmpl::list<>;
  static constexpr Options::String help{
      "The trivial solution, useful as initial guess."};

  Zero() = default;
  Zero(const Zero&) noexcept = default;
  Zero& operator=(const Zero&) noexcept = default;
  Zero(Zero&&) noexcept = default;
  Zero& operator=(Zero&&) noexcept = default;
  ~Zero() noexcept = default;

  /// Retrieve a collection of variables at coordinates `x`
  template <typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataVector, Dim, Frame::Inertial>& x,
      tmpl::list<Tags...> /*meta*/) const noexcept {
    static_assert(tmpl::size<tmpl::list_difference<tmpl::list<Tags...>,
                                                   supported_tags>>::value == 0,
                  "The requested tag is not supported");
    return {make_with_value<typename Tags::type>(x, 0.)...};
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

}  // namespace Poisson::Solutions
