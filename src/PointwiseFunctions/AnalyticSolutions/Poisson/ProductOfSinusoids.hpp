// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"     // IWYU pragma: keep
#include "Elliptic/Systems/Poisson/Tags.hpp"    // IWYU pragma: keep
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/AnalyticSolution.hpp"
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

/// \cond
template <size_t Dim, typename LocalRegistrars>
class ProductOfSinusoids;
/// \endcond

namespace Registrars {
template <size_t Dim>
struct ProductOfSinusoids {
  template <typename LocalRegistrars>
  using f = Solutions::ProductOfSinusoids<Dim, LocalRegistrars>;
};
}  // namespace Registrars

namespace ProductOfSinusoids_detail {

template <size_t Dim>
Scalar<DataVector> variable(
    Tags::Field /* meta */, const tnsr::I<DataVector, Dim>& x,
    const std::array<double, Dim>& wave_numbers) noexcept;

template <size_t Dim>
tnsr::i<DataVector, Dim> variable(
    ::Tags::deriv<Tags::Field, tmpl::size_t<Dim>, Frame::Inertial> /* meta */,
    const tnsr::I<DataVector, Dim>& x,
    const std::array<double, Dim>& wave_numbers) noexcept;

template <size_t Dim>
Scalar<DataVector> variable(
    ::Tags::FixedSource<Tags::Field> /* meta */,
    const tnsr::I<DataVector, Dim>& x,
    const std::array<double, Dim>& wave_numbers) noexcept;

}  // namespace ProductOfSinusoids_detail

/*!
 * \brief A product of sinusoids \f$u(\boldsymbol{x}) = \prod_i \sin(k_i x_i)\f$
 *
 * \details Solves the Poisson equation \f$-\Delta u(x)=f(x)\f$ for a source
 * \f$f(x)=\boldsymbol{k}^2\prod_i \sin(k_i x_i)\f$.
 */
template <size_t Dim, typename LocalRegistrars =
                          tmpl::list<Registrars::ProductOfSinusoids<Dim>>>
class ProductOfSinusoids : public Solution<Dim, LocalRegistrars> {
 public:
  /// \cond
  ProductOfSinusoids() = default;
  explicit ProductOfSinusoids(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ProductOfSinusoids);  // NOLINT
  /// \endcond

  struct WaveNumbers {
    using type = std::array<double, Dim>;
    static constexpr OptionString help{"The wave numbers of the sinusoids"};
  };

  using options = tmpl::list<WaveNumbers>;
  static constexpr OptionString help{
      "A product of sinusoids that are taken of a wave number times the "
      "coordinate in each dimension."};

  explicit ProductOfSinusoids(
      const std::array<double, Dim>& wave_numbers) noexcept
      : wave_numbers_(wave_numbers) {}

  /// Retrieve a collection of variables at coordinates `x`
  template <typename... Tags>
  tuples::TaggedTuple<Tags...> variables_impl(
      const tnsr::I<DataVector, Dim, Frame::Inertial>& x,
      tmpl::list<Tags...> /*meta*/) const noexcept {
    return {ProductOfSinusoids_detail::variable(Tags{}, x, wave_numbers_)...};
  }

  // clang-tidy: no pass by reference
  void pup(PUP::er& p) noexcept {  // NOLINT
    p | wave_numbers_;
  }

  const std::array<double, Dim>& wave_numbers() const noexcept {
    return wave_numbers_;
  }

 private:
  std::array<double, Dim> wave_numbers_{
      {std::numeric_limits<double>::signaling_NaN()}};
};

/// \cond
template <size_t Dim, typename LocalRegistrars>
PUP::able::PUP_ID ProductOfSinusoids<Dim, LocalRegistrars>::my_PUP_ID =
    0;  // NOLINT
/// \endcond

}  // namespace Solutions
}  // namespace Poisson
