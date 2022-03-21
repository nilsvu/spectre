// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Poisson::Solutions {

namespace detail {
template <typename DataType, size_t Dim>
struct LorentzianVariables {
  using Cache = CachedTempBuffer<
      Tags::Field,
      ::Tags::deriv<Tags::Field, tmpl::size_t<Dim>, Frame::Inertial>,
      ::Tags::Flux<Tags::Field, tmpl::size_t<Dim>, Frame::Inertial>,
      ::Tags::FixedSource<Tags::Field>>;

  const tnsr::I<DataType, Dim>& x;

  void operator()(gsl::not_null<Scalar<DataType>*> field,
                  gsl::not_null<Cache*> cache, Tags::Field /*meta*/) const;
  void operator()(gsl::not_null<tnsr::i<DataType, Dim>*> field_gradient,
                  gsl::not_null<Cache*> cache,
                  ::Tags::deriv<Tags::Field, tmpl::size_t<Dim>,
                                Frame::Inertial> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::I<DataType, Dim>*> flux_for_field,
                  gsl::not_null<Cache*> cache,
                  ::Tags::Flux<Tags::Field, tmpl::size_t<Dim>,
                               Frame::Inertial> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> fixed_source_for_field,
                  gsl::not_null<Cache*> cache,
                  ::Tags::FixedSource<Tags::Field> /*meta*/) const;
};
}  // namespace detail

/*!
 * \brief A Lorentzian solution to the Poisson equation
 *
 * \details This implements the Lorentzian solution
 * \f$u(\boldsymbol{x})=\left(1+r^2\right)^{-\frac{1}{2}}\f$ to the
 * three-dimensional Poisson equation
 * \f$-\Delta u(\boldsymbol{x})=f(\boldsymbol{x})\f$, where
 * \f$r^2=x^2+y^2+z^2\f$. The corresponding source is
 * \f$f(\boldsymbol{x})=3\left(1+r^2\right)^{-\frac{5}{2}}\f$.
 *
 * \note Corresponding 1D and 2D solutions are not implemented yet.
 */
template <size_t Dim>
class Lorentzian : public elliptic::analytic_data::AnalyticSolution {
  static_assert(
      Dim == 3,
      "This solution is currently implemented in 3 spatial dimensions only");

 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "A Lorentzian solution to the Poisson equation."};

  Lorentzian() = default;
  Lorentzian(const Lorentzian&) = default;
  Lorentzian& operator=(const Lorentzian&) = default;
  Lorentzian(Lorentzian&&) = default;
  Lorentzian& operator=(Lorentzian&&) = default;
  ~Lorentzian() override = default;
  std::unique_ptr<elliptic::analytic_data::AnalyticSolution> get_clone()
      const override {
    return std::make_unique<Lorentzian>(*this);
  }

  /// \cond
  explicit Lorentzian(CkMigrateMessage* m)
      : elliptic::analytic_data::AnalyticSolution(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Lorentzian);  // NOLINT
  /// \endcond

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, Dim>& x,
      tmpl::list<RequestedTags...> /*meta*/) const {
    using VarsComputer = detail::LorentzianVariables<DataType, Dim>;
    typename VarsComputer::Cache cache{get_size(*x.begin())};
    const VarsComputer computer{x};
    return {cache.get_var(computer, RequestedTags{})...};
  }
};

/// \cond
template <size_t Dim>
PUP::able::PUP_ID Lorentzian<Dim>::my_PUP_ID = 0;  // NOLINT
/// \endcond

template <size_t Dim>
bool operator==(const Lorentzian<Dim>& /*lhs*/,
                const Lorentzian<Dim>& /*rhs*/) {
  return true;
}

template <size_t Dim>
bool operator!=(const Lorentzian<Dim>& lhs, const Lorentzian<Dim>& rhs) {
  return not(lhs == rhs);
}

}  // namespace Poisson::Solutions
