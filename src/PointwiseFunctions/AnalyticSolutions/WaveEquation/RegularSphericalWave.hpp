// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/preprocessor/list/for_each.hpp>
#include <boost/preprocessor/tuple/to_list.hpp>
#include <cstddef>
#include <memory>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Protocols.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Tags {
template <typename>
struct dt;
}  // namespace Tags
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace ScalarWave {
namespace Solutions {

/*!
 * \brief A 3D spherical wave solution to the Euclidean wave equation that is
 * regular at the origin
 *
 * The solution is given by \f$\Psi(\vec{x},t) = \Psi(r,t) =
 * \frac{F(r-t)-F(-r-t)}{r}\f$ describing an outgoing and an ingoing wave
 * with profile \f$F(u)\f$. For small \f$r\f$ the solution is approximated by
 * its Taylor expansion \f$\Psi(r,t)=2 F^\prime(-t) + \mathcal{O}(r^2)\f$. The
 * outgoing and ingoing waves meet at the origin (and cancel each other) when
 * \f$F^\prime(-t)=0\f$.
 *
 * The expansion is employed where \f$r\f$ lies within the cubic root of the
 * machine epsilon. Inside this radius we expect the error due to the truncation
 * of the Taylor expansion to be smaller than the numerical error made when
 * evaluating the full \f$\Psi(r,t)\f$. This is because the truncation error
 * scales as \f$r^2\f$ (since we keep the zeroth order, and the linear order
 * vanishes as all odd orders do) and the numerical error scales as
 * \f$\frac{\epsilon}{r}\f$, so they are comparable at
 * \f$r\propto\epsilon^\frac{1}{3}\f$.
 *
 * \requires the profile \f$F(u)\f$ to have a length scale of order unity so
 * that "small" \f$r\f$ means \f$r\ll 1\f$. This is without loss of generality
 * because of the scale invariance of the wave equation. The profile could be a
 * Gausssian centered at 0 with width 1, for instance.
 */
class RegularSphericalWave : public evolution::protocols::AnalyticSolution {
 public:
  static constexpr size_t volume_dim = 3;

  struct Profile {
    using type = std::unique_ptr<MathFunction<1>>;
    static constexpr OptionString help = {
        "The radial profile of the spherical wave."};
  };

  using options = tmpl::list<Profile>;

  static constexpr OptionString help = {
      "A spherical wave solution of the Euclidean wave equation that is "
      "regular at the origin"};

  RegularSphericalWave() = default;
  explicit RegularSphericalWave(
      std::unique_ptr<MathFunction<1>> profile) noexcept;
  RegularSphericalWave(const RegularSphericalWave&) noexcept = delete;
  RegularSphericalWave& operator=(const RegularSphericalWave&) noexcept =
      delete;
  RegularSphericalWave(RegularSphericalWave&&) noexcept = default;
  RegularSphericalWave& operator=(RegularSphericalWave&&) noexcept = default;
  ~RegularSphericalWave() noexcept = default;

 private:
  template <typename DataType>
  struct PrecomputedDataAtOrigin {
    DataType used_for_size;
    double profile_deriv;
    double profile_second_deriv;
    double profile_third_deriv;
  };

  template <typename DataType>
  struct PrecomputedData {
    DataType radial_distance;
    DataType profile_in;
    DataType profile_out;
    DataType profile_deriv_in;
    DataType profile_deriv_out;
    DataType profile_second_deriv_in;
    DataType profile_second_deriv_out;
  };

 public:
  // @{
  /// Retrieve the evolution variables at time `t` and spatial coordinates `x`
  tuples::TaggedTuple<> variables(const tnsr::I<DataVector, volume_dim>& /*x*/,
                                  double /*t*/, tmpl::list<> /*meta*/) const
      noexcept {
    return tuples::TaggedTuple<>{};
  }

  template <typename... Tags>
  TempBuffer<tmpl::list<Tags...>> variables(
      const tnsr::I<DataVector, volume_dim>& x, double t,
      tmpl::list<Tags...> /*meta*/) const noexcept {
    const DataVector radial_distance = get(magnitude(x));
    const DataVector phase_in = -radial_distance - t;
    const DataVector phase_out = radial_distance - t;
    // See class documentation for choice of cutoff
    const double r_cutoff = cbrt(std::numeric_limits<double>::epsilon());
    TempBuffer<tmpl::list<Tags...>> buffer{get_size(get<0>(x))};
    if (min(radial_distance) > r_cutoff) {
      const PrecomputedData<DataVector> precomputed_data{
          radial_distance,
          profile_->operator()(phase_in),
          profile_->operator()(phase_out),
          profile_->first_deriv(phase_in),
          profile_->first_deriv(phase_out),
          profile_->second_deriv(phase_in),
          profile_->second_deriv(phase_out)};
      const auto set_variable = [this, &x, &t, &precomputed_data,
                                 &buffer](auto tag_v) noexcept {
        using tag = decltype(tag_v);
        get<tag>(buffer) = variable(tag{}, x, t, precomputed_data);
      };
      EXPAND_PACK_LEFT_TO_RIGHT(set_variable(Tags{}));
      return buffer;
    }
    const PrecomputedDataAtOrigin<double> precomputed_data_at_origin{
        std::numeric_limits<double>::signaling_NaN(), profile_->first_deriv(-t),
        profile_->second_deriv(-t), profile_->third_deriv(-t)};
    for (size_t i = 0; i < radial_distance.size(); i++) {
      const double r_i = radial_distance[i];
      // Testing for r=0 here assumes a scale of order unity
      if (equal_within_roundoff(r_i, 0., r_cutoff, 1.)) {
        const auto set_variable_at_origin = [this, &i, &t,
                                             &precomputed_data_at_origin,
                                             &buffer](auto tag_v) noexcept {
          using tag = decltype(tag_v);
          const auto var =
              variable_at_origin(tag{}, t, precomputed_data_at_origin);
          for (size_t j = 0; j < var.size(); j++) {
            get<tag>(buffer)[j][i] = var[j];
          }
        };
        EXPAND_PACK_LEFT_TO_RIGHT(set_variable_at_origin(Tags{}));
      } else {
        PrecomputedData<double> precomputed_data{
            r_i,
            profile_->operator()(phase_in[i]),
            profile_->operator()(phase_out[i]),
            profile_->first_deriv(phase_in[i]),
            profile_->first_deriv(phase_out[i]),
            profile_->second_deriv(phase_in[i]),
            profile_->second_deriv(phase_out[i])};
        const tnsr::I<double, 3> x_i{
            {{get<0>(x)[i], get<1>(x)[i], get<2>(x)[i]}}};
        const auto set_variable = [this, &i, &x_i, &t, &precomputed_data,
                                   &buffer](auto tag_v) noexcept {
          using tag = decltype(tag_v);
          const auto var = variable(tag{}, x_i, t, precomputed_data);
          for (size_t j = 0; j < var.size(); j++) {
            get<tag>(buffer)[j][i] = var[j];
          }
        };
        EXPAND_PACK_LEFT_TO_RIGHT(set_variable(Tags{}));
      }
    }
    return buffer;
  }
  //@}

  // clang-tidy: no pass by reference
  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
// We only need separate macros for tensor types here because scalar wave tags
// are not templated on DataType
#define FUNC_DECL_SCALAR(r, data, elem)                          \
  template <typename DataType>                                   \
  Scalar<DataType> variable_at_origin(                           \
      elem /* meta */, double t,                                 \
      const PrecomputedDataAtOrigin<DataType>& precomputed_data) \
      const noexcept;                                            \
                                                                 \
  template <typename DataType>                                   \
  Scalar<DataType> variable(                                     \
      elem /* meta */, const tnsr::I<DataType, 3>& x, double t,  \
      const PrecomputedData<DataType>& precomputed_data) const noexcept;

#define FUNC_DECL_COVECTOR(r, data, elem)                        \
  template <typename DataType>                                   \
  tnsr::i<DataType, 3> variable_at_origin(                       \
      elem /* meta */, double t,                                 \
      const PrecomputedDataAtOrigin<DataType>& precomputed_data) \
      const noexcept;                                            \
                                                                 \
  template <typename DataType>                                   \
  tnsr::i<DataType, 3> variable(                                 \
      elem /* meta */, const tnsr::I<DataType, 3>& x, double t,  \
      const PrecomputedData<DataType>& precomputed_data) const noexcept;

#define TAG_LIST_SCALARS                                                \
  BOOST_PP_TUPLE_TO_LIST(                                               \
      4, (ScalarWave::Psi, ScalarWave::Pi, ::Tags::dt<ScalarWave::Psi>, \
          ::Tags::dt<ScalarWave::Pi>))
#define TAG_LIST_COVECTORS  \
  BOOST_PP_TUPLE_TO_LIST(2, \
                         (ScalarWave::Phi<3>, ::Tags::dt<ScalarWave::Phi<3>>))

  BOOST_PP_LIST_FOR_EACH(FUNC_DECL_SCALAR, _, TAG_LIST_SCALARS)
  BOOST_PP_LIST_FOR_EACH(FUNC_DECL_COVECTOR, _, TAG_LIST_COVECTORS)
#undef TAG_LIST_SCALARS
#undef TAG_LIST_COVECTORS
#undef FUNC_DECL_SCALAR
#undef FUNC_DECL_COVECTOR

  std::unique_ptr<MathFunction<1>> profile_;
};
}  // namespace Solutions
}  // namespace ScalarWave
