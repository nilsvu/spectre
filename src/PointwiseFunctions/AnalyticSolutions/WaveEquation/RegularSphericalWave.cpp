// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/RegularSphericalWave.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace ScalarWave {
namespace Solutions {

// Psi

template <typename DataType>
Scalar<DataType> RegularSphericalWave::variable_at_origin(
    ScalarWave::Psi /* meta */, const double /* t */,
    const PrecomputedDataAtOrigin<DataType>& precomputed_data) const noexcept {
  return Scalar<DataType>{2. * precomputed_data.profile_deriv};
}

template <typename DataType>
Scalar<DataType> RegularSphericalWave::variable(
    ScalarWave::Psi /* meta */, const tnsr::I<DataType, 3>& /* x */,
    const double /* t */,
    const PrecomputedData<DataType>& precomputed_data) const noexcept {
  return Scalar<DataType>{
      (precomputed_data.profile_out - precomputed_data.profile_in) /
      precomputed_data.radial_distance};
}

template <typename DataType>
Scalar<DataType> RegularSphericalWave::variable_at_origin(
    ::Tags::dt<ScalarWave::Psi> /* meta */, const double /* t */,
    const PrecomputedDataAtOrigin<DataType>& precomputed_data) const noexcept {
  return Scalar<DataType>{-2. * precomputed_data.profile_second_deriv};
}

template <typename DataType>
Scalar<DataType> RegularSphericalWave::variable(
    ::Tags::dt<ScalarWave::Psi> /* meta */, const tnsr::I<DataType, 3>& /* x
                                                                         */
    ,
    const double /* t */,
    const PrecomputedData<DataType>& precomputed_data) const noexcept {
  return Scalar<DataType>{
      (precomputed_data.profile_deriv_in - precomputed_data.profile_deriv_out) /
      precomputed_data.radial_distance};
}

// Pi

template <typename DataType>
Scalar<DataType> RegularSphericalWave::variable_at_origin(
    ScalarWave::Pi /* meta */, const double /* t */,
    const PrecomputedDataAtOrigin<DataType>& precomputed_data) const noexcept {
  return Scalar<DataType>{2. * precomputed_data.profile_second_deriv};
}

template <typename DataType>
Scalar<DataType> RegularSphericalWave::variable(
    ScalarWave::Pi /* meta */, const tnsr::I<DataType, 3>& /* x */,
    const double /* t */,
    const PrecomputedData<DataType>& precomputed_data) const noexcept {
  return Scalar<DataType>{
      (precomputed_data.profile_deriv_out - precomputed_data.profile_deriv_in) /
      precomputed_data.radial_distance};
}

template <typename DataType>
Scalar<DataType> RegularSphericalWave::variable_at_origin(
    ::Tags::dt<ScalarWave::Pi> /* meta */, const double /* t */,
    const PrecomputedDataAtOrigin<DataType>& precomputed_data) const noexcept {
  return Scalar<DataType>{-2. * precomputed_data.profile_third_deriv};
}

template <typename DataType>
Scalar<DataType> RegularSphericalWave::variable(
    ::Tags::dt<ScalarWave::Pi> /* meta */, const tnsr::I<DataType, 3>& /* x
                                                                        */
    ,
    const double /* t */,
    const PrecomputedData<DataType>& precomputed_data) const noexcept {
  return Scalar<DataType>{(precomputed_data.profile_second_deriv_in -
                           precomputed_data.profile_second_deriv_out) /
                          precomputed_data.radial_distance};
}

// Phi

template <typename DataType>
tnsr::i<DataType, 3> RegularSphericalWave::variable_at_origin(
    ScalarWave::Phi<3> /* meta */, const double /* t */,
    const PrecomputedDataAtOrigin<DataType>& precomputed_data) const noexcept {
  const auto zero =
      make_with_value<DataType>(precomputed_data.used_for_size, 0.);
  return tnsr::i<DataType, 3>{{{zero, zero, zero}}};
}

template <typename DataType>
tnsr::i<DataType, 3> RegularSphericalWave::variable(
    ScalarWave::Phi<3> /* meta */, const tnsr::I<DataType, 3>& x,
    const double /* t */,
    const PrecomputedData<DataType>& precomputed_data) const noexcept {
  const DataType phi_isotropic =
      (precomputed_data.profile_deriv_out + precomputed_data.profile_deriv_in -
       (precomputed_data.profile_out - precomputed_data.profile_deriv_in) /
           precomputed_data.radial_distance) /
      square(precomputed_data.radial_distance);
  return tnsr::i<DataType, 3>{
      {{phi_isotropic * get<0>(x), phi_isotropic * get<1>(x),
        phi_isotropic * get<2>(x)}}};
}

template <typename DataType>
tnsr::i<DataType, 3> RegularSphericalWave::variable_at_origin(
    ::Tags::dt<ScalarWave::Phi<3>> /* meta */, const double /* t */,
    const PrecomputedDataAtOrigin<DataType>& precomputed_data) const noexcept {
  const auto zero =
      make_with_value<DataType>(precomputed_data.used_for_size, 0.);
  return tnsr::i<DataType, 3>{{{zero, zero, zero}}};
}

template <typename DataType>
tnsr::i<DataType, 3> RegularSphericalWave::variable(
    ::Tags::dt<ScalarWave::Phi<3>> /* meta */, const tnsr::I<DataType, 3>& x,
    const double /* t */,
    const PrecomputedData<DataType>& precomputed_data) const noexcept {
  const DataType dphi_dt_isotropic =
      -(precomputed_data.profile_second_deriv_out +
        precomputed_data.profile_second_deriv_in +
        (-precomputed_data.profile_deriv_out +
         precomputed_data.profile_deriv_in) /
            precomputed_data.radial_distance) /
      square(precomputed_data.radial_distance);
  return tnsr::i<DataType, 3>{
      {{dphi_dt_isotropic * get<0>(x), dphi_dt_isotropic * get<1>(x),
        dphi_dt_isotropic * get<2>(x)}}};
}

RegularSphericalWave::RegularSphericalWave(
    std::unique_ptr<MathFunction<1>> profile) noexcept
    : profile_(std::move(profile)) {}

void RegularSphericalWave::pup(PUP::er& p) noexcept { p | profile_; }

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define TAG(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE_SCALARS(_, data)                                     \
  template Scalar<DTYPE(data)> RegularSphericalWave::variable_at_origin( \
      TAG(data) /* meta */, double t,                                    \
      const PrecomputedDataAtOrigin<DTYPE(data)>& precomputed_data)      \
      const noexcept;                                                    \
                                                                         \
  template Scalar<DTYPE(data)> RegularSphericalWave::variable(           \
      TAG(data) /* meta */, const tnsr::I<DTYPE(data), 3>& x, double t,  \
      const PrecomputedData<DTYPE(data)>& precomputed_data) const noexcept;

#define INSTANTIATE_COVECTORS(_, data)                                       \
  template tnsr::i<DTYPE(data), 3> RegularSphericalWave::variable_at_origin( \
      TAG(data) /* meta */, double t,                                        \
      const PrecomputedDataAtOrigin<DTYPE(data)>& precomputed_data)          \
      const noexcept;                                                        \
                                                                             \
  template tnsr::i<DTYPE(data), 3> RegularSphericalWave::variable(           \
      TAG(data) /* meta */, const tnsr::I<DTYPE(data), 3>& x, double t,      \
      const PrecomputedData<DTYPE(data)>& precomputed_data) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_SCALARS, (double, DataVector),
                        (ScalarWave::Psi, ScalarWave::Pi,
                         ::Tags::dt<ScalarWave::Psi>,
                         ::Tags::dt<ScalarWave::Pi>))
GENERATE_INSTANTIATIONS(INSTANTIATE_COVECTORS, (double, DataVector),
                        (ScalarWave::Phi<3>, ::Tags::dt<ScalarWave::Phi<3>>))

#undef DTYPE
#undef TAG
#undef INSTANTIATE_SCALARS
#undef INSTANTIATE_COVECTORS

}  // namespace Solutions
}  // namespace ScalarWave
