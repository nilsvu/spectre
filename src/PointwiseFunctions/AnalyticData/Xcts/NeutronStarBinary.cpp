// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/Xcts/NeutronStarBinary.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "ErrorHandling/Error.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/TovStar.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Xcts {
namespace AnalyticData {

NeutronStarBinary::NeutronStarBinary(
    const std::array<double, 2> central_rest_mass_densities,
    const double angular_velocity, const double polytropic_constant,
    const double polytropic_exponent) noexcept
    : central_rest_mass_densities_(std::move(central_rest_mass_densities)),
      angular_velocity_(angular_velocity),
      polytropic_constant_(polytropic_constant),
      polytropic_exponent_(polytropic_exponent),
      equation_of_state_{polytropic_constant_, polytropic_exponent_},
      stars_{{{central_rest_mass_densities_[0], polytropic_constant,
               polytropic_exponent},
              {central_rest_mass_densities_[1], polytropic_constant,
               polytropic_exponent}}} {}

void NeutronStarBinary::pup(PUP::er& p) noexcept {
  p | central_rest_mass_densities_;
  p | polytropic_constant_;
  p | polytropic_exponent_;
  p | equation_of_state_;
  p | stars_;
}

template <typename DataType>
tuples::TaggedTuple<Xcts::Tags::ConformalFactor<DataType>>
NeutronStarBinary::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<Xcts::Tags::ConformalFactor<DataType>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const noexcept {
  return {make_with_value<Scalar<DataType>>(x, 1.)};
}

template <typename DataType>
tuples::TaggedTuple<::Tags::Initial<Xcts::Tags::ConformalFactor<DataType>>>
NeutronStarBinary::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<::Tags::Initial<Xcts::Tags::ConformalFactor<DataType>>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const noexcept {
  return {make_with_value<Scalar<DataType>>(x, 1.)};
}

template <typename DataType>
tuples::TaggedTuple<::Tags::FixedSource<Xcts::Tags::ConformalFactor<DataType>>>
NeutronStarBinary::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<
        ::Tags::FixedSource<Xcts::Tags::ConformalFactor<DataType>>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const noexcept {
  return {make_with_value<Scalar<DataType>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<
    Xcts::Tags::ConformalFactorGradient<3, Frame::Inertial, DataType>>
NeutronStarBinary::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<Xcts::Tags::ConformalFactorGradient<3, Frame::Inertial,
                                                   DataType>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const noexcept {
  return {make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<::Tags::Initial<
    Xcts::Tags::ConformalFactorGradient<3, Frame::Inertial, DataType>>>
NeutronStarBinary::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<::Tags::Initial<Xcts::Tags::ConformalFactorGradient<
        3, Frame::Inertial, DataType>>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const noexcept {
  return {make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<::Tags::FixedSource<
    Xcts::Tags::ConformalFactorGradient<3, Frame::Inertial, DataType>>>
NeutronStarBinary::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<::Tags::FixedSource<Xcts::Tags::ConformalFactorGradient<
        3, Frame::Inertial, DataType>>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const noexcept {
  return {make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<Xcts::Tags::LapseTimesConformalFactor<DataType>>
NeutronStarBinary::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<Xcts::Tags::LapseTimesConformalFactor<DataType>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const noexcept {
  return {make_with_value<Scalar<DataType>>(x, 1.)};
}

template <typename DataType>
tuples::TaggedTuple<
    ::Tags::Initial<Xcts::Tags::LapseTimesConformalFactor<DataType>>>
NeutronStarBinary::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<::Tags::Initial<
        Xcts::Tags::LapseTimesConformalFactor<DataType>>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const noexcept {
  return {make_with_value<Scalar<DataType>>(x, 1.)};
}

template <typename DataType>
tuples::TaggedTuple<
    ::Tags::FixedSource<Xcts::Tags::LapseTimesConformalFactor<DataType>>>
NeutronStarBinary::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<::Tags::FixedSource<
        Xcts::Tags::LapseTimesConformalFactor<DataType>>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const noexcept {
  return {make_with_value<Scalar<DataType>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<
    Xcts::Tags::LapseTimesConformalFactorGradient<3, Frame::Inertial, DataType>>
NeutronStarBinary::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<Xcts::Tags::LapseTimesConformalFactorGradient<
        3, Frame::Inertial, DataType>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const noexcept {
  return {make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<
    ::Tags::Initial<Xcts::Tags::LapseTimesConformalFactorGradient<
        3, Frame::Inertial, DataType>>>
NeutronStarBinary::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<::Tags::Initial<Xcts::Tags::LapseTimesConformalFactorGradient<
        3, Frame::Inertial, DataType>>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const noexcept {
  return {make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<
    ::Tags::FixedSource<Xcts::Tags::LapseTimesConformalFactorGradient<
        3, Frame::Inertial, DataType>>>
NeutronStarBinary::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<
        ::Tags::FixedSource<Xcts::Tags::LapseTimesConformalFactorGradient<
            3, Frame::Inertial, DataType>>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const noexcept {
  return {make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.)};
}

bool operator==(const NeutronStarBinary& lhs,
                const NeutronStarBinary& rhs) noexcept {
  // there is no comparison operator for the EoS, but should be okay as
  // the `polytropic_exponent`s and `polytropic_constant`s are compared
  return lhs.central_rest_mass_densities_ ==
             rhs.central_rest_mass_densities_ and
         lhs.angular_velocity_ == rhs.angular_velocity_ and
         lhs.polytropic_constant_ == rhs.polytropic_constant_ and
         lhs.polytropic_exponent_ == rhs.polytropic_exponent_;
}

bool operator!=(const NeutronStarBinary& lhs,
                const NeutronStarBinary& rhs) noexcept {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE_VARS(_, data)                                              \
  template tuples::TaggedTuple<Xcts::Tags::ConformalFactor<DTYPE(data)>>       \
  NeutronStarBinary::variables(                                                \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<Xcts::Tags::ConformalFactor<DTYPE(data)>> /*meta*/,           \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;         \
  template tuples::TaggedTuple<                                                \
      ::Tags::Initial<Xcts::Tags::ConformalFactor<DTYPE(data)>>>               \
  NeutronStarBinary::variables(                                                \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<                                                              \
          ::Tags::Initial<Xcts::Tags::ConformalFactor<DTYPE(data)>>> /*meta*/, \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;         \
  template tuples::TaggedTuple<                                                \
      ::Tags::FixedSource<Xcts::Tags::ConformalFactor<DTYPE(data)>>>           \
  NeutronStarBinary::variables(                                                \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<::Tags::FixedSource<                                          \
          Xcts::Tags::ConformalFactor<DTYPE(data)>>> /*meta*/,                 \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;         \
  template tuples::TaggedTuple<                                                \
      Xcts::Tags::ConformalFactorGradient<3, Frame::Inertial, DTYPE(data)>>    \
  NeutronStarBinary::variables(                                                \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<Xcts::Tags::ConformalFactorGradient<3, Frame::Inertial,       \
                                                     DTYPE(data)>> /*meta*/,   \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;         \
  template tuples::TaggedTuple<::Tags::Initial<                                \
      Xcts::Tags::ConformalFactorGradient<3, Frame::Inertial, DTYPE(data)>>>   \
  NeutronStarBinary::variables(                                                \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<::Tags::Initial<Xcts::Tags::ConformalFactorGradient<          \
          3, Frame::Inertial, DTYPE(data)>>> /*meta*/,                         \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;         \
  template tuples::TaggedTuple<::Tags::FixedSource<                            \
      Xcts::Tags::ConformalFactorGradient<3, Frame::Inertial, DTYPE(data)>>>   \
  NeutronStarBinary::variables(                                                \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<::Tags::FixedSource<Xcts::Tags::ConformalFactorGradient<      \
          3, Frame::Inertial, DTYPE(data)>>> /*meta*/,                         \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;         \
  template tuples::TaggedTuple<                                                \
      Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>>                      \
  NeutronStarBinary::variables(                                                \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>> /*meta*/, \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;         \
  template tuples::TaggedTuple<                                                \
      ::Tags::Initial<Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>>>     \
  NeutronStarBinary::variables(                                                \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<::Tags::Initial<                                              \
          Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>>> /*meta*/,       \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;         \
  template tuples::TaggedTuple<                                                \
      ::Tags::FixedSource<Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>>> \
  NeutronStarBinary::variables(                                                \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<::Tags::FixedSource<                                          \
          Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>>> /*meta*/,       \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;         \
  template tuples::TaggedTuple<Xcts::Tags::LapseTimesConformalFactorGradient<  \
      3, Frame::Inertial, DTYPE(data)>>                                        \
  NeutronStarBinary::variables(                                                \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<Xcts::Tags::LapseTimesConformalFactorGradient<                \
          3, Frame::Inertial, DTYPE(data)>> /*meta*/,                          \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;         \
  template tuples::TaggedTuple<                                                \
      ::Tags::Initial<Xcts::Tags::LapseTimesConformalFactorGradient<           \
          3, Frame::Inertial, DTYPE(data)>>>                                   \
  NeutronStarBinary::variables(                                                \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<                                                              \
          ::Tags::Initial<Xcts::Tags::LapseTimesConformalFactorGradient<       \
              3, Frame::Inertial, DTYPE(data)>>> /*meta*/,                     \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;         \
  template tuples::TaggedTuple<                                                \
      ::Tags::FixedSource<Xcts::Tags::LapseTimesConformalFactorGradient<       \
          3, Frame::Inertial, DTYPE(data)>>>                                   \
  NeutronStarBinary::variables(                                                \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<                                                              \
          ::Tags::FixedSource<Xcts::Tags::LapseTimesConformalFactorGradient<   \
              3, Frame::Inertial, DTYPE(data)>>> /*meta*/,                     \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_VARS, (double, DataVector))

#undef DTYPE
#undef STYPE
#undef INSTANTIATE
#undef INSTANTIATE_VARS
}  // namespace AnalyticData
}  // namespace Xcts
/// \endcond
