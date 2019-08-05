// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/Xcts/TovStar.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"                  // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"  // IWYU pragma: keep
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "ErrorHandling/Error.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Tov.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

// IWYU pragma: no_include <complex>

namespace Xcts {
namespace Solutions {

template <typename RadialSolution>
TovStar<RadialSolution>::TovStar(const double central_rest_mass_density,
                                 const double polytropic_constant,
                                 const double polytropic_exponent) noexcept
    : central_rest_mass_density_(central_rest_mass_density),
      polytropic_constant_(polytropic_constant),
      polytropic_exponent_(polytropic_exponent),
      equation_of_state_{polytropic_constant_, polytropic_exponent_} {}

template <typename RadialSolution>
void TovStar<RadialSolution>::pup(PUP::er& p) noexcept {
  p | central_rest_mass_density_;
  p | polytropic_constant_;
  p | polytropic_exponent_;
  p | equation_of_state_;
}

template <>
const gr::Solutions::TovSolution&
TovStar<gr::Solutions::TovSolution>::radial_tov_solution() const noexcept {
  static const gr::Solutions::TovSolution solution(
      equation_of_state_, central_rest_mass_density_, 0.0);
  return solution;
}

template <typename RadialSolution>
template <typename DataType>
tuples::TaggedTuple<Xcts::Tags::ConformalFactor<DataType>>
TovStar<RadialSolution>::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<Xcts::Tags::ConformalFactor<DataType>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const noexcept {
  // This only holds asymptotically
  return {make_with_value<Scalar<DataType>>(x, 1.)};
}

template <typename RadialSolution>
template <typename DataType>
tuples::TaggedTuple<::Tags::Initial<Xcts::Tags::ConformalFactor<DataType>>>
TovStar<RadialSolution>::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<::Tags::Initial<Xcts::Tags::ConformalFactor<DataType>>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const noexcept {
  return {make_with_value<Scalar<DataType>>(x, 1.)};
}

template <typename RadialSolution>
template <typename DataType>
tuples::TaggedTuple<::Tags::FixedSource<Xcts::Tags::ConformalFactor<DataType>>>
TovStar<RadialSolution>::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<
        ::Tags::FixedSource<Xcts::Tags::ConformalFactor<DataType>>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const noexcept {
  return {make_with_value<Scalar<DataType>>(x, 0.)};
}

template <typename RadialSolution>
template <typename DataType>
tuples::TaggedTuple<
    Xcts::Tags::ConformalFactorGradient<3, Frame::Inertial, DataType>>
TovStar<RadialSolution>::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<Xcts::Tags::ConformalFactorGradient<3, Frame::Inertial,
                                                   DataType>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const noexcept {
  return {make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.)};
}

template <typename RadialSolution>
template <typename DataType>
tuples::TaggedTuple<::Tags::Initial<
    Xcts::Tags::ConformalFactorGradient<3, Frame::Inertial, DataType>>>
TovStar<RadialSolution>::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<::Tags::Initial<Xcts::Tags::ConformalFactorGradient<
        3, Frame::Inertial, DataType>>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const noexcept {
  return {make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.)};
}

template <typename RadialSolution>
template <typename DataType>
tuples::TaggedTuple<::Tags::FixedSource<
    Xcts::Tags::ConformalFactorGradient<3, Frame::Inertial, DataType>>>
TovStar<RadialSolution>::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<::Tags::FixedSource<Xcts::Tags::ConformalFactorGradient<
        3, Frame::Inertial, DataType>>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const noexcept {
  return {make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.)};
}

template <typename RadialSolution>
template <typename DataType>
tuples::TaggedTuple<Xcts::Tags::LapseTimesConformalFactor<DataType>>
TovStar<RadialSolution>::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<Xcts::Tags::LapseTimesConformalFactor<DataType>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const noexcept {
  // This only holds asymptotically
  return {make_with_value<Scalar<DataType>>(x, 1.)};
}

template <typename RadialSolution>
template <typename DataType>
tuples::TaggedTuple<
    ::Tags::Initial<Xcts::Tags::LapseTimesConformalFactor<DataType>>>
TovStar<RadialSolution>::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<::Tags::Initial<
        Xcts::Tags::LapseTimesConformalFactor<DataType>>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const noexcept {
  return {make_with_value<Scalar<DataType>>(x, 1.)};
}

template <typename RadialSolution>
template <typename DataType>
tuples::TaggedTuple<
    ::Tags::FixedSource<Xcts::Tags::LapseTimesConformalFactor<DataType>>>
TovStar<RadialSolution>::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<::Tags::FixedSource<
        Xcts::Tags::LapseTimesConformalFactor<DataType>>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const noexcept {
  return {make_with_value<Scalar<DataType>>(x, 0.)};
}

template <typename RadialSolution>
template <typename DataType>
tuples::TaggedTuple<
    Xcts::Tags::LapseTimesConformalFactorGradient<3, Frame::Inertial, DataType>>
TovStar<RadialSolution>::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<Xcts::Tags::LapseTimesConformalFactorGradient<
        3, Frame::Inertial, DataType>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const noexcept {
  return {make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.)};
}

template <typename RadialSolution>
template <typename DataType>
tuples::TaggedTuple<
    ::Tags::Initial<Xcts::Tags::LapseTimesConformalFactorGradient<
        3, Frame::Inertial, DataType>>>
TovStar<RadialSolution>::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<::Tags::Initial<Xcts::Tags::LapseTimesConformalFactorGradient<
        3, Frame::Inertial, DataType>>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const noexcept {
  return {make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.)};
}

template <typename RadialSolution>
template <typename DataType>
tuples::TaggedTuple<
    ::Tags::FixedSource<Xcts::Tags::LapseTimesConformalFactorGradient<
        3, Frame::Inertial, DataType>>>
TovStar<RadialSolution>::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<
        ::Tags::FixedSource<Xcts::Tags::LapseTimesConformalFactorGradient<
            3, Frame::Inertial, DataType>>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const noexcept {
  return {make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.)};
}

template <typename RadialSolution>
template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>
TovStar<RadialSolution>::variables(
    const tnsr::I<DataType, 3>& /*x*/,
    tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/,
    const RadialVariables<DataType>& radial_vars) const noexcept {
  ERROR("Invalid radial vars");
  return radial_vars.specific_enthalpy;
}

// template <typename RadialSolution>
// template <typename DataType>
// tuples::TaggedTuple<gr::Tags::Lapse<DataType>>
// TovStar<RadialSolution>::variables(
//     const tnsr::I<DataType, 3>& /*x*/,
//     tmpl::list<gr::Tags::Lapse<DataType>> /*meta*/,
//     const RadialVariables<DataType>& radial_vars) const noexcept {
//   // Fix this for conformal flatness
//   return Scalar<DataType>{exp(radial_vars.metric_time_potential)};
// }

template <typename RadialSolution>
bool operator==(const TovStar<RadialSolution>& lhs,
                const TovStar<RadialSolution>& rhs) noexcept {
  // there is no comparison operator for the EoS, but should be okay as
  // the `polytropic_exponent`s and `polytropic_constant`s are compared
  return lhs.central_rest_mass_density_ == rhs.central_rest_mass_density_ and
         lhs.polytropic_constant_ == rhs.polytropic_constant_ and
         lhs.polytropic_exponent_ == rhs.polytropic_exponent_;
}

template <typename RadialSolution>
bool operator!=(const TovStar<RadialSolution>& lhs,
                const TovStar<RadialSolution>& rhs) noexcept {
  return not(lhs == rhs);
}

#define STYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define INSTANTIATE_VARS(_, data)                                              \
  template tuples::TaggedTuple<Xcts::Tags::ConformalFactor<DTYPE(data)>>       \
  TovStar<STYPE(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<Xcts::Tags::ConformalFactor<DTYPE(data)>> /*meta*/,           \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;         \
  template tuples::TaggedTuple<                                                \
      ::Tags::Initial<Xcts::Tags::ConformalFactor<DTYPE(data)>>>               \
  TovStar<STYPE(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<                                                              \
          ::Tags::Initial<Xcts::Tags::ConformalFactor<DTYPE(data)>>> /*meta*/, \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;         \
  template tuples::TaggedTuple<                                                \
      ::Tags::FixedSource<Xcts::Tags::ConformalFactor<DTYPE(data)>>>           \
  TovStar<STYPE(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<::Tags::FixedSource<                                          \
          Xcts::Tags::ConformalFactor<DTYPE(data)>>> /*meta*/,                 \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;         \
  template tuples::TaggedTuple<                                                \
      Xcts::Tags::ConformalFactorGradient<3, Frame::Inertial, DTYPE(data)>>    \
  TovStar<STYPE(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<Xcts::Tags::ConformalFactorGradient<3, Frame::Inertial,       \
                                                     DTYPE(data)>> /*meta*/,   \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;         \
  template tuples::TaggedTuple<::Tags::Initial<                                \
      Xcts::Tags::ConformalFactorGradient<3, Frame::Inertial, DTYPE(data)>>>   \
  TovStar<STYPE(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<::Tags::Initial<Xcts::Tags::ConformalFactorGradient<          \
          3, Frame::Inertial, DTYPE(data)>>> /*meta*/,                         \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;         \
  template tuples::TaggedTuple<::Tags::FixedSource<                            \
      Xcts::Tags::ConformalFactorGradient<3, Frame::Inertial, DTYPE(data)>>>   \
  TovStar<STYPE(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<::Tags::FixedSource<Xcts::Tags::ConformalFactorGradient<      \
          3, Frame::Inertial, DTYPE(data)>>> /*meta*/,                         \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;         \
  template tuples::TaggedTuple<                                                \
      Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>>                      \
  TovStar<STYPE(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>> /*meta*/, \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;         \
  template tuples::TaggedTuple<                                                \
      ::Tags::Initial<Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>>>     \
  TovStar<STYPE(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<::Tags::Initial<                                              \
          Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>>> /*meta*/,       \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;         \
  template tuples::TaggedTuple<                                                \
      ::Tags::FixedSource<Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>>> \
  TovStar<STYPE(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<::Tags::FixedSource<                                          \
          Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>>> /*meta*/,       \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;         \
  template tuples::TaggedTuple<Xcts::Tags::LapseTimesConformalFactorGradient<  \
      3, Frame::Inertial, DTYPE(data)>>                                        \
  TovStar<STYPE(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<Xcts::Tags::LapseTimesConformalFactorGradient<                \
          3, Frame::Inertial, DTYPE(data)>> /*meta*/,                          \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;         \
  template tuples::TaggedTuple<                                                \
      ::Tags::Initial<Xcts::Tags::LapseTimesConformalFactorGradient<           \
          3, Frame::Inertial, DTYPE(data)>>>                                   \
  TovStar<STYPE(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<                                                              \
          ::Tags::Initial<Xcts::Tags::LapseTimesConformalFactorGradient<       \
              3, Frame::Inertial, DTYPE(data)>>> /*meta*/,                     \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;         \
  template tuples::TaggedTuple<                                                \
      ::Tags::FixedSource<Xcts::Tags::LapseTimesConformalFactorGradient<       \
          3, Frame::Inertial, DTYPE(data)>>>                                   \
  TovStar<STYPE(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<                                                              \
          ::Tags::FixedSource<Xcts::Tags::LapseTimesConformalFactorGradient<   \
              3, Frame::Inertial, DTYPE(data)>>> /*meta*/,                     \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;         \
  template tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DTYPE(data)>>     \
  TovStar<STYPE(data)>::variables(                                             \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<hydro::Tags::SpecificEnthalpy<DTYPE(data)>> /*meta*/,         \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;

#define INSTANTIATE(_, data)                                \
  template class TovStar<STYPE(data)>;                      \
  template bool operator!=(const TovStar<STYPE(data)>& lhs, \
                           const TovStar<STYPE(data)>& rhs) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_VARS, (gr::Solutions::TovSolution),
                        (double, DataVector))
GENERATE_INSTANTIATIONS(INSTANTIATE, (gr::Solutions::TovSolution))

#undef DTYPE
#undef STYPE
#undef INSTANTIATE
#undef INSTANTIATE_VARS
}  // namespace Solutions
}  // namespace Xcts
/// \endcond
