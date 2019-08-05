// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/Xcts/TovStar.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "ErrorHandling/Error.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Tov.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Xcts {
namespace Solutions {

TovStar::TovStar(const double central_rest_mass_density,
                 const double polytropic_constant,
                 const double polytropic_exponent) noexcept
    : central_rest_mass_density_(central_rest_mass_density),
      polytropic_constant_(polytropic_constant),
      polytropic_exponent_(polytropic_exponent),
      equation_of_state_{polytropic_constant_, polytropic_exponent_},
      radial_solution_{equation_of_state_, central_rest_mass_density_, 0.0} {}

void TovStar::pup(PUP::er& p) noexcept {
  p | central_rest_mass_density_;
  p | polytropic_constant_;
  p | polytropic_exponent_;
  p | equation_of_state_;
  p | radial_solution_;
}

template <typename DataType>
tuples::TaggedTuple<Xcts::Tags::ConformalFactor<DataType>> TovStar::variables(
    const tnsr::I<DataType, 3>& /*x*/,
    tmpl::list<Xcts::Tags::ConformalFactor<DataType>> /*meta*/,
    const RadialVariables<DataType>& radial_vars) const noexcept {
  return {radial_vars.conformal_factor};
}

template <typename DataType>
tuples::TaggedTuple<::Tags::Initial<Xcts::Tags::ConformalFactor<DataType>>>
TovStar::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<::Tags::Initial<Xcts::Tags::ConformalFactor<DataType>>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const noexcept {
  return {make_with_value<Scalar<DataType>>(x, 1.)};
}

template <typename DataType>
tuples::TaggedTuple<::Tags::FixedSource<Xcts::Tags::ConformalFactor<DataType>>>
TovStar::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<
        ::Tags::FixedSource<Xcts::Tags::ConformalFactor<DataType>>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const noexcept {
  return {make_with_value<Scalar<DataType>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>,
                                  tmpl::size_t<3>, Frame::Inertial>>
TovStar::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>,
                             tmpl::size_t<3>, Frame::Inertial>> /*meta*/,
    const RadialVariables<DataType>& radial_vars) const noexcept {
  auto result = make_with_value<tnsr::i<DataType, 3>>(x, 0.);
  get<0>(result) = get<0>(x) / radial_vars.isotropic_radius *
                   get(radial_vars.dr_conformal_factor);
  get<1>(result) = get<1>(x) / radial_vars.isotropic_radius *
                   get(radial_vars.dr_conformal_factor);
  get<2>(result) = get<2>(x) / radial_vars.isotropic_radius *
                   get(radial_vars.dr_conformal_factor);
  return {std::move(result)};
}

template <typename DataType>
tuples::TaggedTuple<::Tags::Initial<::Tags::deriv<
    Xcts::Tags::ConformalFactor<DataType>, tmpl::size_t<3>, Frame::Inertial>>>
TovStar::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<::Tags::Initial<
        ::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>, tmpl::size_t<3>,
                      Frame::Inertial>>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const noexcept {
  return {make_with_value<tnsr::i<DataType, 3, Frame::Inertial>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<Xcts::Tags::LapseTimesConformalFactor<DataType>>
TovStar::variables(
    const tnsr::I<DataType, 3>& /*x*/,
    tmpl::list<Xcts::Tags::LapseTimesConformalFactor<DataType>> /*meta*/,
    const RadialVariables<DataType>& radial_vars) const noexcept {
  return {Scalar<DataType>{get(radial_vars.lapse) *
                           get(radial_vars.conformal_factor)}};
}

template <typename DataType>
tuples::TaggedTuple<
    ::Tags::Initial<Xcts::Tags::LapseTimesConformalFactor<DataType>>>
TovStar::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<::Tags::Initial<
        Xcts::Tags::LapseTimesConformalFactor<DataType>>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const noexcept {
  return {make_with_value<Scalar<DataType>>(x, 1.)};
}

template <typename DataType>
tuples::TaggedTuple<
    ::Tags::FixedSource<Xcts::Tags::LapseTimesConformalFactor<DataType>>>
TovStar::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<::Tags::FixedSource<
        Xcts::Tags::LapseTimesConformalFactor<DataType>>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const noexcept {
  return {make_with_value<Scalar<DataType>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<
    ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
                  tmpl::size_t<3>, Frame::Inertial>>
TovStar::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
                             tmpl::size_t<3>, Frame::Inertial>> /*meta*/,
    const RadialVariables<DataType>& radial_vars) const noexcept {
  DataType deriv_factor =
      get(radial_vars.dr_lapse) * get(radial_vars.conformal_factor) +
      get(radial_vars.dr_conformal_factor) * get(radial_vars.lapse);
  auto result = make_with_value<tnsr::i<DataType, 3>>(x, 0.);
  get<0>(result) = get<0>(x) / radial_vars.isotropic_radius * deriv_factor;
  get<1>(result) = get<1>(x) / radial_vars.isotropic_radius * deriv_factor;
  get<2>(result) = get<2>(x) / radial_vars.isotropic_radius * deriv_factor;
  return {std::move(result)};
}

template <typename DataType>
tuples::TaggedTuple<::Tags::Initial<
    ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
                  tmpl::size_t<3>, Frame::Inertial>>>
TovStar::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<::Tags::Initial<
        ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
                      tmpl::size_t<3>, Frame::Inertial>>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const noexcept {
  return {make_with_value<tnsr::i<DataType, 3, Frame::Inertial>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<gr::Tags::Shift<3, Frame::Inertial, DataType>>
TovStar::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<gr::Tags::Shift<3, Frame::Inertial, DataType>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const noexcept {
  return {make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<
    ::Tags::Initial<gr::Tags::Shift<3, Frame::Inertial, DataType>>>
TovStar::variables(const tnsr::I<DataType, 3>& x,
                   tmpl::list<::Tags::Initial<
                       gr::Tags::Shift<3, Frame::Inertial, DataType>>> /*meta*/,
                   const RadialVariables<DataType>& /*radial_vars*/) const
    noexcept {
  return {make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<
    ::Tags::FixedSource<gr::Tags::Shift<3, Frame::Inertial, DataType>>>
TovStar::variables(const tnsr::I<DataType, 3>& x,
                   tmpl::list<::Tags::FixedSource<
                       gr::Tags::Shift<3, Frame::Inertial, DataType>>> /*meta*/,
                   const RadialVariables<DataType>& /*radial_vars*/) const
    noexcept {
  return {make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<Xcts::Tags::ShiftStrain<3, Frame::Inertial, DataType>>
TovStar::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<Xcts::Tags::ShiftStrain<3, Frame::Inertial, DataType>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const noexcept {
  return {make_with_value<tnsr::ii<DataType, 3, Frame::Inertial>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<
    ::Tags::Initial<Xcts::Tags::ShiftStrain<3, Frame::Inertial, DataType>>>
TovStar::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<::Tags::Initial<
        Xcts::Tags::ShiftStrain<3, Frame::Inertial, DataType>>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const noexcept {
  return {make_with_value<tnsr::ii<DataType, 3, Frame::Inertial>>(x, 0.)};
}

template <typename DataType>
tuples::TaggedTuple<gr::Tags::EnergyDensity<DataType>> TovStar::variables(
    const tnsr::I<DataType, 3>& /*x*/,
    tmpl::list<gr::Tags::EnergyDensity<DataType>> /*meta*/,
    const RadialVariables<DataType>& radial_vars) const noexcept {
  return {Scalar<DataType>{get(radial_vars.specific_enthalpy) *
                               get(radial_vars.rest_mass_density) -
                           get(radial_vars.pressure)}};
}

template <typename DataType>
tuples::TaggedTuple<gr::Tags::StressTrace<DataType>> TovStar::variables(
    const tnsr::I<DataType, 3>& /*x*/,
    tmpl::list<gr::Tags::StressTrace<DataType>> /*meta*/,
    const RadialVariables<DataType>& radial_vars) const noexcept {
  return {Scalar<DataType>{3. * get(radial_vars.pressure)}};
}

template <typename DataType>
tuples::TaggedTuple<gr::Tags::MomentumDensity<3, Frame::Inertial, DataType>>
TovStar::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<
        ::gr::Tags::MomentumDensity<3, Frame::Inertial, DataType>> /*meta*/,
    const RadialVariables<DataType>& /*radial_vars*/) const noexcept {
  return {make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.)};
}

bool operator==(const TovStar& lhs, const TovStar& rhs) noexcept {
  // there is no comparison operator for the EoS, but should be okay as
  // the `polytropic_exponent`s and `polytropic_constant`s are compared
  return lhs.central_rest_mass_density_ == rhs.central_rest_mass_density_ and
         lhs.polytropic_constant_ == rhs.polytropic_constant_ and
         lhs.polytropic_exponent_ == rhs.polytropic_exponent_;
}

bool operator!=(const TovStar& lhs, const TovStar& rhs) noexcept {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE_VARS(_, data)                                              \
  template tuples::TaggedTuple<Xcts::Tags::ConformalFactor<DTYPE(data)>>       \
  TovStar::variables(                                                          \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<Xcts::Tags::ConformalFactor<DTYPE(data)>> /*meta*/,           \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;         \
  template tuples::TaggedTuple<                                                \
      ::Tags::Initial<Xcts::Tags::ConformalFactor<DTYPE(data)>>>               \
  TovStar::variables(                                                          \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<                                                              \
          ::Tags::Initial<Xcts::Tags::ConformalFactor<DTYPE(data)>>> /*meta*/, \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;         \
  template tuples::TaggedTuple<                                                \
      ::Tags::FixedSource<Xcts::Tags::ConformalFactor<DTYPE(data)>>>           \
  TovStar::variables(const tnsr::I<DTYPE(data), 3>& x,                         \
                     tmpl::list<::Tags::FixedSource<                           \
                         Xcts::Tags::ConformalFactor<DTYPE(data)>>> /*meta*/,  \
                     const RadialVariables<DTYPE(data)>& radial_vars)          \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      ::Tags::deriv<Xcts::Tags::ConformalFactor<DTYPE(data)>, tmpl::size_t<3>, \
                    Frame::Inertial>>                                          \
  TovStar::variables(                                                          \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<::Tags::deriv<Xcts::Tags::ConformalFactor<DTYPE(data)>,       \
                               tmpl::size_t<3>, Frame::Inertial>> /*meta*/,    \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;         \
  template tuples::TaggedTuple<                                                \
      ::Tags::Initial<::Tags::deriv<Xcts::Tags::ConformalFactor<DTYPE(data)>,  \
                                    tmpl::size_t<3>, Frame::Inertial>>>        \
  TovStar::variables(                                                          \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<::Tags::Initial<                                              \
          ::Tags::deriv<Xcts::Tags::ConformalFactor<DTYPE(data)>,              \
                        tmpl::size_t<3>, Frame::Inertial>>> /*meta*/,          \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;         \
  template tuples::TaggedTuple<                                                \
      Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>>                      \
  TovStar::variables(                                                          \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>> /*meta*/, \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;         \
  template tuples::TaggedTuple<                                                \
      ::Tags::Initial<Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>>>     \
  TovStar::variables(                                                          \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<::Tags::Initial<                                              \
          Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>>> /*meta*/,       \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;         \
  template tuples::TaggedTuple<                                                \
      ::Tags::FixedSource<Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>>> \
  TovStar::variables(                                                          \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<::Tags::FixedSource<                                          \
          Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>>> /*meta*/,       \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;         \
  template tuples::TaggedTuple<                                                \
      ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>,        \
                    tmpl::size_t<3>, Frame::Inertial>>                         \
  TovStar::variables(                                                          \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<                                                              \
          ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>,    \
                        tmpl::size_t<3>, Frame::Inertial>> /*meta*/,           \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;         \
  template tuples::TaggedTuple<::Tags::Initial<                                \
      ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>,        \
                    tmpl::size_t<3>, Frame::Inertial>>>                        \
  TovStar::variables(                                                          \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<::Tags::Initial<                                              \
          ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>,    \
                        tmpl::size_t<3>, Frame::Inertial>>> /*meta*/,          \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;         \
  template tuples::TaggedTuple<                                                \
      gr::Tags::Shift<3, Frame::Inertial, DTYPE(data)>>                        \
  TovStar::variables(                                                          \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<gr::Tags::Shift<3, Frame::Inertial, DTYPE(data)>> /*meta*/,   \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;         \
  template tuples::TaggedTuple<                                                \
      ::Tags::Initial<gr::Tags::Shift<3, Frame::Inertial, DTYPE(data)>>>       \
  TovStar::variables(                                                          \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<::Tags::Initial<                                              \
          gr::Tags::Shift<3, Frame::Inertial, DTYPE(data)>>> /*meta*/,         \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;         \
  template tuples::TaggedTuple<                                                \
      ::Tags::FixedSource<gr::Tags::Shift<3, Frame::Inertial, DTYPE(data)>>>   \
  TovStar::variables(                                                          \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<::Tags::FixedSource<                                          \
          gr::Tags::Shift<3, Frame::Inertial, DTYPE(data)>>> /*meta*/,         \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;         \
  template tuples::TaggedTuple<                                                \
      Xcts::Tags::ShiftStrain<3, Frame::Inertial, DTYPE(data)>>                \
  TovStar::variables(                                                          \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<                                                              \
          Xcts::Tags::ShiftStrain<3, Frame::Inertial, DTYPE(data)>> /*meta*/,  \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;         \
  template tuples::TaggedTuple<::Tags::Initial<                                \
      Xcts::Tags::ShiftStrain<3, Frame::Inertial, DTYPE(data)>>>               \
  TovStar::variables(                                                          \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<::Tags::Initial<                                              \
          Xcts::Tags::ShiftStrain<3, Frame::Inertial, DTYPE(data)>>> /*meta*/, \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;         \
  template tuples::TaggedTuple<gr::Tags::EnergyDensity<DTYPE(data)>>           \
  TovStar::variables(                                                          \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<gr::Tags::EnergyDensity<DTYPE(data)>> /*meta*/,               \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;         \
  template tuples::TaggedTuple<gr::Tags::StressTrace<DTYPE(data)>>             \
  TovStar::variables(const tnsr::I<DTYPE(data), 3>& x,                         \
                     tmpl::list<gr::Tags::StressTrace<DTYPE(data)>> /*meta*/,  \
                     const RadialVariables<DTYPE(data)>& radial_vars)          \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      gr::Tags::MomentumDensity<3, Frame::Inertial, DTYPE(data)>>              \
  TovStar::variables(                                                          \
      const tnsr::I<DTYPE(data), 3>& x,                                        \
      tmpl::list<gr::Tags::MomentumDensity<3, Frame::Inertial,                 \
                                           DTYPE(data)>> /*meta*/,             \
      const RadialVariables<DTYPE(data)>& radial_vars) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_VARS, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE_VARS
}  // namespace Solutions
}  // namespace Xcts
/// \endcond
