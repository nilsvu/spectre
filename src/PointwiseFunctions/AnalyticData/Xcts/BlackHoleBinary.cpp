// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/Xcts/BlackHoleBinary.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <pup.h>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"               // IWYU pragma: keep
#include "NumericalAlgorithms/RootFinding/NewtonRaphson.hpp"
#include "Options/Options.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

/// \cond
namespace Xcts {
namespace AnalyticData {

std::ostream& operator<<(std::ostream& os,
                         const BackgroundSpacetime& coords) noexcept {
  switch (coords) {
    case BackgroundSpacetime::FlatMaximallySliced:
      return os << "FlatMaximallySliced";
    case BackgroundSpacetime::FlatKerrSchildIsotropicSliced:
      return os << "FlatKerrSchildIsotropicSliced";
    case BackgroundSpacetime::SuperposedKerrSchild:
      return os << "SuperposedKerrSchild";
    case BackgroundSpacetime::SuperposedHarmonic:
      return os << "SuperposedHarmonic";
    default:
      ERROR("Unknown BackgroundSpacetime");
  }
}

template <BackgroundSpacetime Background>
BlackHoleBinary<Background>::BlackHoleBinary(
    const double mass_ratio, const double separation,
    const double angular_velocity) noexcept
    : mass_ratio_(mass_ratio),
      separation_(separation),
      angular_velocity_(angular_velocity) {
  if constexpr (Background == BackgroundSpacetime::FlatMaximallySliced or
                Background ==
                    BackgroundSpacetime::FlatKerrSchildIsotropicSliced) {
    isolated_solutions_ = {};
  } else if constexpr (Background ==
                       BackgroundSpacetime::SuperposedKerrSchild) {
    isolated_solutions_ = {{{1., {{0., 0., 0.}}, {{0., 0., 0.}}},
                            {1., {{0., 0., 0.}}, {{0., 0., 0.}}}}};
  }
}

// Extrinsic curvature trace

template <BackgroundSpacetime Background>
template <typename DataType>
tuples::TaggedTuple<gr::Tags::TraceExtrinsicCurvature<DataType>>
BlackHoleBinary<Background>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<gr::Tags::TraceExtrinsicCurvature<DataType>> /*meta*/)
    const noexcept {
  return {superposition<gr::Tags::TraceExtrinsicCurvature<DataType>>(x)};
}

// Extrinsic curvature trace gradient

template <BackgroundSpacetime Background>
template <typename DataType>
tuples::TaggedTuple<::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataType>,
                                  tmpl::size_t<3>, Frame::Inertial>>
BlackHoleBinary<Background>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataType>,
                             tmpl::size_t<3>, Frame::Inertial>> /*meta*/)
    const noexcept {
  return {
      superposition<::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataType>,
                                  tmpl::size_t<3>, Frame::Inertial>>(x)};
}

template <BackgroundSpacetime Background>
template <typename DataType>
tuples::TaggedTuple<::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>>>
BlackHoleBinary<Background>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<
        ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>>> /*meta*/)
    const noexcept {
  return {
      superposition<::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>>>(
          x)};
}

template <BackgroundSpacetime Background>
template <typename DataType>
tuples::TaggedTuple<Xcts::Tags::ShiftBackground<DataType, 3, Frame::Inertial>>
BlackHoleBinary<Background>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<
        Xcts::Tags::ShiftBackground<DataType, 3, Frame::Inertial>> /*meta*/)
    const noexcept {
  auto shift_background =
      make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.);
  get<0>(shift_background) = -angular_velocity_ * get<1>(x);
  get<1>(shift_background) = angular_velocity_ * get<0>(x);
  return {std::move(shift_background)};
}

template <BackgroundSpacetime Background>
template <typename DataType>
tuples::TaggedTuple<
    Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
        DataType, 3, Frame::Inertial>>
BlackHoleBinary<Background>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
        DataType, 3, Frame::Inertial>> /*meta*/) const noexcept {
  static_assert(
      Background == BackgroundSpacetime::FlatMaximallySliced or
          Background == BackgroundSpacetime::FlatKerrSchildIsotropicSliced,
      "Not yet implemented for this background spacetime");
  return {make_with_value<tnsr::II<DataType, 3>>(x, 0.)};
}

template <BackgroundSpacetime Background>
template <typename DataType>
tuples::TaggedTuple<
    ::Tags::div<Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
        DataType, 3, Frame::Inertial>>>
BlackHoleBinary<Background>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<::Tags::div<
        Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
            DataType, 3, Frame::Inertial>>> /*meta*/) const noexcept {
  static_assert(
      Background == BackgroundSpacetime::FlatMaximallySliced or
          Background == BackgroundSpacetime::FlatKerrSchildIsotropicSliced,
      "Not yet implemented for this background spacetime");
  return {make_with_value<tnsr::I<DataType, 3>>(x, 0.)};
}

// Initial conformal factor

template <BackgroundSpacetime Background>
template <typename DataType>
tuples::TaggedTuple<::Tags::Initial<Xcts::Tags::ConformalFactor<DataType>>>
BlackHoleBinary<Background>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<::Tags::Initial<Xcts::Tags::ConformalFactor<DataType>>> /*meta*/)
    const noexcept {
  return {superposition<Xcts::Tags::ConformalFactor<DataType>, true>(x)};
}

// Initial conformal factor gradient

template <BackgroundSpacetime Background>
template <typename DataType>
tuples::TaggedTuple<::Tags::Initial<::Tags::deriv<
    Xcts::Tags::ConformalFactor<DataType>, tmpl::size_t<3>, Frame::Inertial>>>
BlackHoleBinary<Background>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<::Tags::Initial<
        ::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>, tmpl::size_t<3>,
                      Frame::Inertial>>> /*meta*/) const noexcept {
  return {superposition<::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>,
                                      tmpl::size_t<3>, Frame::Inertial>,
                        true>(x)};
}

// Initial lapse (times conformal factor)

template <BackgroundSpacetime Background>
template <typename DataType>
tuples::TaggedTuple<
    ::Tags::Initial<Xcts::Tags::LapseTimesConformalFactor<DataType>>>
BlackHoleBinary<Background>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<::Tags::Initial<
        Xcts::Tags::LapseTimesConformalFactor<DataType>>> /*meta*/)
    const noexcept {
  return {
      superposition<Xcts::Tags::LapseTimesConformalFactor<DataType>, true>(x)};
}

// Initial lapse (times conformal factor) gradient

template <BackgroundSpacetime Background>
template <typename DataType>
tuples::TaggedTuple<::Tags::Initial<
    ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
                  tmpl::size_t<3>, Frame::Inertial>>>
BlackHoleBinary<Background>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<::Tags::Initial<
        ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
                      tmpl::size_t<3>, Frame::Inertial>>> /*meta*/)
    const noexcept {
  return {superposition<
      ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
                    tmpl::size_t<3>, Frame::Inertial>,
      true>(x)};
}

// Shift

template <BackgroundSpacetime Background>
template <typename DataType>
tuples::TaggedTuple<
    ::Tags::Initial<Xcts::Tags::ShiftExcess<DataType, 3, Frame::Inertial>>>
BlackHoleBinary<Background>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<::Tags::Initial<
        Xcts::Tags::ShiftExcess<DataType, 3, Frame::Inertial>>> /*meta*/)
    const noexcept {
  auto shift_excess =
      superposition<Xcts::Tags::ShiftExcess<DataType, 3, Frame::Inertial>,
                    true>(x);
  //   const auto shift_background = get<
  //       Xcts::Tags::ShiftBackground<DataType, 3, Frame::Inertial>>(variables(
  //       x,
  //       tmpl::list<Xcts::Tags::ShiftBackground<DataType, 3,
  //       Frame::Inertial>>{}));
  //   for (size_t i = 0; i < 3; ++i) {
  //     shift_excess.get(i) -= shift_background.get(i);
  //   }
  return {std::move(shift_excess)};
}

// Shift strain

template <BackgroundSpacetime Background>
template <typename DataType>
tuples::TaggedTuple<
    ::Tags::Initial<Xcts::Tags::ShiftStrain<DataType, 3, Frame::Inertial>>>
BlackHoleBinary<Background>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<::Tags::Initial<
        Xcts::Tags::ShiftStrain<DataType, 3, Frame::Inertial>>> /*meta*/)
    const noexcept {
  return {superposition<Xcts::Tags::ShiftStrain<DataType, 3, Frame::Inertial>,
                        true>(x)};
}

// Fixed sources (all zero)

template <BackgroundSpacetime Background>
template <typename DataType>
tuples::TaggedTuple<::Tags::FixedSource<Xcts::Tags::ConformalFactor<DataType>>>
BlackHoleBinary<Background>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<
        ::Tags::FixedSource<Xcts::Tags::ConformalFactor<DataType>>> /*meta*/)
    const noexcept {
  return {make_with_value<Scalar<DataType>>(x, 0.)};
}

template <BackgroundSpacetime Background>
template <typename DataType>
tuples::TaggedTuple<
    ::Tags::FixedSource<Xcts::Tags::LapseTimesConformalFactor<DataType>>>
BlackHoleBinary<Background>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<::Tags::FixedSource<
        Xcts::Tags::LapseTimesConformalFactor<DataType>>> /*meta*/)
    const noexcept {
  return {make_with_value<Scalar<DataType>>(x, 0.)};
}

template <BackgroundSpacetime Background>
template <typename DataType>
tuples::TaggedTuple<
    ::Tags::FixedSource<Xcts::Tags::ShiftExcess<DataType, 3, Frame::Inertial>>>
BlackHoleBinary<Background>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<::Tags::FixedSource<
        Xcts::Tags::ShiftExcess<DataType, 3, Frame::Inertial>>> /*meta*/)
    const noexcept {
  return {make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.)};
}

// Matter sources (all zero)

template <BackgroundSpacetime Background>
template <typename DataType>
tuples::TaggedTuple<gr::Tags::EnergyDensity<DataType>>
BlackHoleBinary<Background>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<gr::Tags::EnergyDensity<DataType>> /*meta*/) const noexcept {
  return {make_with_value<Scalar<DataType>>(x, 0.)};
}

template <BackgroundSpacetime Background>
template <typename DataType>
tuples::TaggedTuple<gr::Tags::StressTrace<DataType>>
BlackHoleBinary<Background>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<gr::Tags::StressTrace<DataType>> /*meta*/) const noexcept {
  return {make_with_value<Scalar<DataType>>(x, 0.)};
}

template <BackgroundSpacetime Background>
template <typename DataType>
tuples::TaggedTuple<gr::Tags::MomentumDensity<3, Frame::Inertial, DataType>>
BlackHoleBinary<Background>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<gr::Tags::MomentumDensity<3, Frame::Inertial,
                                         DataType>> /*meta*/) const noexcept {
  return {make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.)};
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define BG(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                   \
  template tuples::TaggedTuple<gr::Tags::TraceExtrinsicCurvature<DTYPE(data)>> \
  BlackHoleBinary<BG(data)>::variables(                                        \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<gr::Tags::TraceExtrinsicCurvature<DTYPE(data)>>)              \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      ::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DTYPE(data)>,            \
                    tmpl::size_t<3>, Frame::Inertial>>                         \
  BlackHoleBinary<BG(data)>::variables(                                        \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DTYPE(data)>, \
                               tmpl::size_t<3>, Frame::Inertial>>)             \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DTYPE(data)>>>              \
  BlackHoleBinary<BG(data)>::variables(                                        \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DTYPE(data)>>>)  \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      Xcts::Tags::ShiftBackground<DTYPE(data), 3, Frame::Inertial>>            \
  BlackHoleBinary<BG(data)>::variables(                                        \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<                                                              \
          Xcts::Tags::ShiftBackground<DTYPE(data), 3, Frame::Inertial>>)       \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<           \
          DTYPE(data), 3, Frame::Inertial>>                                    \
  BlackHoleBinary<BG(data)>::variables(                                        \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<                                                              \
          Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<       \
              DTYPE(data), 3, Frame::Inertial>>) const noexcept;               \
  template tuples::TaggedTuple<::Tags::div<                                    \
      Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<           \
          DTYPE(data), 3, Frame::Inertial>>>                                   \
  BlackHoleBinary<BG(data)>::variables(                                        \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<::Tags::div<                                                  \
          Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<       \
              DTYPE(data), 3, Frame::Inertial>>>) const noexcept;              \
  template tuples::TaggedTuple<                                                \
      ::Tags::Initial<Xcts::Tags::ConformalFactor<DTYPE(data)>>>               \
  BlackHoleBinary<BG(data)>::variables(                                        \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<::Tags::Initial<Xcts::Tags::ConformalFactor<DTYPE(data)>>>)   \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      ::Tags::Initial<::Tags::deriv<Xcts::Tags::ConformalFactor<DTYPE(data)>,  \
                                    tmpl::size_t<3>, Frame::Inertial>>>        \
  BlackHoleBinary<BG(data)>::variables(                                        \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<::Tags::Initial<                                              \
          ::Tags::deriv<Xcts::Tags::ConformalFactor<DTYPE(data)>,              \
                        tmpl::size_t<3>, Frame::Inertial>>>) const noexcept;   \
  template tuples::TaggedTuple<                                                \
      ::Tags::Initial<Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>>>     \
  BlackHoleBinary<BG(data)>::variables(                                        \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<::Tags::Initial<                                              \
          Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>>>)                \
      const noexcept;                                                          \
  template tuples::TaggedTuple<::Tags::Initial<                                \
      ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>,        \
                    tmpl::size_t<3>, Frame::Inertial>>>                        \
  BlackHoleBinary<BG(data)>::variables(                                        \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<::Tags::Initial<                                              \
          ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>,    \
                        tmpl::size_t<3>, Frame::Inertial>>>) const noexcept;   \
  template tuples::TaggedTuple<::Tags::Initial<                                \
      Xcts::Tags::ShiftExcess<DTYPE(data), 3, Frame::Inertial>>>               \
  BlackHoleBinary<BG(data)>::variables(                                        \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<::Tags::Initial<                                              \
          Xcts::Tags::ShiftExcess<DTYPE(data), 3, Frame::Inertial>>>)          \
      const noexcept;                                                          \
  template tuples::TaggedTuple<::Tags::Initial<                                \
      Xcts::Tags::ShiftStrain<DTYPE(data), 3, Frame::Inertial>>>               \
  BlackHoleBinary<BG(data)>::variables(                                        \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<::Tags::Initial<                                              \
          Xcts::Tags::ShiftStrain<DTYPE(data), 3, Frame::Inertial>>>)          \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      ::Tags::FixedSource<Xcts::Tags::ConformalFactor<DTYPE(data)>>>           \
  BlackHoleBinary<BG(data)>::variables(                                        \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<                                                              \
          ::Tags::FixedSource<Xcts::Tags::ConformalFactor<DTYPE(data)>>>)      \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      ::Tags::FixedSource<Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>>> \
  BlackHoleBinary<BG(data)>::variables(                                        \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<::Tags::FixedSource<                                          \
          Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>>>)                \
      const noexcept;                                                          \
  template tuples::TaggedTuple<::Tags::FixedSource<                            \
      Xcts::Tags::ShiftExcess<DTYPE(data), 3, Frame::Inertial>>>               \
  BlackHoleBinary<BG(data)>::variables(                                        \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<::Tags::FixedSource<                                          \
          Xcts::Tags::ShiftExcess<DTYPE(data), 3, Frame::Inertial>>>)          \
      const noexcept;                                                          \
  template tuples::TaggedTuple<gr::Tags::EnergyDensity<DTYPE(data)>>           \
  BlackHoleBinary<BG(data)>::variables(                                        \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<gr::Tags::EnergyDensity<DTYPE(data)>>) const noexcept;        \
  template tuples::TaggedTuple<gr::Tags::StressTrace<DTYPE(data)>>             \
  BlackHoleBinary<BG(data)>::variables(                                        \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<gr::Tags::StressTrace<DTYPE(data)>>) const noexcept;          \
  template tuples::TaggedTuple<                                                \
      gr::Tags::MomentumDensity<3, Frame::Inertial, DTYPE(data)>>              \
  BlackHoleBinary<BG(data)>::variables(                                        \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<gr::Tags::MomentumDensity<3, Frame::Inertial, DTYPE(data)>>)  \
      const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector),
                        (BackgroundSpacetime::FlatMaximallySliced,
                         BackgroundSpacetime::FlatKerrSchildIsotropicSliced))
template class BlackHoleBinary<BackgroundSpacetime::FlatMaximallySliced>;
template class BlackHoleBinary<
    BackgroundSpacetime::FlatKerrSchildIsotropicSliced>;

#undef DTYPE
#undef INSTANTIATE

}  // namespace AnalyticData
}  // namespace Xcts
/// \endcond
