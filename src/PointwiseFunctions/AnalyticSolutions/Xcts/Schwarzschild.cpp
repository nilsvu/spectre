// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/Xcts/Schwarzschild.hpp"

#include <ostream>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Equations.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

/// \cond
namespace Xcts::Solutions {

std::ostream& operator<<(std::ostream& os,
                         const SchwarzschildCoordinates coords) noexcept {
  switch (coords) {
    case SchwarzschildCoordinates::Isotropic:
      return os << "Isotropic";
    default:
      ERROR("Unknown SchwarzschildCoordinates");
  }
}

template <SchwarzschildCoordinates Coords>
Schwarzschild<Coords>::Schwarzschild(const double mass) noexcept
    : mass_(mass) {}

template <SchwarzschildCoordinates Coords>
double Schwarzschild<Coords>::mass() const noexcept {
  return mass_;
}

template <>
double Schwarzschild<SchwarzschildCoordinates::Isotropic>::radius_at_horizon()
    const noexcept {
  return 0.5 * mass_;
}

// Conformal metric

template <SchwarzschildCoordinates Coords>
template <typename DataType>
tuples::TaggedTuple<Xcts::Tags::ConformalMetric<DataType, 3, Frame::Inertial>>
Schwarzschild<Coords>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<Xcts::Tags::ConformalMetric<DataType, 3,
                                           Frame::Inertial>> /*meta*/) const
    noexcept {
  auto conformal_metric =
      make_with_value<tnsr::ii<DataType, 3, Frame::Inertial>>(x, 0.);
  get<0, 0>(conformal_metric) = 1.;
  get<1, 1>(conformal_metric) = 1.;
  get<2, 2>(conformal_metric) = 1.;
  return {std::move(conformal_metric)};
}

// Extrinsic curvature trace

template <>
template <typename DataType>
tuples::TaggedTuple<gr::Tags::TraceExtrinsicCurvature<DataType>>
Schwarzschild<SchwarzschildCoordinates::Isotropic>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<gr::Tags::TraceExtrinsicCurvature<DataType>> /*meta*/) const
    noexcept {
  return {make_with_value<Scalar<DataType>>(x, 0.)};
}

// Extrinsic curvature trace gradient

template <>
template <typename DataType>
tuples::TaggedTuple<::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataType>,
                                  tmpl::size_t<3>, Frame::Inertial>>
Schwarzschild<SchwarzschildCoordinates::Isotropic>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataType>,
                             tmpl::size_t<3>, Frame::Inertial>> /*meta*/) const
    noexcept {
  return {make_with_value<tnsr::i<DataType, 3, Frame::Inertial>>(x, 0.)};
}

// Extrinsic curvature trace time derivative

template <SchwarzschildCoordinates Coords>
template <typename DataType>
tuples::TaggedTuple<::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>>>
Schwarzschild<Coords>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<
        ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>>> /*meta*/)
    const noexcept {
  return {make_with_value<Scalar<DataType>>(x, 0.)};
}

// Conformal factor

template <>
template <typename DataType>
tuples::TaggedTuple<Xcts::Tags::ConformalFactor<DataType>>
Schwarzschild<SchwarzschildCoordinates::Isotropic>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<Xcts::Tags::ConformalFactor<DataType>> /*meta*/) const noexcept {
  return {Scalar<DataType>{1. + 0.5 * mass_ / get(magnitude(x))}};
}

// Conformal factor gradient

template <>
template <typename DataType>
tuples::TaggedTuple<::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>,
                                  tmpl::size_t<3>, Frame::Inertial>>
Schwarzschild<SchwarzschildCoordinates::Isotropic>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>,
                             tmpl::size_t<3>, Frame::Inertial>> /*meta*/) const
    noexcept {
  const DataType isotropic_prefactor = -0.5 * mass_ / cube(get(magnitude(x)));
  auto conformal_factor_gradient =
      make_with_value<tnsr::i<DataType, 3, Frame::Inertial>>(x, 0.);
  get<0>(conformal_factor_gradient) = isotropic_prefactor * get<0>(x);
  get<1>(conformal_factor_gradient) = isotropic_prefactor * get<1>(x);
  get<2>(conformal_factor_gradient) = isotropic_prefactor * get<2>(x);
  return {std::move(conformal_factor_gradient)};
}

// Lapse (times conformal factor)

template <>
template <typename DataType>
tuples::TaggedTuple<Xcts::Tags::LapseTimesConformalFactor<DataType>>
Schwarzschild<SchwarzschildCoordinates::Isotropic>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<Xcts::Tags::LapseTimesConformalFactor<DataType>> /*meta*/) const
    noexcept {
  return {Scalar<DataType>{1. - 0.5 * mass_ / get(magnitude(x))}};
}

// Lapse (times conformal factor) gradient

template <>
template <typename DataType>
tuples::TaggedTuple<
    ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
                  tmpl::size_t<3>, Frame::Inertial>>
Schwarzschild<SchwarzschildCoordinates::Isotropic>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
                             tmpl::size_t<3>, Frame::Inertial>> /*meta*/) const
    noexcept {
  auto lapse_times_conformal_factor_gradient = get<::Tags::deriv<
      Xcts::Tags::ConformalFactor<DataType>, tmpl::size_t<3>, Frame::Inertial>>(
      variables(x,
                tmpl::list<::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>,
                                         tmpl::size_t<3>, Frame::Inertial>>{}));
  get<0>(lapse_times_conformal_factor_gradient) *= -1.;
  get<1>(lapse_times_conformal_factor_gradient) *= -1.;
  get<2>(lapse_times_conformal_factor_gradient) *= -1.;
  return {std::move(lapse_times_conformal_factor_gradient)};
}

// Shift background

template <SchwarzschildCoordinates Coords>
template <typename DataType>
tuples::TaggedTuple<Xcts::Tags::ShiftBackground<DataType, 3, Frame::Inertial>>
Schwarzschild<Coords>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<Xcts::Tags::ShiftBackground<DataType, 3,
                                           Frame::Inertial>> /*meta*/) const
    noexcept {
  return {make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.)};
}

template <SchwarzschildCoordinates Coords>
template <typename DataType>
tuples::TaggedTuple<
    Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
        DataType, 3, Frame::Inertial>>
Schwarzschild<Coords>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
        DataType, 3, Frame::Inertial>> /*meta*/) const noexcept {
  return {make_with_value<tnsr::II<DataType, 3, Frame::Inertial>>(x, 0.)};
}

template <SchwarzschildCoordinates Coords>
template <typename DataType>
tuples::TaggedTuple<
    ::Tags::div<Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
        DataType, 3, Frame::Inertial>>>
Schwarzschild<Coords>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<::Tags::div<
        Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
            DataType, 3, Frame::Inertial>>> /*meta*/) const noexcept {
  return {make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.)};
}

// Shift excess

template <>
template <typename DataType>
tuples::TaggedTuple<Xcts::Tags::ShiftExcess<DataType, 3, Frame::Inertial>>
Schwarzschild<SchwarzschildCoordinates::Isotropic>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<Xcts::Tags::ShiftExcess<DataType, 3, Frame::Inertial>> /*meta*/)
    const noexcept {
  return {make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.)};
}

// Shift strain

template <>
template <typename DataType>
tuples::TaggedTuple<Xcts::Tags::ShiftStrain<DataType, 3, Frame::Inertial>>
Schwarzschild<SchwarzschildCoordinates::Isotropic>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<Xcts::Tags::ShiftStrain<DataType, 3, Frame::Inertial>> /*meta*/)
    const noexcept {
  return {make_with_value<tnsr::ii<DataType, 3, Frame::Inertial>>(x, 0.)};
}

// Longitudinal shift square (for Hamiltonian and lapse equations only)

template <SchwarzschildCoordinates Coords>
template <typename DataType>
tuples::TaggedTuple<
    Xcts::Tags::LongitudinalShiftMinusDtConformalMetricSquare<DataType>>
Schwarzschild<Coords>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<Xcts::Tags::LongitudinalShiftMinusDtConformalMetricSquare<
        DataType>> /*meta*/) const noexcept {
  const auto shift_strain =
      get<Xcts::Tags::ShiftStrain<DataType, 3, Frame::Inertial>>(variables(
          x,
          tmpl::list<Xcts::Tags::ShiftStrain<DataType, 3, Frame::Inertial>>{}));
  auto longitudinal_shift = make_with_value<tnsr::II<DataType, 3>>(x, 0.);
  Xcts::euclidean_longitudinal_shift(make_not_null(&longitudinal_shift),
                                     shift_strain);
  auto longitudinal_shift_square = make_with_value<Scalar<DataType>>(x, 0.);
  // Using the tensor-contraction helper function from the XCTS namespace here,
  // which should be replaced with a tensor expression once those work
  Xcts::detail::fully_contract(make_not_null(&longitudinal_shift_square),
                               longitudinal_shift, longitudinal_shift);
  return {std::move(longitudinal_shift_square)};
}

// Longitudinal shift over lapse square (for Hamiltonian equation only)

template <SchwarzschildCoordinates Coords>
template <typename DataType>
tuples::TaggedTuple<
    Xcts::Tags::LongitudinalShiftMinusDtConformalMetricOverLapseSquare<
        DataType>>
Schwarzschild<Coords>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<
        Xcts::Tags::LongitudinalShiftMinusDtConformalMetricOverLapseSquare<
            DataType>> /*meta*/) const noexcept {
  auto vars = variables(
      x, tmpl::list<Xcts::Tags::ConformalFactor<DataType>,
                    Xcts::Tags::LapseTimesConformalFactor<DataType>,
                    Xcts::Tags::LongitudinalShiftMinusDtConformalMetricSquare<
                        DataType>>{});
  const auto& conformal_factor =
      get<Xcts::Tags::ConformalFactor<DataType>>(vars);
  const auto& lapse_times_conformal_factor =
      get<Xcts::Tags::LapseTimesConformalFactor<DataType>>(vars);
  auto& longitudinal_shift_over_lapse_square =
      get<Xcts::Tags::LongitudinalShiftMinusDtConformalMetricSquare<DataType>>(
          vars);
  get(longitudinal_shift_over_lapse_square) /=
      square(get(lapse_times_conformal_factor) / get(conformal_factor));
  return {std::move(longitudinal_shift_over_lapse_square)};
}

// Shift dot extrinsic curvature trace gradient (for lapse equation only)

template <SchwarzschildCoordinates Coords>
template <typename DataType>
tuples::TaggedTuple<Xcts::Tags::ShiftDotDerivExtrinsicCurvatureTrace<DataType>>
Schwarzschild<Coords>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<
        Xcts::Tags::ShiftDotDerivExtrinsicCurvatureTrace<DataType>> /*meta*/)
    const noexcept {
  auto vars = variables(
      x, tmpl::list<Xcts::Tags::ShiftExcess<DataType, 3, Frame::Inertial>,
                    ::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataType>,
                                  tmpl::size_t<3>, Frame::Inertial>>{});
  const auto& shift =
      get<Xcts::Tags::ShiftExcess<DataType, 3, Frame::Inertial>>(vars);
  const auto& extrinsic_curvature_trace_gradient =
      get<::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataType>,
                        tmpl::size_t<3>, Frame::Inertial>>(vars);
  return {dot_product(shift, extrinsic_curvature_trace_gradient)};
}

// Fixed sources (all zero)

template <SchwarzschildCoordinates Coords>
template <typename DataType>
tuples::TaggedTuple<::Tags::FixedSource<Xcts::Tags::ConformalFactor<DataType>>>
Schwarzschild<Coords>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<
        ::Tags::FixedSource<Xcts::Tags::ConformalFactor<DataType>>> /*meta*/)
    const noexcept {
  return {make_with_value<Scalar<DataType>>(x, 0.)};
}

template <SchwarzschildCoordinates Coords>
template <typename DataType>
tuples::TaggedTuple<
    ::Tags::FixedSource<Xcts::Tags::LapseTimesConformalFactor<DataType>>>
Schwarzschild<Coords>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<::Tags::FixedSource<
        Xcts::Tags::LapseTimesConformalFactor<DataType>>> /*meta*/) const
    noexcept {
  return {make_with_value<Scalar<DataType>>(x, 0.)};
}

template <SchwarzschildCoordinates Coords>
template <typename DataType>
tuples::TaggedTuple<
    ::Tags::FixedSource<Xcts::Tags::ShiftExcess<DataType, 3, Frame::Inertial>>>
Schwarzschild<Coords>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<::Tags::FixedSource<
        Xcts::Tags::ShiftExcess<DataType, 3, Frame::Inertial>>> /*meta*/) const
    noexcept {
  return {make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.)};
}

// Matter sources (all zero)

template <SchwarzschildCoordinates Coords>
template <typename DataType>
tuples::TaggedTuple<gr::Tags::EnergyDensity<DataType>>
Schwarzschild<Coords>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<gr::Tags::EnergyDensity<DataType>> /*meta*/) const noexcept {
  return {make_with_value<Scalar<DataType>>(x, 0.)};
}

template <SchwarzschildCoordinates Coords>
template <typename DataType>
tuples::TaggedTuple<gr::Tags::StressTrace<DataType>>
Schwarzschild<Coords>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<gr::Tags::StressTrace<DataType>> /*meta*/) const noexcept {
  return {make_with_value<Scalar<DataType>>(x, 0.)};
}

template <SchwarzschildCoordinates Coords>
template <typename DataType>
tuples::TaggedTuple<gr::Tags::MomentumDensity<3, Frame::Inertial, DataType>>
Schwarzschild<Coords>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<
        gr::Tags::MomentumDensity<3, Frame::Inertial, DataType>> /*meta*/) const
    noexcept {
  return {make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.)};
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define COORDS(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                   \
  template tuples::TaggedTuple<                                                \
      Xcts::Tags::ConformalMetric<DTYPE(data), 3, Frame::Inertial>>            \
  Schwarzschild<COORDS(data)>::variables(                                      \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<                                                              \
          Xcts::Tags::ConformalMetric<DTYPE(data), 3, Frame::Inertial>>)       \
      const noexcept;                                                          \
  template tuples::TaggedTuple<gr::Tags::TraceExtrinsicCurvature<DTYPE(data)>> \
  Schwarzschild<COORDS(data)>::variables(                                      \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<gr::Tags::TraceExtrinsicCurvature<DTYPE(data)>>)              \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      Xcts::Tags::LongitudinalShiftMinusDtConformalMetricSquare<DTYPE(data)>>  \
  Schwarzschild<COORDS(data)>::variables(                                      \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<Xcts::Tags::LongitudinalShiftMinusDtConformalMetricSquare<    \
          DTYPE(data)>>) const noexcept;                                       \
  template tuples::TaggedTuple<                                                \
      Xcts::Tags::LongitudinalShiftMinusDtConformalMetricOverLapseSquare<      \
          DTYPE(data)>>                                                        \
  Schwarzschild<COORDS(data)>::variables(                                      \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<                                                              \
          Xcts::Tags::LongitudinalShiftMinusDtConformalMetricOverLapseSquare<  \
              DTYPE(data)>>) const noexcept;                                   \
  template tuples::TaggedTuple<                                                \
      Xcts::Tags::ShiftDotDerivExtrinsicCurvatureTrace<DTYPE(data)>>           \
  Schwarzschild<COORDS(data)>::variables(                                      \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<                                                              \
          Xcts::Tags::ShiftDotDerivExtrinsicCurvatureTrace<DTYPE(data)>>)      \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DTYPE(data)>>>              \
  Schwarzschild<COORDS(data)>::variables(                                      \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DTYPE(data)>>>)  \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      ::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DTYPE(data)>,            \
                    tmpl::size_t<3>, Frame::Inertial>>                         \
  Schwarzschild<COORDS(data)>::variables(                                      \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DTYPE(data)>, \
                               tmpl::size_t<3>, Frame::Inertial>>)             \
      const noexcept;                                                          \
  template tuples::TaggedTuple<Xcts::Tags::ConformalFactor<DTYPE(data)>>       \
  Schwarzschild<COORDS(data)>::variables(                                      \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<Xcts::Tags::ConformalFactor<DTYPE(data)>>) const noexcept;    \
  template tuples::TaggedTuple<                                                \
      ::Tags::deriv<Xcts::Tags::ConformalFactor<DTYPE(data)>, tmpl::size_t<3>, \
                    Frame::Inertial>>                                          \
  Schwarzschild<COORDS(data)>::variables(                                      \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<::Tags::deriv<Xcts::Tags::ConformalFactor<DTYPE(data)>,       \
                               tmpl::size_t<3>, Frame::Inertial>>)             \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>>                      \
  Schwarzschild<COORDS(data)>::variables(                                      \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>>)          \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>,        \
                    tmpl::size_t<3>, Frame::Inertial>>                         \
  Schwarzschild<COORDS(data)>::variables(                                      \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<                                                              \
          ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>,    \
                        tmpl::size_t<3>, Frame::Inertial>>) const noexcept;    \
  template tuples::TaggedTuple<                                                \
      Xcts::Tags::ShiftBackground<DTYPE(data), 3, Frame::Inertial>>            \
  Schwarzschild<COORDS(data)>::variables(                                      \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<                                                              \
          Xcts::Tags::ShiftBackground<DTYPE(data), 3, Frame::Inertial>>)       \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<           \
          DTYPE(data), 3, Frame::Inertial>>                                    \
  Schwarzschild<COORDS(data)>::variables(                                      \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<                                                              \
          Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<       \
              DTYPE(data), 3, Frame::Inertial>>) const noexcept;               \
  template tuples::TaggedTuple<::Tags::div<                                    \
      Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<           \
          DTYPE(data), 3, Frame::Inertial>>>                                   \
  Schwarzschild<COORDS(data)>::variables(                                      \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<::Tags::div<                                                  \
          Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<       \
              DTYPE(data), 3, Frame::Inertial>>>) const noexcept;              \
  template tuples::TaggedTuple<                                                \
      Xcts::Tags::ShiftExcess<DTYPE(data), 3, Frame::Inertial>>                \
  Schwarzschild<COORDS(data)>::variables(                                      \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<Xcts::Tags::ShiftExcess<DTYPE(data), 3, Frame::Inertial>>)    \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      Xcts::Tags::ShiftStrain<DTYPE(data), 3, Frame::Inertial>>                \
  Schwarzschild<COORDS(data)>::variables(                                      \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<Xcts::Tags::ShiftStrain<DTYPE(data), 3, Frame::Inertial>>)    \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      ::Tags::FixedSource<Xcts::Tags::ConformalFactor<DTYPE(data)>>>           \
  Schwarzschild<COORDS(data)>::variables(                                      \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<                                                              \
          ::Tags::FixedSource<Xcts::Tags::ConformalFactor<DTYPE(data)>>>)      \
      const noexcept;                                                          \
  template tuples::TaggedTuple<                                                \
      ::Tags::FixedSource<Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>>> \
  Schwarzschild<COORDS(data)>::variables(                                      \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<::Tags::FixedSource<                                          \
          Xcts::Tags::LapseTimesConformalFactor<DTYPE(data)>>>)                \
      const noexcept;                                                          \
  template tuples::TaggedTuple<::Tags::FixedSource<                            \
      Xcts::Tags::ShiftExcess<DTYPE(data), 3, Frame::Inertial>>>               \
  Schwarzschild<COORDS(data)>::variables(                                      \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<::Tags::FixedSource<                                          \
          Xcts::Tags::ShiftExcess<DTYPE(data), 3, Frame::Inertial>>>)          \
      const noexcept;                                                          \
  template tuples::TaggedTuple<gr::Tags::EnergyDensity<DTYPE(data)>>           \
  Schwarzschild<COORDS(data)>::variables(                                      \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<gr::Tags::EnergyDensity<DTYPE(data)>>) const noexcept;        \
  template tuples::TaggedTuple<gr::Tags::StressTrace<DTYPE(data)>>             \
  Schwarzschild<COORDS(data)>::variables(                                      \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<gr::Tags::StressTrace<DTYPE(data)>>) const noexcept;          \
  template tuples::TaggedTuple<                                                \
      gr::Tags::MomentumDensity<3, Frame::Inertial, DTYPE(data)>>              \
  Schwarzschild<COORDS(data)>::variables(                                      \
      const tnsr::I<DTYPE(data), 3, Frame::Inertial>&,                         \
      tmpl::list<gr::Tags::MomentumDensity<3, Frame::Inertial, DTYPE(data)>>)  \
      const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector),
                        (SchwarzschildCoordinates::Isotropic))

template class Schwarzschild<SchwarzschildCoordinates::Isotropic>;

#undef DTYPE
#undef INSTANTIATE

}  // namespace Xcts::Solutions
/// \endcond
