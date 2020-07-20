// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/Xcts/Schwarzschild.hpp"

#include <ostream>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/RootFinding/NewtonRaphson.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {

template <typename DataType>
DataType kerr_schild_isotropic_radius_from_areal(
    const DataType& areal_radius) noexcept {
  const DataType one_over_lapse = sqrt(1. + 2. / areal_radius);
  return 0.25 * areal_radius * square(1. + one_over_lapse) *
         exp(2. - 2. * one_over_lapse);
}

template <typename DataType>
DataType kerr_schild_isotropic_radius_from_areal_deriv(
    const DataType& areal_radius) noexcept {
  const DataType one_over_lapse = sqrt(1. + 2. / areal_radius);
  const DataType sqrt_dr = -1. / one_over_lapse / square(areal_radius);
  return 0.25 *
         (square(1. + one_over_lapse) +
          2. * areal_radius * (1. + one_over_lapse) * sqrt_dr -
          2. * areal_radius * square(1. + one_over_lapse) * sqrt_dr) *
         exp(2. - 2. * one_over_lapse);
}

double kerr_schild_areal_radius_from_isotropic(
    const double isotropic_radius) noexcept {
  return RootFinder::newton_raphson(
      [&isotropic_radius](const double areal_radius) noexcept {
        return std::make_pair(
            kerr_schild_isotropic_radius_from_areal(areal_radius) -
                isotropic_radius,
            kerr_schild_isotropic_radius_from_areal_deriv(areal_radius));
      },
      isotropic_radius, 1., std::numeric_limits<double>::max(), 12);
}

DataVector kerr_schild_areal_radius_from_isotropic(
    const DataVector& isotropic_radius) noexcept {
  return RootFinder::newton_raphson(
      [&isotropic_radius](const double areal_radius, const size_t i) noexcept {
        return std::make_pair(
            kerr_schild_isotropic_radius_from_areal(areal_radius) -
                isotropic_radius[i],
            kerr_schild_isotropic_radius_from_areal_deriv(areal_radius));
      },
      isotropic_radius, make_with_value<DataVector>(isotropic_radius, 1.),
      make_with_value<DataVector>(isotropic_radius,
                                  std::numeric_limits<double>::max()),
      12);
}

}  // namespace

/// \cond
namespace Xcts::Solutions {

std::ostream& operator<<(std::ostream& os,
                         const SchwarzschildCoordinates& coords) noexcept {
  switch (coords) {
    case SchwarzschildCoordinates::Isotropic:
      return os << "Isotropic";
    case SchwarzschildCoordinates::PainleveGullstrand:
      return os << "PainleveGullstrand";
    case SchwarzschildCoordinates::KerrSchildIsotropic:
      return os << "KerrSchildIsotropic";
    default:
      ERROR("Unknown SchwarzschildCoordinates");
  }
}

template <>
double Schwarzschild<
    SchwarzschildCoordinates::Isotropic>::radius_at_horizon() noexcept {
  return 0.5;
}

template <>
double Schwarzschild<SchwarzschildCoordinates::PainleveGullstrand>::
    radius_at_horizon() noexcept {
  return 2.;
}

template <>
double Schwarzschild<SchwarzschildCoordinates::KerrSchildIsotropic>::
    radius_at_horizon() noexcept {
  return kerr_schild_isotropic_radius_from_areal(2.);
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

template <>
template <typename DataType>
tuples::TaggedTuple<gr::Tags::TraceExtrinsicCurvature<DataType>>
Schwarzschild<SchwarzschildCoordinates::PainleveGullstrand>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<gr::Tags::TraceExtrinsicCurvature<DataType>> /*meta*/) const
    noexcept {
  const DataType r = get(magnitude(x));
  return {Scalar<DataType>{1.5 * sqrt(2.) / pow(r, 1.5)}};
}

template <>
template <typename DataType>
tuples::TaggedTuple<gr::Tags::TraceExtrinsicCurvature<DataType>>
Schwarzschild<SchwarzschildCoordinates::KerrSchildIsotropic>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<gr::Tags::TraceExtrinsicCurvature<DataType>> /*meta*/) const
    noexcept {
  const DataType r = kerr_schild_areal_radius_from_isotropic(get(magnitude(x)));
  return {Scalar<DataType>{2. / square(r) * pow(1. + 2. / r, -1.5) *
                           (1. + 3. / r)}};
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

template <>
template <typename DataType>
tuples::TaggedTuple<::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataType>,
                                  tmpl::size_t<3>, Frame::Inertial>>
Schwarzschild<SchwarzschildCoordinates::PainleveGullstrand>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataType>,
                             tmpl::size_t<3>, Frame::Inertial>> /*meta*/) const
    noexcept {
  const DataType r = get(magnitude(x));
  const DataType isotropic_prefactor = -4.5 / sqrt(2.) / pow(r, 3.5);
  auto extrinsic_curvature_trace_gradient =
      make_with_value<tnsr::i<DataType, 3, Frame::Inertial>>(r, 0.);
  get<0>(extrinsic_curvature_trace_gradient) = isotropic_prefactor * get<0>(x);
  get<1>(extrinsic_curvature_trace_gradient) = isotropic_prefactor * get<1>(x);
  get<2>(extrinsic_curvature_trace_gradient) = isotropic_prefactor * get<2>(x);
  return {std::move(extrinsic_curvature_trace_gradient)};
}

template <>
template <typename DataType>
tuples::TaggedTuple<::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataType>,
                                  tmpl::size_t<3>, Frame::Inertial>>
Schwarzschild<SchwarzschildCoordinates::KerrSchildIsotropic>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataType>,
                             tmpl::size_t<3>, Frame::Inertial>> /*meta*/) const
    noexcept {
  const DataType rbar = get(magnitude(x));
  const DataType r = kerr_schild_areal_radius_from_isotropic(rbar);
  const DataType lapse = 1. / sqrt(1. + 2. / r);
  const DataType extrinsic_curvature_trace_dr_areal =
      2. * cube(lapse) / square(r) *
      (9. * square(lapse) / cube(r) + (3. * square(lapse) - 9.) / square(r) -
       2. / r);
  const DataType isotropic_prefactor =
      extrinsic_curvature_trace_dr_areal /
      kerr_schild_isotropic_radius_from_areal_deriv(r) / rbar;
  auto extrinsic_curvature_trace_gradient =
      make_with_value<tnsr::i<DataType, 3, Frame::Inertial>>(r, 0.);
  get<0>(extrinsic_curvature_trace_gradient) = isotropic_prefactor * get<0>(x);
  get<1>(extrinsic_curvature_trace_gradient) = isotropic_prefactor * get<1>(x);
  get<2>(extrinsic_curvature_trace_gradient) = isotropic_prefactor * get<2>(x);
  return {std::move(extrinsic_curvature_trace_gradient)};
}

// Conformal factor

template <>
template <typename DataType>
tuples::TaggedTuple<Xcts::Tags::ConformalFactor<DataType>>
Schwarzschild<SchwarzschildCoordinates::Isotropic>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<Xcts::Tags::ConformalFactor<DataType>> /*meta*/) const noexcept {
  const DataType r = get(magnitude(x));
  return {Scalar<DataType>{1. + 0.5 / r}};
}

template <>
template <typename DataType>
tuples::TaggedTuple<Xcts::Tags::ConformalFactor<DataType>>
Schwarzschild<SchwarzschildCoordinates::PainleveGullstrand>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<Xcts::Tags::ConformalFactor<DataType>> /*meta*/) const noexcept {
  return {make_with_value<Scalar<DataType>>(x, 1.)};
}

template <>
template <typename DataType>
tuples::TaggedTuple<Xcts::Tags::ConformalFactor<DataType>>
Schwarzschild<SchwarzschildCoordinates::KerrSchildIsotropic>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<Xcts::Tags::ConformalFactor<DataType>> /*meta*/) const noexcept {
  const DataType r = kerr_schild_areal_radius_from_isotropic(get(magnitude(x)));
  const DataType sqrt_one_plus_2_over_r = sqrt(1. + 2. / r);
  return {Scalar<DataType>{2. * exp(sqrt_one_plus_2_over_r - 1.) /
                           (1. + sqrt_one_plus_2_over_r)}};
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
  const DataType r = get(magnitude(x));
  const DataType isotropic_prefactor = -0.5 / cube(r);
  auto conformal_factor_gradient =
      make_with_value<tnsr::i<DataType, 3, Frame::Inertial>>(r, 0.);
  get<0>(conformal_factor_gradient) = isotropic_prefactor * get<0>(x);
  get<1>(conformal_factor_gradient) = isotropic_prefactor * get<1>(x);
  get<2>(conformal_factor_gradient) = isotropic_prefactor * get<2>(x);
  return {std::move(conformal_factor_gradient)};
}

template <>
template <typename DataType>
tuples::TaggedTuple<::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>,
                                  tmpl::size_t<3>, Frame::Inertial>>
Schwarzschild<SchwarzschildCoordinates::PainleveGullstrand>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>,
                             tmpl::size_t<3>, Frame::Inertial>> /*meta*/) const
    noexcept {
  return {make_with_value<tnsr::i<DataType, 3, Frame::Inertial>>(x, 0.)};
}

template <>
template <typename DataType>
tuples::TaggedTuple<::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>,
                                  tmpl::size_t<3>, Frame::Inertial>>
Schwarzschild<SchwarzschildCoordinates::KerrSchildIsotropic>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>,
                             tmpl::size_t<3>, Frame::Inertial>> /*meta*/) const
    noexcept {
  const DataType rbar = get(magnitude(x));
  const DataType r = kerr_schild_areal_radius_from_isotropic(rbar);
  const DataType one_over_lapse = sqrt(1. + 2. / r);
  const DataType conformal_factor_dr =
      -2. * exp(one_over_lapse - 1.) / square(1. + one_over_lapse) / square(r);
  const DataType isotropic_prefactor =
      conformal_factor_dr / kerr_schild_isotropic_radius_from_areal_deriv(r) /
      rbar;
  auto conformal_factor_gradient =
      make_with_value<tnsr::i<DataType, 3, Frame::Inertial>>(r, 0.);
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
  const DataType r = get(magnitude(x));
  return {Scalar<DataType>{1. - 0.5 / r}};
}

template <>
template <typename DataType>
tuples::TaggedTuple<Xcts::Tags::LapseTimesConformalFactor<DataType>>
Schwarzschild<SchwarzschildCoordinates::PainleveGullstrand>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<Xcts::Tags::LapseTimesConformalFactor<DataType>> /*meta*/) const
    noexcept {
  return {make_with_value<Scalar<DataType>>(x, 1.)};
}

template <>
template <typename DataType>
tuples::TaggedTuple<Xcts::Tags::LapseTimesConformalFactor<DataType>>
Schwarzschild<SchwarzschildCoordinates::KerrSchildIsotropic>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<Xcts::Tags::LapseTimesConformalFactor<DataType>> /*meta*/) const
    noexcept {
  const DataType r = kerr_schild_areal_radius_from_isotropic(get(magnitude(x)));
  const DataType lapse = 1. / sqrt(1. + 2. / r);
  const DataType conformal_factor =
      get(get<Xcts::Tags::ConformalFactor<DataType>>(
          variables(x, tmpl::list<Xcts::Tags::ConformalFactor<DataType>>{})));
  return {Scalar<DataType>{lapse * conformal_factor}};
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
  const DataType r = get(magnitude(x));
  auto conformal_factor_gradient =
      make_with_value<tnsr::i<DataType, 3, Frame::Inertial>>(r, 0.);
  const DataType isotropic_prefactor = 0.5 / cube(r);
  get<0>(conformal_factor_gradient) = isotropic_prefactor * get<0>(x);
  get<1>(conformal_factor_gradient) = isotropic_prefactor * get<1>(x);
  get<2>(conformal_factor_gradient) = isotropic_prefactor * get<2>(x);
  return {std::move(conformal_factor_gradient)};
}

template <>
template <typename DataType>
tuples::TaggedTuple<
    ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
                  tmpl::size_t<3>, Frame::Inertial>>
Schwarzschild<SchwarzschildCoordinates::PainleveGullstrand>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
                             tmpl::size_t<3>, Frame::Inertial>> /*meta*/) const
    noexcept {
  return {make_with_value<tnsr::i<DataType, 3, Frame::Inertial>>(x, 0.)};
}

template <>
template <typename DataType>
tuples::TaggedTuple<
    ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
                  tmpl::size_t<3>, Frame::Inertial>>
Schwarzschild<SchwarzschildCoordinates::KerrSchildIsotropic>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
                             tmpl::size_t<3>, Frame::Inertial>> /*meta*/) const
    noexcept {
  const DataType rbar = get(magnitude(x));
  const DataType r = kerr_schild_areal_radius_from_isotropic(rbar);
  const DataType lapse = 1. / sqrt(1. + 2. / r);
  const auto vars = variables(
      x, tmpl::list<Xcts::Tags::ConformalFactor<DataType>,
                    ::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>,
                                  tmpl::size_t<3>, Frame::Inertial>>{});
  const auto& conformal_factor =
      get(get<Xcts::Tags::ConformalFactor<DataType>>(vars));
  const DataType lapse_dr = cube(lapse) / square(r);
  const DataType isotropic_prefactor =
      conformal_factor * lapse_dr /
      kerr_schild_isotropic_radius_from_areal_deriv(r) / rbar;
  auto lapse_times_conformal_factor_gradient =
      get<::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>, tmpl::size_t<3>,
                        Frame::Inertial>>(vars);
  get<0>(lapse_times_conformal_factor_gradient) *= lapse;
  get<1>(lapse_times_conformal_factor_gradient) *= lapse;
  get<2>(lapse_times_conformal_factor_gradient) *= lapse;
  get<0>(lapse_times_conformal_factor_gradient) +=
      isotropic_prefactor * get<0>(x);
  get<1>(lapse_times_conformal_factor_gradient) +=
      isotropic_prefactor * get<1>(x);
  get<2>(lapse_times_conformal_factor_gradient) +=
      isotropic_prefactor * get<2>(x);
  return {std::move(lapse_times_conformal_factor_gradient)};
}

// Shift

template <SchwarzschildCoordinates Coords>
template <typename DataType>
tuples::TaggedTuple<Xcts::Tags::ShiftBackground<DataType, 3, Frame::Inertial>>
Schwarzschild<Coords>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<
        Xcts::Tags::ShiftBackground<DataType, 3, Frame::Inertial>> /*meta*/)
    const noexcept {
  return {make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.)};
}

template <>
template <typename DataType>
tuples::TaggedTuple<Xcts::Tags::ShiftExcess<DataType, 3, Frame::Inertial>>
Schwarzschild<SchwarzschildCoordinates::Isotropic>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<Xcts::Tags::ShiftExcess<DataType, 3, Frame::Inertial>> /*meta*/)
    const noexcept {
  return {make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.)};
}

template <>
template <typename DataType>
tuples::TaggedTuple<Xcts::Tags::ShiftExcess<DataType, 3, Frame::Inertial>>
Schwarzschild<SchwarzschildCoordinates::PainleveGullstrand>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<Xcts::Tags::ShiftExcess<DataType, 3, Frame::Inertial>> /*meta*/)
    const noexcept {
  const DataType r = get(magnitude(x));
  const DataType isotropic_prefactor = sqrt(2.) / pow(r, 1.5);
  auto shift = x;
  get<0>(shift) *= isotropic_prefactor;
  get<1>(shift) *= isotropic_prefactor;
  get<2>(shift) *= isotropic_prefactor;
  return {std::move(shift)};
}

template <>
template <typename DataType>
tuples::TaggedTuple<Xcts::Tags::ShiftExcess<DataType, 3, Frame::Inertial>>
Schwarzschild<SchwarzschildCoordinates::KerrSchildIsotropic>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<Xcts::Tags::ShiftExcess<DataType, 3, Frame::Inertial>> /*meta*/)
    const noexcept {
  const DataType rbar = get(magnitude(x));
  const DataType r = kerr_schild_areal_radius_from_isotropic(rbar);
  const DataType lapse = 1. / sqrt(1. + 2. / r);
  const DataType conformal_factor =
      get(get<Xcts::Tags::ConformalFactor<DataType>>(
          variables(x, tmpl::list<Xcts::Tags::ConformalFactor<DataType>>{})));
  const DataType isotropic_prefactor =
      2. * lapse / r / square(conformal_factor) / rbar;
  auto shift = x;
  get<0>(shift) *= isotropic_prefactor;
  get<1>(shift) *= isotropic_prefactor;
  get<2>(shift) *= isotropic_prefactor;
  return {std::move(shift)};
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

template <>
template <typename DataType>
tuples::TaggedTuple<Xcts::Tags::ShiftStrain<DataType, 3, Frame::Inertial>>
Schwarzschild<SchwarzschildCoordinates::PainleveGullstrand>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<Xcts::Tags::ShiftStrain<DataType, 3, Frame::Inertial>> /*meta*/)
    const noexcept {
  const DataType r = get(magnitude(x));
  const DataType diagonal_prefactor = sqrt(2.) / pow(r, 1.5);
  const DataType isotropic_prefactor = -1.5 * diagonal_prefactor / square(r);
  auto shift_strain =
      make_with_value<tnsr::ii<DataType, 3, Frame::Inertial>>(x, 0.);
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = i; j < 3; j++) {
      shift_strain.get(i, j) = isotropic_prefactor * x.get(i) * x.get(j);
    }
    shift_strain.get(i, i) += diagonal_prefactor;
  }
  return {std::move(shift_strain)};
}

template <>
template <typename DataType>
tuples::TaggedTuple<Xcts::Tags::ShiftStrain<DataType, 3, Frame::Inertial>>
Schwarzschild<SchwarzschildCoordinates::KerrSchildIsotropic>::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<Xcts::Tags::ShiftStrain<DataType, 3, Frame::Inertial>> /*meta*/)
    const noexcept {
  const DataType rbar = get(magnitude(x));
  const DataType r = kerr_schild_areal_radius_from_isotropic(rbar);
  const DataType lapse = 1. / sqrt(1. + 2. / r);
  const DataType conformal_factor =
      get(get<Xcts::Tags::ConformalFactor<DataType>>(
          variables(x, tmpl::list<Xcts::Tags::ConformalFactor<DataType>>{})));
  const DataType shift_magnitude = 2. * lapse / r / square(conformal_factor);
  const DataType shift_magnitude_dr =
      shift_magnitude / r *
      ((square(lapse) + 2. * lapse / (1. + lapse)) / r - 1.);
  const DataType isotropic_prefactor =
      (shift_magnitude_dr / kerr_schild_isotropic_radius_from_areal_deriv(r) -
       shift_magnitude / rbar) /
      square(rbar);
  const DataType diagonal_prefactor = shift_magnitude / rbar;
  auto shift_strain =
      make_with_value<tnsr::ii<DataType, 3, Frame::Inertial>>(x, 0.);
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = i; j < 3; j++) {
      shift_strain.get(i, j) = isotropic_prefactor * x.get(i) * x.get(j);
    }
    shift_strain.get(i, i) += diagonal_prefactor;
  }
  return {std::move(shift_strain)};
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
        Xcts::Tags::ShiftExcess<DataType, 3, Frame::Inertial>>> /*meta*/)
    const noexcept {
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
                        (SchwarzschildCoordinates::Isotropic,
                         SchwarzschildCoordinates::PainleveGullstrand,
                         SchwarzschildCoordinates::KerrSchildIsotropic))

#undef DTYPE
#undef INSTANTIATE

}  // namespace Xcts::Solutions
/// \endcond
