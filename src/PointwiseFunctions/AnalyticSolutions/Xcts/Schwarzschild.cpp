// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/Xcts/Schwarzschild.hpp"

#include <algorithm>
#include <ostream>
#include <pup.h>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/CommonVariables.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

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

}  // namespace Xcts::Solutions

template <>
Xcts::Solutions::SchwarzschildCoordinates
Options::create_from_yaml<Xcts::Solutions::SchwarzschildCoordinates>::create<
    void>(const Options::Option& options) {
  const auto type_read = options.parse_as<std::string>();
  if ("Isotropic" == type_read) {
    return Xcts::Solutions::SchwarzschildCoordinates::Isotropic;
  }
  PARSE_ERROR(options.context(),
              "Failed to convert \""
                  << type_read
                  << "\" to Xcts::Solutions::SchwarzschildCoordinates. Must be "
                     "'Isotropic'.");
}

namespace Xcts::Solutions::detail {

SchwarzschildImpl::SchwarzschildImpl(
    const double mass,
    const SchwarzschildCoordinates coordinate_system) noexcept
    : mass_(mass), coordinate_system_(coordinate_system) {}

double SchwarzschildImpl::mass() const noexcept { return mass_; }

SchwarzschildCoordinates SchwarzschildImpl::coordinate_system() const noexcept {
  return coordinate_system_;
}

double SchwarzschildImpl::radius_at_horizon() const noexcept {
  switch (coordinate_system_) {
    case SchwarzschildCoordinates::Isotropic:
      return 0.5 * mass_;
    default:
      ERROR("Missing case for SchwarzschildCoordinates");
  }
}

void SchwarzschildImpl::pup(PUP::er& p) noexcept {
  p | mass_;
  p | coordinate_system_;
}

bool operator==(const SchwarzschildImpl& lhs,
                const SchwarzschildImpl& rhs) noexcept {
  return lhs.mass() == rhs.mass() and
         lhs.coordinate_system() == rhs.coordinate_system();
}

bool operator!=(const SchwarzschildImpl& lhs,
                const SchwarzschildImpl& rhs) noexcept {
  return not(lhs == rhs);
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> conformal_metric,
    const gsl::not_null<Cache*> /*cache*/,
    Tags::ConformalMetric<DataType, 3, Frame::Inertial> /*meta*/)
    const noexcept {
  get<0, 0>(*conformal_metric) = 1.;
  get<1, 1>(*conformal_metric) = 1.;
  get<2, 2>(*conformal_metric) = 1.;
  get<0, 1>(*conformal_metric) = 0.;
  get<0, 2>(*conformal_metric) = 0.;
  get<1, 2>(*conformal_metric) = 0.;
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<tnsr::II<DataType, 3>*> inv_conformal_metric,
    const gsl::not_null<Cache*> /*cache*/,
    Tags::InverseConformalMetric<DataType, 3, Frame::Inertial> /*meta*/)
    const noexcept {
  get<0, 0>(*inv_conformal_metric) = 1.;
  get<1, 1>(*inv_conformal_metric) = 1.;
  get<2, 2>(*inv_conformal_metric) = 1.;
  get<0, 1>(*inv_conformal_metric) = 0.;
  get<0, 2>(*inv_conformal_metric) = 0.;
  get<1, 2>(*inv_conformal_metric) = 0.;
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ijj<DataType, 3>*> deriv_conformal_metric,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<Tags::ConformalMetric<DataType, 3, Frame::Inertial>,
                  tmpl::size_t<3>, Frame::Inertial> /*meta*/) const noexcept {
  std::fill(deriv_conformal_metric->begin(), deriv_conformal_metric->end(), 0.);
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ijj<DataType, Dim>*>
        conformal_christoffel_first_kind,
    const gsl::not_null<Cache*> cache,
    Tags::ConformalChristoffelFirstKind<
        DataType, Dim, Frame::Inertial> /*meta*/) const noexcept {
  const auto& deriv_conformal_metric = cache->get_var(
      ::Tags::deriv<Tags::ConformalMetric<DataType, 3, Frame::Inertial>,
                    tmpl::size_t<3>, Frame::Inertial>{});
  gr::christoffel_first_kind(conformal_christoffel_first_kind,
                             deriv_conformal_metric);
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<tnsr::Ijj<DataType, Dim>*>
        conformal_christoffel_second_kind,
    const gsl::not_null<Cache*> cache,
    Tags::ConformalChristoffelSecondKind<
        DataType, Dim, Frame::Inertial> /*meta*/) const noexcept {
  const auto& conformal_christoffel_first_kind = cache->get_var(
      Tags::ConformalChristoffelFirstKind<DataType, Dim, Frame::Inertial>{});
  const auto& inv_conformal_metric = cache->get_var(
      Tags::InverseConformalMetric<DataType, 3, Frame::Inertial>{});
  raise_or_lower_first_index(conformal_christoffel_second_kind,
                             conformal_christoffel_first_kind,
                             inv_conformal_metric);
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<tnsr::i<DataType, Dim>*>
        conformal_christoffel_contracted,
    const gsl::not_null<Cache*> cache,
    Tags::ConformalChristoffelContracted<
        DataType, Dim, Frame::Inertial> /*meta*/) const noexcept {
  const auto& conformal_christoffel_second_kind = cache->get_var(
      Tags::ConformalChristoffelSecondKind<DataType, Dim, Frame::Inertial>{});
  for (size_t i = 0; i < Dim; ++i) {
    conformal_christoffel_contracted->get(i) =
        conformal_christoffel_second_kind.get(0, i, 0);
    for (size_t j = 1; j < Dim; ++j) {
      conformal_christoffel_contracted->get(i) +=
          conformal_christoffel_second_kind.get(j, i, j);
    }
  }
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> trace_extrinsic_curvature,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::TraceExtrinsicCurvature<DataType> /*meta*/) const noexcept {
  get(*trace_extrinsic_curvature) = 0.;
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> dt_trace_extrinsic_curvature,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>> /*meta*/)
    const noexcept {
  get(*dt_trace_extrinsic_curvature) = 0.;
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<tnsr::i<DataType, 3>*>
        trace_extrinsic_curvature_gradient,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataType>, tmpl::size_t<3>,
                  Frame::Inertial> /*meta*/) const noexcept {
  std::fill(trace_extrinsic_curvature_gradient->begin(),
            trace_extrinsic_curvature_gradient->end(), 0.);
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> conformal_factor,
    const gsl::not_null<Cache*> /*cache*/,
    Tags::ConformalFactor<DataType> /*meta*/) const noexcept {
  get(*conformal_factor) = 1. + 0.5 * mass / get(magnitude(x));
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<tnsr::i<DataType, 3>*> conformal_factor_gradient,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<Tags::ConformalFactor<DataType>, tmpl::size_t<3>,
                  Frame::Inertial> /*meta*/) const noexcept {
  const DataType isotropic_prefactor = -0.5 * mass / cube(get(magnitude(x)));
  get<0>(*conformal_factor_gradient) = isotropic_prefactor * get<0>(x);
  get<1>(*conformal_factor_gradient) = isotropic_prefactor * get<1>(x);
  get<2>(*conformal_factor_gradient) = isotropic_prefactor * get<2>(x);
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> conformal_factor_flux,
    const gsl::not_null<Cache*> cache,
    ::Tags::Flux<Tags::ConformalFactor<DataType>, tmpl::size_t<3>,
                 Frame::Inertial> /*meta*/) const noexcept {
  const auto& conformal_factor_gradient =
      cache->get_var(::Tags::deriv<Tags::ConformalFactor<DataType>,
                                   tmpl::size_t<3>, Frame::Inertial>{});
  get<0>(*conformal_factor_flux) = get<0>(conformal_factor_gradient);
  get<1>(*conformal_factor_flux) = get<1>(conformal_factor_gradient);
  get<2>(*conformal_factor_flux) = get<2>(conformal_factor_gradient);
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> lapse_times_conformal_factor,
    const gsl::not_null<Cache*> /*cache*/,
    Tags::LapseTimesConformalFactor<DataType> /*meta*/) const noexcept {
  get(*lapse_times_conformal_factor) = 1. - 0.5 * mass / get(magnitude(x));
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<tnsr::i<DataType, 3>*>
        lapse_times_conformal_factor_gradient,
    const gsl::not_null<Cache*> cache,
    ::Tags::deriv<Tags::LapseTimesConformalFactor<DataType>, tmpl::size_t<3>,
                  Frame::Inertial> /*meta*/) const noexcept {
  *lapse_times_conformal_factor_gradient =
      cache->get_var(::Tags::deriv<Tags::ConformalFactor<DataType>,
                                   tmpl::size_t<3>, Frame::Inertial>{});
  get<0>(*lapse_times_conformal_factor_gradient) *= -1.;
  get<1>(*lapse_times_conformal_factor_gradient) *= -1.;
  get<2>(*lapse_times_conformal_factor_gradient) *= -1.;
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*>
        lapse_times_conformal_factor_flux,
    const gsl::not_null<Cache*> cache,
    ::Tags::Flux<Tags::LapseTimesConformalFactor<DataType>, tmpl::size_t<3>,
                 Frame::Inertial> /*meta*/) const noexcept {
  const auto& lapse_times_conformal_factor_gradient =
      cache->get_var(::Tags::deriv<Tags::LapseTimesConformalFactor<DataType>,
                                   tmpl::size_t<3>, Frame::Inertial>{});
  get<0>(*lapse_times_conformal_factor_flux) =
      get<0>(lapse_times_conformal_factor_gradient);
  get<1>(*lapse_times_conformal_factor_flux) =
      get<1>(lapse_times_conformal_factor_gradient);
  get<2>(*lapse_times_conformal_factor_flux) =
      get<2>(lapse_times_conformal_factor_gradient);
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> shift_background,
    const gsl::not_null<Cache*> /*cache*/,
    Tags::ShiftBackground<DataType, 3, Frame::Inertial> /*meta*/)
    const noexcept {
  std::fill(shift_background->begin(), shift_background->end(), 0.);
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> shift_excess,
    const gsl::not_null<Cache*> /*cache*/,
    Tags::ShiftExcess<DataType, 3, Frame::Inertial> /*meta*/) const noexcept {
  std::fill(shift_excess->begin(), shift_excess->end(), 0.);
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> shift_dot_deriv_extrinsic_curvature_trace,
    const gsl::not_null<Cache*> /*cache*/,
    Tags::ShiftDotDerivExtrinsicCurvatureTrace<DataType> /*meta*/)
    const noexcept {
  std::fill(shift_dot_deriv_extrinsic_curvature_trace->begin(),
            shift_dot_deriv_extrinsic_curvature_trace->end(), 0.);
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<tnsr::II<DataType, 3>*> longitudinal_shift_excess,
    const gsl::not_null<Cache*> /*cache*/,
    Tags::LongitudinalShiftExcess<DataType, 3, Frame::Inertial> /*meta*/) const noexcept {
  std::fill(longitudinal_shift_excess->begin(),
            longitudinal_shift_excess->end(), 0.);
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*>
        longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
    const gsl::not_null<Cache*> /*cache*/,
    Tags::LongitudinalShiftMinusDtConformalMetricOverLapseSquare<
        DataType> /*meta*/) const noexcept {
  std::fill(
      longitudinal_shift_minus_dt_conformal_metric_over_lapse_square->begin(),
      longitudinal_shift_minus_dt_conformal_metric_over_lapse_square->end(),
      0.);
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*>
        longitudinal_shift_minus_dt_conformal_metric_square,
    const gsl::not_null<Cache*> /*cache*/,
    Tags::LongitudinalShiftMinusDtConformalMetricSquare<DataType> /*meta*/)
    const noexcept {
  std::fill(longitudinal_shift_minus_dt_conformal_metric_square->begin(),
            longitudinal_shift_minus_dt_conformal_metric_square->end(), 0.);
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<tnsr::II<DataType, 3, Frame::Inertial>*>
        longitudinal_shift_background_minus_dt_conformal_metric,
    const gsl::not_null<Cache*> /*cache*/,
    Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
        DataType, 3, Frame::Inertial> /*meta*/) const noexcept {
  std::fill(longitudinal_shift_background_minus_dt_conformal_metric->begin(),
            longitudinal_shift_background_minus_dt_conformal_metric->end(), 0.);
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3, Frame::Inertial>*>
        div_longitudinal_shift_background_minus_dt_conformal_metric,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::div<Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
        DataType, 3, Frame::Inertial>> /*meta*/) const noexcept {
  std::fill(
      div_longitudinal_shift_background_minus_dt_conformal_metric->begin(),
      div_longitudinal_shift_background_minus_dt_conformal_metric->end(), 0.);
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> energy_density,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::EnergyDensity<DataType> /*meta*/) const noexcept {
  std::fill(energy_density->begin(), energy_density->end(), 0.);
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> stress_trace,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::StressTrace<DataType> /*meta*/) const noexcept {
  std::fill(stress_trace->begin(), stress_trace->end(), 0.);
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> momentum_density,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::MomentumDensity<3, Frame::Inertial, DataType> /*meta*/)
    const noexcept {
  std::fill(momentum_density->begin(), momentum_density->end(), 0.);
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*>
        fixed_source_for_hamiltonian_constraint,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::FixedSource<Tags::ConformalFactor<DataType>> /*meta*/)
    const noexcept {
  std::fill(fixed_source_for_hamiltonian_constraint->begin(),
            fixed_source_for_hamiltonian_constraint->end(), 0.);
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> fixed_source_for_lapse_equation,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::FixedSource<Tags::LapseTimesConformalFactor<DataType>> /*meta*/)
    const noexcept {
  std::fill(fixed_source_for_lapse_equation->begin(),
            fixed_source_for_lapse_equation->end(), 0.);
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> fixed_source_momentum_constraint,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::FixedSource<
        Tags::ShiftExcess<DataType, 3, Frame::Inertial>> /*meta*/)
    const noexcept {
  std::fill(fixed_source_momentum_constraint->begin(),
            fixed_source_momentum_constraint->end(), 0.);
}

template class SchwarzschildVariables<double>;
template class SchwarzschildVariables<DataVector>;

}  // namespace Xcts::Solutions::detail

/// \endcond
