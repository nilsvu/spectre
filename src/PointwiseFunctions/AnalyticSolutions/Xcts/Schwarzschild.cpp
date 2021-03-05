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
#include "Elliptic/Systems/Xcts/Equations.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/RootFinding/NewtonRaphson.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/CommonVariables.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Xcts/LongitudinalOperator.hpp"
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
    case SchwarzschildCoordinates::PainleveGullstrand:
      return os << "PainleveGullstrand";
    case SchwarzschildCoordinates::KerrSchildIsotropic:
      return os << "KerrSchildIsotropic";
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
  } else if ("PainleveGullstrand" == type_read) {
    return Xcts::Solutions::SchwarzschildCoordinates::PainleveGullstrand;
  } else if ("KerrSchildIsotropic" == type_read) {
    return Xcts::Solutions::SchwarzschildCoordinates::KerrSchildIsotropic;
  }
  PARSE_ERROR(options.context(),
              "Failed to convert \""
                  << type_read
                  << "\" to Xcts::Solutions::SchwarzschildCoordinates. Must be "
                     "one of 'Isotropic', 'PainleveGullstrand', "
                     "'KerrSchildIsotropic'.");
}

namespace Xcts::Solutions::detail {

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
    case SchwarzschildCoordinates::PainleveGullstrand:
      return 2. * mass_;
    case SchwarzschildCoordinates::KerrSchildIsotropic:
      return kerr_schild_isotropic_radius_from_areal(2.) * mass_;
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
    const gsl::not_null<Scalar<DataType>*> isotropic_radius,
    const gsl::not_null<Cache*> /*cache*/,
    IsotropicRadius /*meta*/)
    const noexcept {
  magnitude(isotropic_radius, x);
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> areal_radius,
    const gsl::not_null<Cache*> cache, ArealRadius /*meta*/) const noexcept {
  const auto& isotropic_radius = cache->get_var(IsotropicRadius{});
  get(*areal_radius) =
      kerr_schild_areal_radius_from_isotropic(get(isotropic_radius));
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> deriv_isotropic_radius_from_areal,
    const gsl::not_null<Cache*> cache,
    DerivIsotropicRadiusFromAreal /*meta*/) const noexcept {
  const auto& areal_radius = cache->get_var(ArealRadius{});
  get(*deriv_isotropic_radius_from_areal) =
      kerr_schild_isotropic_radius_from_areal_deriv(get(areal_radius));
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
    const gsl::not_null<Cache*> cache,
    gr::Tags::TraceExtrinsicCurvature<DataType> /*meta*/) const noexcept {
  switch (coordinate_system) {
    case SchwarzschildCoordinates::Isotropic: {
      get(*trace_extrinsic_curvature) = 0.;
      break;
    }
    case SchwarzschildCoordinates::PainleveGullstrand: {
      const auto& r = get(cache->get_var(IsotropicRadius{}));
      get(*trace_extrinsic_curvature) = 1.5 * sqrt(2.) / pow(r, 1.5);
      break;
    }
    case SchwarzschildCoordinates::KerrSchildIsotropic: {
      const auto& r = get(cache->get_var(ArealRadius{}));
      get(*trace_extrinsic_curvature) =
          2. / square(r) * pow(1. + 2. / r, -1.5) * (1. + 3. / r);
      break;
    }
    default:
      ERROR("Missing case for SchwarzschildCoordinates");
  }
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
    const gsl::not_null<Scalar<DataType>*> conformal_factor,
    const gsl::not_null<Cache*> cache,
    Tags::ConformalFactor<DataType> /*meta*/) const noexcept {
  switch (coordinate_system) {
    case SchwarzschildCoordinates::Isotropic: {
      const auto& r = get(cache->get_var(IsotropicRadius{}));
      get(*conformal_factor) = 1. + 0.5 * mass / r;
      break;
    }
    case SchwarzschildCoordinates::PainleveGullstrand: {
      get(*conformal_factor) = 1.;
      break;
    }
    case SchwarzschildCoordinates::KerrSchildIsotropic: {
      const auto& r = get(cache->get_var(ArealRadius{}));
      get(*conformal_factor) =
          2. * exp(sqrt(1. + 2. / r) - 1.) / (1. + sqrt(1. + 2. / r));
      break;
    }
    default:
      ERROR("Missing case for SchwarzschildCoordinates");
  }
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<tnsr::i<DataType, 3>*> conformal_factor_gradient,
    const gsl::not_null<Cache*> cache,
    ::Tags::deriv<Tags::ConformalFactor<DataType>, tmpl::size_t<3>,
                  Frame::Inertial> /*meta*/) const noexcept {
  switch (coordinate_system) {
    case SchwarzschildCoordinates::Isotropic: {
      const auto& r = get(cache->get_var(IsotropicRadius{}));
      const DataType isotropic_prefactor = -0.5 * mass / cube(r);
      get<0>(*conformal_factor_gradient) = isotropic_prefactor * get<0>(x);
      get<1>(*conformal_factor_gradient) = isotropic_prefactor * get<1>(x);
      get<2>(*conformal_factor_gradient) = isotropic_prefactor * get<2>(x);
      break;
    }
    case SchwarzschildCoordinates::PainleveGullstrand: {
      std::fill(conformal_factor_gradient->begin(),
                conformal_factor_gradient->end(), 0.);
      break;
    }
    case SchwarzschildCoordinates::KerrSchildIsotropic: {
      const auto& rbar = get(cache->get_var(IsotropicRadius{}));
      const auto& r = get(cache->get_var(ArealRadius{}));
      const auto& deriv_rbar_from_r =
          get(cache->get_var(DerivIsotropicRadiusFromAreal{}));
      const auto one_over_lapse = sqrt(1. + 2. / r);
      const auto conformal_factor_dr = -2. * exp(one_over_lapse - 1.) /
                                       square(1. + one_over_lapse) / square(r);
      const DataType isotropic_prefactor =
          conformal_factor_dr / deriv_rbar_from_r / rbar;
      get<0>(*conformal_factor_gradient) = isotropic_prefactor * get<0>(x);
      get<1>(*conformal_factor_gradient) = isotropic_prefactor * get<1>(x);
      get<2>(*conformal_factor_gradient) = isotropic_prefactor * get<2>(x);
      break;
    }
    default:
      ERROR("Missing case for SchwarzschildCoordinates");
  }
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
    const gsl::not_null<Scalar<DataType>*> lapse,
    const gsl::not_null<Cache*> cache,
    gr::Tags::Lapse<DataType> /*meta*/) const noexcept {
  switch (coordinate_system) {
    case SchwarzschildCoordinates::Isotropic: {
      const auto& r = get(cache->get_var(IsotropicRadius{}));
      get(*lapse) = (1. - 0.5 * mass / r) / (1. + 0.5 * mass / r);
      break;
    }
    case SchwarzschildCoordinates::PainleveGullstrand: {
      get(*lapse) = 1.;
      break;
    }
    case SchwarzschildCoordinates::KerrSchildIsotropic: {
      const auto& r = get(cache->get_var(ArealRadius{}));
      get(*lapse) = 1. / sqrt(1. + 2. / r);
      break;
    }
    default:
      ERROR("Missing case for SchwarzschildCoordinates");
  }
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> lapse_times_conformal_factor,
    const gsl::not_null<Cache*> cache,
    Tags::LapseTimesConformalFactor<DataType> /*meta*/) const noexcept {
  switch (coordinate_system) {
    case SchwarzschildCoordinates::Isotropic: {
      const auto& r = get(cache->get_var(IsotropicRadius{}));
      get(*lapse_times_conformal_factor) = 1. - 0.5 * mass / r;
      break;
    }
    case SchwarzschildCoordinates::PainleveGullstrand: {
      get(*lapse_times_conformal_factor) = 1.;
      break;
    }
    case SchwarzschildCoordinates::KerrSchildIsotropic: {
      const auto& lapse = get(cache->get_var(gr::Tags::Lapse<DataType>{}));
      const auto& conformal_factor =
          get(cache->get_var(Tags::ConformalFactor<DataType>{}));
      get(*lapse_times_conformal_factor) = lapse * conformal_factor;
      break;
    }
    default:
      ERROR("Missing case for SchwarzschildCoordinates");
  }
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<tnsr::i<DataType, 3>*>
        lapse_times_conformal_factor_gradient,
    const gsl::not_null<Cache*> cache,
    ::Tags::deriv<Tags::LapseTimesConformalFactor<DataType>, tmpl::size_t<3>,
                  Frame::Inertial> /*meta*/) const noexcept {
  switch (coordinate_system) {
    case SchwarzschildCoordinates::Isotropic: {
      *lapse_times_conformal_factor_gradient =
          cache->get_var(::Tags::deriv<Tags::ConformalFactor<DataType>,
                                       tmpl::size_t<3>, Frame::Inertial>{});
      get<0>(*lapse_times_conformal_factor_gradient) *= -1.;
      get<1>(*lapse_times_conformal_factor_gradient) *= -1.;
      get<2>(*lapse_times_conformal_factor_gradient) *= -1.;
      break;
    }
    case SchwarzschildCoordinates::PainleveGullstrand: {
      std::fill(lapse_times_conformal_factor_gradient->begin(),
                lapse_times_conformal_factor_gradient->end(), 0.);
      break;
    }
    case SchwarzschildCoordinates::KerrSchildIsotropic: {
      const auto& rbar = get(cache->get_var(IsotropicRadius{}));
      const auto& r = get(cache->get_var(ArealRadius{}));
      const auto& deriv_rbar_from_r =
          get(cache->get_var(DerivIsotropicRadiusFromAreal{}));
      const auto& conformal_factor =
          get(cache->get_var(Xcts::Tags::ConformalFactor<DataType>{}));
      const auto& conformal_factor_gradient =
          cache->get_var(::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>,
                                       tmpl::size_t<3>, Frame::Inertial>{});
      const auto& lapse = get(cache->get_var(gr::Tags::Lapse<DataType>{}));
      const auto lapse_dr = cube(lapse) / square(r);
      const DataType isotropic_prefactor =
          conformal_factor * lapse_dr / deriv_rbar_from_r / rbar;
      *lapse_times_conformal_factor_gradient = conformal_factor_gradient;
      get<0>(*lapse_times_conformal_factor_gradient) *= lapse;
      get<1>(*lapse_times_conformal_factor_gradient) *= lapse;
      get<2>(*lapse_times_conformal_factor_gradient) *= lapse;
      get<0>(*lapse_times_conformal_factor_gradient) +=
          isotropic_prefactor * get<0>(x);
      get<1>(*lapse_times_conformal_factor_gradient) +=
          isotropic_prefactor * get<1>(x);
      get<2>(*lapse_times_conformal_factor_gradient) +=
          isotropic_prefactor * get<2>(x);
      break;
    }
    default:
      ERROR("Missing case for SchwarzschildCoordinates");
  }
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
    const gsl::not_null<Cache*> cache,
    Tags::ShiftExcess<DataType, 3, Frame::Inertial> /*meta*/) const noexcept {
  switch (coordinate_system) {
    case SchwarzschildCoordinates::Isotropic: {
      std::fill(shift_excess->begin(), shift_excess->end(), 0.);
      break;
    }
    case SchwarzschildCoordinates::PainleveGullstrand: {
      const auto& r = get(cache->get_var(IsotropicRadius{}));
      const DataType isotropic_prefactor = sqrt(2.) / pow(r, 1.5);
      *shift_excess = x;
      get<0>(*shift_excess) *= isotropic_prefactor;
      get<1>(*shift_excess) *= isotropic_prefactor;
      get<2>(*shift_excess) *= isotropic_prefactor;
      break;
    }
    case SchwarzschildCoordinates::KerrSchildIsotropic: {
      const auto& rbar = get(cache->get_var(IsotropicRadius{}));
      const auto& r = get(cache->get_var(ArealRadius{}));
      const auto& lapse = get(cache->get_var(gr::Tags::Lapse<DataType>{}));
      const auto& conformal_factor =
          get(cache->get_var(Xcts::Tags::ConformalFactor<DataType>{}));
      const DataType isotropic_prefactor =
          2. * lapse / r / square(conformal_factor) / rbar;
      *shift_excess = x;
      get<0>(*shift_excess) *= isotropic_prefactor;
      get<1>(*shift_excess) *= isotropic_prefactor;
      get<2>(*shift_excess) *= isotropic_prefactor;
      break;
    }
    default:
      ERROR("Missing case for SchwarzschildCoordinates");
  }
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> shift,
    const gsl::not_null<Cache*> cache,
    gr::Tags::Shift<3, Frame::Inertial, DataType> /*meta*/) const noexcept {
  *shift = cache->get_var(Tags::ShiftExcess<DataType, Dim, Frame::Inertial>{});
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> shift_strain,
    const gsl::not_null<Cache*> cache,
    Tags::ShiftStrain<DataType, 3, Frame::Inertial> /*meta*/) const noexcept {
  switch (coordinate_system) {
    case SchwarzschildCoordinates::Isotropic: {
      std::fill(shift_strain->begin(), shift_strain->end(), 0.);
      break;
    }
    case SchwarzschildCoordinates::PainleveGullstrand: {
      const auto& r = get(cache->get_var(IsotropicRadius{}));
      const DataType diagonal_prefactor = sqrt(2.) / pow(r, 1.5);
      const DataType isotropic_prefactor =
          -1.5 * diagonal_prefactor / square(r);
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = i; j < 3; ++j) {
          shift_strain->get(i, j) = isotropic_prefactor * x.get(i) * x.get(j);
        }
        shift_strain->get(i, i) += diagonal_prefactor;
      }
      break;
    }
    case SchwarzschildCoordinates::KerrSchildIsotropic: {
      const auto& rbar = get(cache->get_var(IsotropicRadius{}));
      const auto& r = get(cache->get_var(ArealRadius{}));
      const auto& deriv_rbar_from_r =
          get(cache->get_var(DerivIsotropicRadiusFromAreal{}));
      const auto& conformal_factor =
          get(cache->get_var(Xcts::Tags::ConformalFactor<DataType>{}));
      const auto& lapse = get(cache->get_var(gr::Tags::Lapse<DataType>{}));
      const auto shift_magnitude = 2. * lapse / r / square(conformal_factor);
      const auto shift_magnitude_dr =
          shift_magnitude / r *
          ((square(lapse) + 2. * lapse / (1. + lapse)) / r - 1.);
      const DataType isotropic_prefactor =
          (shift_magnitude_dr / deriv_rbar_from_r - shift_magnitude / rbar) /
          square(rbar);
      const DataType diagonal_prefactor = shift_magnitude / rbar;
      for (size_t i = 0; i < 3; i++) {
        for (size_t j = i; j < 3; j++) {
          shift_strain->get(i, j) = isotropic_prefactor * x.get(i) * x.get(j);
        }
        shift_strain->get(i, i) += diagonal_prefactor;
      }
      break;
    }
    default:
      ERROR("Missing case for SchwarzschildCoordinates");
  }
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<tnsr::II<DataType, 3>*> longitudinal_shift_excess,
    const gsl::not_null<Cache*> cache,
    Tags::LongitudinalShiftExcess<DataType, 3, Frame::Inertial> /*meta*/)
    const noexcept {
  const auto& shift_strain =
      cache->get_var(Tags::ShiftStrain<DataType, 3, Frame::Inertial>{});
  Xcts::longitudinal_operator_flat_cartesian(longitudinal_shift_excess,
                                             shift_strain);
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*>
        longitudinal_shift_minus_dt_conformal_metric_square,
    const gsl::not_null<Cache*> cache,
    Tags::LongitudinalShiftMinusDtConformalMetricSquare<DataType> /*meta*/)
    const noexcept {
  const auto& longitudinal_shift = cache->get_var(
      Tags::LongitudinalShiftExcess<DataType, 3, Frame::Inertial>{});
  Xcts::detail::fully_contract(
      longitudinal_shift_minus_dt_conformal_metric_square, longitudinal_shift,
      longitudinal_shift);
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*>
        longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
    const gsl::not_null<Cache*> cache,
    Tags::LongitudinalShiftMinusDtConformalMetricOverLapseSquare<
        DataType> /*meta*/) const noexcept {
  *longitudinal_shift_minus_dt_conformal_metric_over_lapse_square =
      cache->get_var(
          Tags::LongitudinalShiftMinusDtConformalMetricSquare<DataType>{});
  const auto& lapse = cache->get_var(gr::Tags::Lapse<DataType>{});
  get(*longitudinal_shift_minus_dt_conformal_metric_over_lapse_square) /=
      square(get(lapse));
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
