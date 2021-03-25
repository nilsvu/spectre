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
#include "NumericalAlgorithms/RootFinding/NewtonRaphson.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/CommonVariables.tpp"
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
    IntermediateTags::IsotropicRadius<DataType> /*meta*/) const noexcept {
  magnitude(isotropic_radius, x);
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> areal_radius,
    const gsl::not_null<Cache*> cache,
    IntermediateTags::ArealRadius<DataType> /*meta*/) const noexcept {
  const auto& isotropic_radius =
      cache->get_var(IntermediateTags::IsotropicRadius<DataType>{});
  get(*areal_radius) =
      kerr_schild_areal_radius_from_isotropic(get(isotropic_radius));
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> deriv_isotropic_radius_from_areal,
    const gsl::not_null<Cache*> cache,
    IntermediateTags::DerivIsotropicRadiusFromAreal<DataType> /*meta*/)
    const noexcept {
  const auto& areal_radius =
      cache->get_var(IntermediateTags::ArealRadius<DataType>{});
  ASSERT(
      coordinate_system == SchwarzschildCoordinates::KerrSchildIsotropic,
      "Only compute the areal radius for 'KerrSchildIsotropic' coordinates.");
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
    const gsl::not_null<Scalar<DataType>*> trace_extrinsic_curvature,
    const gsl::not_null<Cache*> cache,
    gr::Tags::TraceExtrinsicCurvature<DataType> /*meta*/) const noexcept {
  switch (coordinate_system) {
    case SchwarzschildCoordinates::Isotropic: {
      get(*trace_extrinsic_curvature) = 0.;
      break;
    }
    case SchwarzschildCoordinates::PainleveGullstrand: {
      const auto& r =
          get(cache->get_var(IntermediateTags::IsotropicRadius<DataType>{}));
      get(*trace_extrinsic_curvature) = 1.5 * sqrt(2.) / pow(r, 1.5);
      break;
    }
    case SchwarzschildCoordinates::KerrSchildIsotropic: {
      const auto& r =
          get(cache->get_var(IntermediateTags::ArealRadius<DataType>{}));
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
    const gsl::not_null<tnsr::i<DataType, 3>*>
        trace_extrinsic_curvature_gradient,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataType>, tmpl::size_t<3>,
                  Frame::Inertial> /*meta*/) const noexcept {
  switch (coordinate_system) {
    case SchwarzschildCoordinates::Isotropic: {
      get(*trace_extrinsic_curvature_gradient) = 0.;
      break;
    }
    case SchwarzschildCoordinates::PainleveGullstrand: {
      const auto& r =
          get(cache->get_var(IntermediateTags::IsotropicRadius<DataType>{}));
      const auto isotropic_prefactor = -2.25 * sqrt(2.) / pow(r, 3.5);
      get<0>(*trace_extrinsic_curvature_gradient) =
          isotropic_prefactor * get<0>(x);
      get<1>(*trace_extrinsic_curvature_gradient) =
          isotropic_prefactor * get<1>(x);
      get<2>(*trace_extrinsic_curvature_gradient) =
          isotropic_prefactor * get<2>(x);
      break;
    }
    case SchwarzschildCoordinates::KerrSchildIsotropic: {
      // TODO
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
      const auto& r =
          get(cache->get_var(IntermediateTags::IsotropicRadius<DataType>{}));
      get(*conformal_factor) = 1. + 0.5 * mass / r;
      break;
    }
    case SchwarzschildCoordinates::PainleveGullstrand: {
      get(*conformal_factor) = 1.;
      break;
    }
    case SchwarzschildCoordinates::KerrSchildIsotropic: {
      const auto& r =
          get(cache->get_var(IntermediateTags::ArealRadius<DataType>{}));
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
    const gsl::not_null<tnsr::i<DataType, 3>*> deriv_conformal_factor,
    const gsl::not_null<Cache*> cache,
    ::Tags::deriv<Tags::ConformalFactor<DataType>, tmpl::size_t<3>,
                  Frame::Inertial> /*meta*/) const noexcept {
  switch (coordinate_system) {
    case SchwarzschildCoordinates::Isotropic: {
      const auto& r =
          get(cache->get_var(IntermediateTags::IsotropicRadius<DataType>{}));
      const DataType isotropic_prefactor = -0.5 * mass / cube(r);
      get<0>(*deriv_conformal_factor) = isotropic_prefactor * get<0>(x);
      get<1>(*deriv_conformal_factor) = isotropic_prefactor * get<1>(x);
      get<2>(*deriv_conformal_factor) = isotropic_prefactor * get<2>(x);
      break;
    }
    case SchwarzschildCoordinates::PainleveGullstrand: {
      std::fill(deriv_conformal_factor->begin(), deriv_conformal_factor->end(),
                0.);
      break;
    }
    case SchwarzschildCoordinates::KerrSchildIsotropic: {
      const auto& rbar =
          get(cache->get_var(IntermediateTags::IsotropicRadius<DataType>{}));
      const auto& r =
          get(cache->get_var(IntermediateTags::ArealRadius<DataType>{}));
      const auto& deriv_rbar_from_r = get(cache->get_var(
          IntermediateTags::DerivIsotropicRadiusFromAreal<DataType>{}));
      const auto one_over_lapse = sqrt(1. + 2. / r);
      const auto conformal_factor_dr = -2. * exp(one_over_lapse - 1.) /
                                       square(1. + one_over_lapse) / square(r);
      const DataType isotropic_prefactor =
          conformal_factor_dr / deriv_rbar_from_r / rbar;
      get<0>(*deriv_conformal_factor) = isotropic_prefactor * get<0>(x);
      get<1>(*deriv_conformal_factor) = isotropic_prefactor * get<1>(x);
      get<2>(*deriv_conformal_factor) = isotropic_prefactor * get<2>(x);
      break;
    }
    default:
      ERROR("Missing case for SchwarzschildCoordinates");
  }
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> lapse,
    const gsl::not_null<Cache*> cache,
    gr::Tags::Lapse<DataType> /*meta*/) const noexcept {
  switch (coordinate_system) {
    case SchwarzschildCoordinates::Isotropic: {
      const auto& r =
          get(cache->get_var(IntermediateTags::IsotropicRadius<DataType>{}));
      get(*lapse) = (1. - 0.5 * mass / r) / (1. + 0.5 * mass / r);
      break;
    }
    case SchwarzschildCoordinates::PainleveGullstrand: {
      get(*lapse) = 1.;
      break;
    }
    case SchwarzschildCoordinates::KerrSchildIsotropic: {
      const auto& r =
          get(cache->get_var(IntermediateTags::ArealRadius<DataType>{}));
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
      const auto& r =
          get(cache->get_var(IntermediateTags::IsotropicRadius<DataType>{}));
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
    const gsl::not_null<Scalar<DataType>*> lapse,
    const gsl::not_null<Cache*> cache,
    gr::Tags::Lapse<DataType> /*meta*/) const noexcept {
  *lapse = cache->get_var(Tags::LapseTimesConformalFactor<DataType>{});
  const auto& conformal_factor =
      cache->get_var(Tags::ConformalFactor<DataType>{});
  get(*lapse) /= get(conformal_factor);
}

template <typename DataType>
void SchwarzschildVariables<DataType>::operator()(
    const gsl::not_null<tnsr::i<DataType, 3>*>
        deriv_lapse_times_conformal_factor,
    const gsl::not_null<Cache*> cache,
    ::Tags::deriv<Tags::LapseTimesConformalFactor<DataType>, tmpl::size_t<3>,
                  Frame::Inertial> /*meta*/) const noexcept {
  switch (coordinate_system) {
    case SchwarzschildCoordinates::Isotropic: {
      *deriv_lapse_times_conformal_factor =
          cache->get_var(::Tags::deriv<Tags::ConformalFactor<DataType>,
                                       tmpl::size_t<3>, Frame::Inertial>{});
      get<0>(*deriv_lapse_times_conformal_factor) *= -1.;
      get<1>(*deriv_lapse_times_conformal_factor) *= -1.;
      get<2>(*deriv_lapse_times_conformal_factor) *= -1.;
      break;
    }
    case SchwarzschildCoordinates::PainleveGullstrand: {
      std::fill(deriv_lapse_times_conformal_factor->begin(),
                deriv_lapse_times_conformal_factor->end(), 0.);
      break;
    }
    case SchwarzschildCoordinates::KerrSchildIsotropic: {
      const auto& rbar =
          get(cache->get_var(IntermediateTags::IsotropicRadius<DataType>{}));
      const auto& r =
          get(cache->get_var(IntermediateTags::ArealRadius<DataType>{}));
      const auto& deriv_rbar_from_r = get(cache->get_var(
          IntermediateTags::DerivIsotropicRadiusFromAreal<DataType>{}));
      const auto& conformal_factor =
          get(cache->get_var(Xcts::Tags::ConformalFactor<DataType>{}));
      const auto& conformal_factor_gradient =
          cache->get_var(::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>,
                                       tmpl::size_t<3>, Frame::Inertial>{});
      const auto& lapse = get(cache->get_var(gr::Tags::Lapse<DataType>{}));
      const auto lapse_dr = cube(lapse) / square(r);
      const DataType isotropic_prefactor =
          conformal_factor * lapse_dr / deriv_rbar_from_r / rbar;
      *deriv_lapse_times_conformal_factor = conformal_factor_gradient;
      get<0>(*deriv_lapse_times_conformal_factor) *= lapse;
      get<1>(*deriv_lapse_times_conformal_factor) *= lapse;
      get<2>(*deriv_lapse_times_conformal_factor) *= lapse;
      get<0>(*deriv_lapse_times_conformal_factor) +=
          isotropic_prefactor * get<0>(x);
      get<1>(*deriv_lapse_times_conformal_factor) +=
          isotropic_prefactor * get<1>(x);
      get<2>(*deriv_lapse_times_conformal_factor) +=
          isotropic_prefactor * get<2>(x);
      break;
    }
    default:
      ERROR("Missing case for SchwarzschildCoordinates");
  }
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
    const gsl::not_null<tnsr::I<DataType, 3>*> shift_excess,
    const gsl::not_null<Cache*> cache,
    Tags::ShiftExcess<DataType, 3, Frame::Inertial> /*meta*/) const noexcept {
  switch (coordinate_system) {
    case SchwarzschildCoordinates::Isotropic: {
      std::fill(shift_excess->begin(), shift_excess->end(), 0.);
      break;
    }
    case SchwarzschildCoordinates::PainleveGullstrand: {
      const auto& r =
          get(cache->get_var(IntermediateTags::IsotropicRadius<DataType>{}));
      const DataType isotropic_prefactor = sqrt(2.) / pow(r, 1.5);
      *shift_excess = x;
      get<0>(*shift_excess) *= isotropic_prefactor;
      get<1>(*shift_excess) *= isotropic_prefactor;
      get<2>(*shift_excess) *= isotropic_prefactor;
      break;
    }
    case SchwarzschildCoordinates::KerrSchildIsotropic: {
      const auto& rbar =
          get(cache->get_var(IntermediateTags::IsotropicRadius<DataType>{}));
      const auto& r =
          get(cache->get_var(IntermediateTags::ArealRadius<DataType>{}));
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
    const gsl::not_null<tnsr::ii<DataType, 3>*> shift_strain,
    const gsl::not_null<Cache*> cache,
    Tags::ShiftStrain<DataType, 3, Frame::Inertial> /*meta*/) const noexcept {
  switch (coordinate_system) {
    case SchwarzschildCoordinates::Isotropic: {
      std::fill(shift_strain->begin(), shift_strain->end(), 0.);
      break;
    }
    case SchwarzschildCoordinates::PainleveGullstrand: {
      const auto& r =
          get(cache->get_var(IntermediateTags::IsotropicRadius<DataType>{}));
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
      const auto& rbar =
          get(cache->get_var(IntermediateTags::IsotropicRadius<DataType>{}));
      const auto& r =
          get(cache->get_var(IntermediateTags::ArealRadius<DataType>{}));
      const auto& deriv_rbar_from_r = get(cache->get_var(
          IntermediateTags::DerivIsotropicRadiusFromAreal<DataType>{}));
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

template class SchwarzschildVariables<double>;
template class SchwarzschildVariables<DataVector>;

}  // namespace Xcts::Solutions::detail

// Instantiate implementations for common variables
template class Xcts::Solutions::CommonVariables<
    double,
    typename Xcts::Solutions::detail::SchwarzschildVariables<double>::Cache>;
template class Xcts::Solutions::CommonVariables<
    DataVector, typename Xcts::Solutions::detail::SchwarzschildVariables<
                    DataVector>::Cache>;
template class Xcts::AnalyticData::CommonVariables<
    double,
    typename Xcts::Solutions::detail::SchwarzschildVariables<double>::Cache>;
template class Xcts::AnalyticData::CommonVariables<
    DataVector, typename Xcts::Solutions::detail::SchwarzschildVariables<
                    DataVector>::Cache>;

/// \endcond
