// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/Xcts/Kerr.hpp"

#include <algorithm>
#include <ostream>
#include <pup.h>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Equations.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Xcts/LongitudinalOperator.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
namespace Xcts::AnalyticData {

template <size_t Dim, typename DataType, typename Cache>
void CommonVariables<DataType, Dim, Cache>::operator()(
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

template <size_t Dim, typename DataType, typename Cache>
void CommonVariables<DataType, Dim, Cache>::operator()(
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

template <size_t Dim, typename DataType, typename Cache>
void CommonVariables<DataType, Dim, Cache>::operator()(
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

template <size_t Dim, typename DataType, typename Cache>
void CommonVariables<DataType, Dim, Cache>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> conformal_factor_flux,
    const gsl::not_null<Cache*> cache,
    ::Tags::Flux<Tags::ConformalFactor<DataType>, tmpl::size_t<3>,
                 Frame::Inertial> /*meta*/) const noexcept {
  const auto& conformal_factor_gradient =
      cache->get_var(::Tags::deriv<Tags::ConformalFactor<DataType>,
                                   tmpl::size_t<3>, Frame::Inertial>{});
  const auto& inv_conformal_metric = cache->get_var(
      Tags::InverseConformalMetric<DataType, 3, Frame::Inertial>{});
  raise_or_lower_index(conformal_factor_flux, conformal_factor_gradient,
                       inv_conformal_metric);
}

template <size_t Dim, typename DataType, typename Cache>
void CommonVariables<DataType, Dim, Cache>::operator()(
    const gsl::not_null<Scalar<DataType>*> lapse_times_conformal_factor,
    const gsl::not_null<Cache*> cache,
    Tags::LapseTimesConformalFactor<DataType> /*meta*/) const noexcept {
  *lapse_times_conformal_factor = cache->get_var(gr::Tags::Lapse<DataType>{});
  const auto& conformal_factor =
      cache->get_var(Xcts::Tags::ConformalFactor<DataType>{});
  get(*lapse_times_conformal_factor) *= get(conformal_factor);
}

template <size_t Dim, typename DataType, typename Cache>
void CommonVariables<DataType, Dim, Cache>::operator()(
    const gsl::not_null<tnsr::i<DataType, 3>*>
        deriv_lapse_times_conformal_factor,
    const gsl::not_null<Cache*> cache,
    ::Tags::deriv<Tags::LapseTimesConformalFactor<DataType>, tmpl::size_t<3>,
                  Frame::Inertial> /*meta*/) const noexcept {
  const auto& conformal_factor =
      cache->get_var(Xcts::Tags::ConformalFactor<DataType>{});
  const auto& deriv_conformal_factor =
      cache->get_var(::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>,
                                   tmpl::size_t<3>, Frame::Inertial>{});
  const auto& lapse = cache->get_var(gr::Tags::Lapse<DataType>{});
  const auto& deriv_lapse =
      cache->get_var(::Tags::deriv<gr::Tags::Lapse<DataType>, tmpl::size_t<3>,
                                   Frame::Inertial>{});
  for (size_t i = 0; i < 3; ++i) {
    deriv_lapse_times_conformal_factor->get(i) =
        deriv_lapse.get(i) * get(conformal_factor) +
        get(lapse) * deriv_conformal_factor.get(i);
  }
}

template <size_t Dim, typename DataType, typename Cache>
void CommonVariables<DataType, Dim, Cache>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*>
        lapse_times_conformal_factor_flux,
    const gsl::not_null<Cache*> cache,
    ::Tags::Flux<Tags::LapseTimesConformalFactor<DataType>, tmpl::size_t<3>,
                 Frame::Inertial> /*meta*/) const noexcept {
  const auto& deriv_lapse_times_conformal_factor =
      cache->get_var(::Tags::deriv<Tags::LapseTimesConformalFactor<DataType>,
                                   tmpl::size_t<3>, Frame::Inertial>{});
  const auto& inv_conformal_metric = cache->get_var(
      Tags::InverseConformalMetric<DataType, 3, Frame::Inertial>{});
  raise_or_lower_index(lapse_times_conformal_factor_flux,
                       deriv_lapse_times_conformal_factor,
                       inv_conformal_metric);
}

template <size_t Dim, typename DataType, typename Cache>
void CommonVariables<DataType, Dim, Cache>::operator()(
    const gsl::not_null<tnsr::I<DataType, Dim>*> shift,
    const gsl::not_null<Cache*> cache,
    gr::Tags::Shift<Dim, Frame::Inertial, DataType> /*meta*/) const noexcept {
  const auto& shift_background =
      cache->get_var(Tags::ShiftBackground<DataType, 3, Frame::Inertial>{});
  const auto& shift_excess =
      cache->get_var(Tags::ShiftExcess<DataType, 3, Frame::Inertial>{});
  for (size_t i = 0; i < Dim; ++i) {
    shift->get(i) = shift_background.get(i) + shift_excess.get(i);
  }
}

template <size_t Dim, typename DataType, typename Cache>
void CommonVariables<DataType, Dim, Cache>::operator()(
    const gsl::not_null<Scalar<DataType>*>
        shift_dot_deriv_extrinsic_curvature_trace,
    const gsl::not_null<Cache*> cache,
    Tags::ShiftDotDerivExtrinsicCurvatureTrace<DataType> /*meta*/)
    const noexcept {
  const auto& shift =
      cache->get_var(gr::Tags::Shift<3, Frame::Inertial, DataType>{});
  const auto& deriv_extrinsic_curvature_trace =
      cache->get_var(::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataType>,
                                   tmpl::size_t<3>, Frame::Inertial>{});
  dot_product(shift_dot_deriv_extrinsic_curvature_trace, shift,
              deriv_extrinsic_curvature_trace);
}

template <size_t Dim, typename DataType, typename Cache>
void CommonVariables<DataType, Dim, Cache>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> shift_strain,
    const gsl::not_null<Cache*> cache,
    Tags::ShiftStrain<DataType, 3, Frame::Inertial> /*meta*/) const noexcept {
  const auto& shift_excess =
      cache->get_var(Tags::ShiftExcess<DataType, 3, Frame::Inertial>{});
  const auto& deriv_shift_excess = cache->get_var(
      ::Tags::deriv<Tags::ShiftExcess<DataType, 3, Frame::Inertial>,
                    tmpl::size_t<3>, Frame::Inertial>{});
  const auto& conformal_metric = cache->get_var(
      Xcts::Tags::ConformalMetric<DataType, 3, Frame::Inertial>{});
  const auto& deriv_conformal_metric = cache->get_var(
      ::Tags::deriv<Xcts::Tags::ConformalMetric<DataType, 3, Frame::Inertial>,
                    tmpl::size_t<3>, Frame::Inertial>{});
  const auto& conformal_christoffel_first_kind = cache->get_var(
      Tags::ConformalChristoffelFirstKind<DataType, 3, Frame::Inertial>{});
  auto deriv_shift_lowered = make_with_value<tnsr::ij<DataType, 3>>(x, 0.);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        deriv_shift_lowered.get(i, j) +=
            conformal_metric.get(j, k) * deriv_shift.get(i, k) +
            shift.get(k) * deriv_conformal_metric.get(i, j, k);
      }
    }
  }
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      shift_strain->get(i, j) =
          0.5 * (deriv_shift_lowered.get(i, j) + deriv_shift_lowered.get(j, i));
      for (size_t k = 0; k < 3; ++k) {
        shift_strain->get(i, j) -=
            conformal_christoffel_first_kind.get(k, i, j) * shift.get(k);
      }
    }
  }
}

template <size_t Dim, typename DataType, typename Cache>
void CommonVariables<DataType, Dim, Cache>::operator()(
    const gsl::not_null<tnsr::II<DataType, 3>*> longitudinal_shift_excess,
    const gsl::not_null<Cache*> cache,
    Tags::LongitudinalShiftExcess<DataType, 3, Frame::Inertial> /*meta*/)
    const noexcept {
  const auto& shift_strain =
      cache->get_var(Tags::ShiftStrain<DataType, 3, Frame::Inertial>{});
  const auto& inv_conformal_metric = cache->get_var(
      Tags::InverseConformalMetric<DataType, 3, Frame::Inertial>{});
  Xcts::longitudinal_operator(longitudinal_shift_excess, shift_strain,
                              inv_conformal_metric);
}

template <size_t Dim, typename DataType, typename Cache>
void CommonVariables<DataType, Dim, Cache>::operator()(
    const gsl::not_null<Scalar<DataType>*>
        longitudinal_shift_minus_dt_conformal_metric_square,
    const gsl::not_null<Cache*> cache,
    Tags::LongitudinalShiftMinusDtConformalMetricSquare<DataType> /*meta*/)
    const noexcept {
  auto longitudinal_shift = cache->get_var(
      Tags::LongitudinalShiftExcess<DataType, 3, Frame::Inertial>{});
  longitudinal_shift +=
      cache->get_var(Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
                     DataType, 3, Frame::Inertial>{});
  const auto& conformal_metric = cache->get_var(
      Tags::InverseConformalMetric<DataType, 3, Frame::Inertial>{});
  Xcts::detail::fully_contract(
      longitudinal_shift_minus_dt_conformal_metric_square, longitudinal_shift,
      longitudinal_shift, conformal_metric);
}

template <size_t Dim, typename DataType, typename Cache>
void CommonVariables<DataType, Dim, Cache>::operator()(
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

template <size_t Dim, typename DataType, typename Cache>
void CommonVariables<DataType, Dim, Cache>::operator()(
    const gsl::not_null<Scalar<DataType>*>
        fixed_source_for_hamiltonian_constraint,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::FixedSource<Tags::ConformalFactor<DataType>> /*meta*/)
    const noexcept {
  std::fill(fixed_source_for_hamiltonian_constraint->begin(),
            fixed_source_for_hamiltonian_constraint->end(), 0.);
}

template <size_t Dim, typename DataType, typename Cache>
void CommonVariables<DataType, Dim, Cache>::operator()(
    const gsl::not_null<Scalar<DataType>*> fixed_source_for_lapse_equation,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::FixedSource<Tags::LapseTimesConformalFactor<DataType>> /*meta*/)
    const noexcept {
  std::fill(fixed_source_for_lapse_equation->begin(),
            fixed_source_for_lapse_equation->end(), 0.);
}

template <size_t Dim, typename DataType, typename Cache>
void CommonVariables<DataType, Dim, Cache>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> fixed_source_momentum_constraint,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::FixedSource<
        Tags::ShiftExcess<DataType, 3, Frame::Inertial>> /*meta*/)
    const noexcept {
  std::fill(fixed_source_momentum_constraint->begin(),
            fixed_source_momentum_constraint->end(), 0.);
}

template class KerrVariables<double>;
template class KerrVariables<DataVector>;

}  // namespace Xcts::AnalyticData

/// \endcond
