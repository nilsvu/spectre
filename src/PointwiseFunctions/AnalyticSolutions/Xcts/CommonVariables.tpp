// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "PointwiseFunctions/AnalyticSolutions/Xcts/CommonVariables.hpp"

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/CommonVariables.tpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Xcts/LongitudinalOperator.hpp"
#include "Utilities/Gsl.hpp"

namespace Xcts::Solutions {

template <typename DataType, typename Cache>
void CommonVariables<DataType, Cache>::operator()(
    const gsl::not_null<tnsr::I<DataType, Dim>*> conformal_factor_flux,
    const gsl::not_null<Cache*> cache,
    ::Tags::Flux<Tags::ConformalFactor<DataType>, tmpl::size_t<Dim>,
                 Frame::Inertial> /*meta*/) const {
  const auto& conformal_factor_gradient = cache->get_var(
      *this, ::Tags::deriv<Tags::ConformalFactor<DataType>, tmpl::size_t<Dim>,
                           Frame::Inertial>{});
  const auto& inv_conformal_metric = cache->get_var(
      *this, Tags::InverseConformalMetric<DataType, Dim, Frame::Inertial>{});
  raise_or_lower_index(conformal_factor_flux, conformal_factor_gradient,
                       inv_conformal_metric);
}

template <typename DataType, typename Cache>
void CommonVariables<DataType, Cache>::operator()(
    const gsl::not_null<tnsr::I<DataType, Dim>*>
        lapse_times_conformal_factor_flux,
    const gsl::not_null<Cache*> cache,
    ::Tags::Flux<Tags::LapseTimesConformalFactor<DataType>, tmpl::size_t<Dim>,
                 Frame::Inertial> /*meta*/) const {
  const auto& lapse_times_conformal_factor_gradient = cache->get_var(
      *this, ::Tags::deriv<Tags::LapseTimesConformalFactor<DataType>,
                           tmpl::size_t<Dim>, Frame::Inertial>{});
  const auto& inv_conformal_metric = cache->get_var(
      *this, Tags::InverseConformalMetric<DataType, Dim, Frame::Inertial>{});
  raise_or_lower_index(lapse_times_conformal_factor_flux,
                       lapse_times_conformal_factor_gradient,
                       inv_conformal_metric);
}

template <typename DataType, typename Cache>
void CommonVariables<DataType, Cache>::operator()(
    const gsl::not_null<tnsr::II<DataType, Dim>*> longitudinal_shift_excess,
    const gsl::not_null<Cache*> cache,
    Tags::LongitudinalShiftExcess<DataType, Dim, Frame::Inertial> /*meta*/)
    const {
  const auto& shift_strain = cache->get_var(
      *this, Tags::ShiftStrain<DataType, Dim, Frame::Inertial>{});
  const auto& inv_conformal_metric = cache->get_var(
      *this, Tags::InverseConformalMetric<DataType, Dim, Frame::Inertial>{});
  Xcts::longitudinal_operator(longitudinal_shift_excess, shift_strain,
                              inv_conformal_metric);
}

template <typename DataType, typename Cache>
void CommonVariables<DataType, Cache>::operator()(
    const gsl::not_null<tnsr::I<DataType, Dim>*> shift,
    const gsl::not_null<Cache*> cache,
    gr::Tags::Shift<Dim, Frame::Inertial, DataType> /*meta*/) const {
  *shift = cache->get_var(*this,
                          Tags::ShiftExcess<DataType, Dim, Frame::Inertial>{});
  const auto& shift_background = cache->get_var(
      *this, Tags::ShiftBackground<DataType, Dim, Frame::Inertial>{});
  for (size_t d = 0; d < Dim; ++d) {
    shift->get(d) += shift_background.get(d);
  }
}

template <typename DataType, typename Cache>
void CommonVariables<DataType, Cache>::operator()(
    const gsl::not_null<tnsr::ii<DataType, Dim>*> spatial_metric,
    const gsl::not_null<Cache*> cache,
    gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataType> /*meta*/) const {
  *spatial_metric = cache->get_var(
      *this, Tags::ConformalMetric<DataType, Dim, Frame::Inertial>{});
  const auto& conformal_factor =
      cache->get_var(*this, Tags::ConformalFactor<DataType>{});
  for (size_t i = 0; i < spatial_metric->size(); ++i) {
    (*spatial_metric)[i] *= pow<4>(get(conformal_factor));
  }
}

template <typename DataType, typename Cache>
void CommonVariables<DataType, Cache>::operator()(
    const gsl::not_null<tnsr::II<DataType, Dim>*> inv_spatial_metric,
    const gsl::not_null<Cache*> cache,
    gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataType> /*meta*/)
    const {
  const auto& conformal_factor =
      cache->get_var(*this, Tags::ConformalFactor<DataType>{});
  *inv_spatial_metric = cache->get_var(
      *this, Tags::InverseConformalMetric<DataType, Dim, Frame::Inertial>{});
  for (size_t i = 0; i < inv_spatial_metric->size(); ++i) {
    (*inv_spatial_metric)[i] /= pow<4>(get(conformal_factor));
  }
}

template <typename DataType, typename Cache>
void CommonVariables<DataType, Cache>::operator()(
    const gsl::not_null<tnsr::ijj<DataType, Dim>*> deriv_spatial_metric,
    const gsl::not_null<Cache*> cache,
    ::Tags::deriv<gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataType>,
                  tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const {
  const auto& conformal_metric = cache->get_var(
      *this, Tags::ConformalMetric<DataType, Dim, Frame::Inertial>{});
  const auto& conformal_factor =
      cache->get_var(*this, Tags::ConformalFactor<DataType>{});
  const auto& deriv_conformal_factor = cache->get_var(
      *this, ::Tags::deriv<Tags::ConformalFactor<DataType>, tmpl::size_t<Dim>,
                           Frame::Inertial>{});
  *deriv_spatial_metric = cache->get_var(
      *this,
      ::Tags::deriv<Tags::ConformalMetric<DataType, Dim, Frame::Inertial>,
                    tmpl::size_t<3>, Frame::Inertial>{});
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k <= j; ++k) {
        deriv_spatial_metric->get(i, j, k) *= pow<4>(get(conformal_factor));
        deriv_spatial_metric->get(i, j, k) +=
            4. * pow<3>(get(conformal_factor)) * deriv_conformal_factor.get(i) *
            conformal_metric.get(j, k);
      }
    }
  }
}

template <typename DataType, typename Cache>
void CommonVariables<DataType, Cache>::operator()(
    const gsl::not_null<Scalar<DataType>*>
        longitudinal_shift_minus_dt_conformal_metric_square,
    const gsl::not_null<Cache*> cache,
    Tags::LongitudinalShiftMinusDtConformalMetricSquare<DataType> /*meta*/)
    const {
  const auto& longitudinal_shift_background = cache->get_var(
      *this, Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
                 DataType, Dim, Frame::Inertial>{});
  const auto& longitudinal_shift_excess = cache->get_var(
      *this, Tags::LongitudinalShiftExcess<DataType, Dim, Frame::Inertial>{});
  const auto& conformal_metric = cache->get_var(
      *this, Tags::ConformalMetric<DataType, Dim, Frame::Inertial>{});
  get(*longitudinal_shift_minus_dt_conformal_metric_square) = 0.;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        for (size_t l = 0; l < 3; ++l) {
          get(*longitudinal_shift_minus_dt_conformal_metric_square) +=
              conformal_metric.get(i, k) * conformal_metric.get(j, l) *
              (longitudinal_shift_background.get(i, j) +
               longitudinal_shift_excess.get(i, j)) *
              (longitudinal_shift_background.get(k, l) +
               longitudinal_shift_excess.get(k, l));
        }
      }
    }
  }
}

template <typename DataType, typename Cache>
void CommonVariables<DataType, Cache>::operator()(
    const gsl::not_null<Scalar<DataType>*>
        longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
    const gsl::not_null<Cache*> cache,
    Tags::LongitudinalShiftMinusDtConformalMetricOverLapseSquare<
        DataType> /*meta*/) const {
  *longitudinal_shift_minus_dt_conformal_metric_over_lapse_square =
      cache->get_var(
          *this,
          Tags::LongitudinalShiftMinusDtConformalMetricSquare<DataType>{});
  const auto& lapse = cache->get_var(*this, gr::Tags::Lapse<DataType>{});
  get(*longitudinal_shift_minus_dt_conformal_metric_over_lapse_square) /=
      square(get(lapse));
}

template <typename DataType, typename Cache>
void CommonVariables<DataType, Cache>::operator()(
    const gsl::not_null<Scalar<DataType>*>
        shift_dot_deriv_extrinsic_curvature_trace,
    const gsl::not_null<Cache*> cache,
    Tags::ShiftDotDerivExtrinsicCurvatureTrace<DataType> /*meta*/) const {
  const auto& shift =
      cache->get_var(*this, gr::Tags::Shift<Dim, Frame::Inertial, DataType>{});
  const auto& deriv_extrinsic_curvature_trace = cache->get_var(
      *this, ::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataType>,
                           tmpl::size_t<Dim>, Frame::Inertial>{});
  dot_product(shift_dot_deriv_extrinsic_curvature_trace, shift,
              deriv_extrinsic_curvature_trace);
}

}  // namespace Xcts::Solutions
