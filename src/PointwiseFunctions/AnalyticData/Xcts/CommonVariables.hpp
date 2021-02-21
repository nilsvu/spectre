// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

namespace Xcts::AnalyticData {

template <typename DataType, size_t Dim, typename Cache>
struct CommonVariables {
  void operator()(
      gsl::not_null<tnsr::ijj<DataType, Dim>*> conformal_christoffel_first_kind,
      gsl::not_null<Cache*> cache,
      Tags::ConformalChristoffelFirstKind<
          DataType, Dim, Frame::Inertial> /*meta*/) const noexcept;
  void operator()(gsl::not_null<tnsr::Ijj<DataType, Dim>*>
                      conformal_christoffel_second_kind,
                  gsl::not_null<Cache*> cache,
                  Tags::ConformalChristoffelSecondKind<
                      DataType, Dim, Frame::Inertial> /*meta*/) const noexcept;
  void operator()(
      gsl::not_null<tnsr::i<DataType, Dim>*> conformal_christoffel_contracted,
      gsl::not_null<Cache*> cache,
      Tags::ConformalChristoffelContracted<
          DataType, Dim, Frame::Inertial> /*meta*/) const noexcept;
  void operator()(gsl::not_null<tnsr::I<DataType, 3>*> conformal_factor_flux,
                  gsl::not_null<Cache*> cache,
                  ::Tags::Flux<Tags::ConformalFactor<DataType>, tmpl::size_t<3>,
                               Frame::Inertial> /*meta*/) const noexcept;
  void operator()(
      gsl::not_null<Scalar<DataType>*> lapse_times_conformal_factor,
      gsl::not_null<Cache*> cache,
      Tags::LapseTimesConformalFactor<DataType> /*meta*/) const noexcept;
  void operator()(
      gsl::not_null<tnsr::i<DataType, 3>*>
          lapse_times_conformal_factor_gradient,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<Tags::LapseTimesConformalFactor<DataType>, tmpl::size_t<3>,
                    Frame::Inertial> /*meta*/) const noexcept;
  void operator()(
      gsl::not_null<tnsr::I<DataType, 3>*> lapse_times_conformal_factor_flux,
      gsl::not_null<Cache*> cache,
      ::Tags::Flux<Tags::LapseTimesConformalFactor<DataType>, tmpl::size_t<3>,
                   Frame::Inertial> /*meta*/) const noexcept;
  void operator()(gsl::not_null<Scalar<DataType>*>
                      shift_dot_deriv_extrinsic_curvature_trace,
                  gsl::not_null<Cache*> cache,
                  Tags::ShiftDotDerivExtrinsicCurvatureTrace<DataType> /*meta*/)
      const noexcept;
  void operator()(
      gsl::not_null<tnsr::ii<DataType, 3>*> shift_strain,
      gsl::not_null<Cache*> cache,
      Tags::ShiftStrain<DataType, 3, Frame::Inertial> /*meta*/) const noexcept;
  void operator()(
      gsl::not_null<tnsr::II<DataType, 3>*> longitudinal_shift_excess,
      gsl::not_null<Cache*> cache,
      Tags::LongitudinalShiftExcess<DataType, 3, Frame::Inertial> /*meta*/)
      const noexcept;
  void operator()(
      gsl::not_null<Scalar<DataType>*>
          longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
      gsl::not_null<Cache*> cache,
      Tags::LongitudinalShiftMinusDtConformalMetricOverLapseSquare<
          DataType> /*meta*/) const noexcept;
  void operator()(
      gsl::not_null<Scalar<DataType>*>
          longitudinal_shift_minus_dt_conformal_metric_square,
      gsl::not_null<Cache*> cache,
      Tags::LongitudinalShiftMinusDtConformalMetricSquare<DataType> /*meta*/)
      const noexcept;
  void operator()(gsl::not_null<tnsr::II<DataType, 3, Frame::Inertial>*>
                      longitudinal_shift_background_minus_dt_conformal_metric,
                  gsl::not_null<Cache*> cache,
                  Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
                      DataType, 3, Frame::Inertial> /*meta*/) const noexcept;
  void operator()(
      gsl::not_null<tnsr::I<DataType, 3, Frame::Inertial>*>
          div_longitudinal_shift_background_minus_dt_conformal_metric,
      gsl::not_null<Cache*> cache,
      ::Tags::div<Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
          DataType, 3, Frame::Inertial>> /*meta*/) const noexcept;
  void operator()(gsl::not_null<Scalar<DataType>*> energy_density,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::EnergyDensity<DataType> /*meta*/) const noexcept;
  void operator()(gsl::not_null<Scalar<DataType>*> stress_trace,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::StressTrace<DataType> /*meta*/) const noexcept;
  void operator()(gsl::not_null<tnsr::I<DataType, 3>*> momentum_density,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::MomentumDensity<3, Frame::Inertial,
                                            DataType> /*meta*/) const noexcept;
  void operator()(
      gsl::not_null<Scalar<DataType>*> fixed_source_for_hamiltonian_constraint,
      gsl::not_null<Cache*> cache,
      ::Tags::FixedSource<Tags::ConformalFactor<DataType>> /*meta*/)
      const noexcept;
  void operator()(
      gsl::not_null<Scalar<DataType>*> fixed_source_for_lapse_equation,
      gsl::not_null<Cache*> cache,
      ::Tags::FixedSource<Tags::LapseTimesConformalFactor<DataType>> /*meta*/)
      const noexcept;
  void operator()(
      gsl::not_null<tnsr::I<DataType, 3>*> fixed_source_momentum_constraint,
      gsl::not_null<Cache*> cache,
      ::Tags::FixedSource<
          Tags::ShiftExcess<DataType, 3, Frame::Inertial>> /*meta*/)
      const noexcept;
};

}  // namespace Xcts::AnalyticData
