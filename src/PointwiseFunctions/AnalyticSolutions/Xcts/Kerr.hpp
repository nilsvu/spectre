// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <ostream>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/AnalyticSolution.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Xcts::Solutions {
namespace detail {

template <typename DataType>
struct KerrVariables {
  static constexpr size_t Dim = 3;
  using Cache = CachedTempBuffer<
      KerrVariables, Tags::ConformalMetric<DataType, 3, Frame::Inertial>,
      Tags::InverseConformalMetric<DataType, 3, Frame::Inertial>,
      ::Tags::deriv<Tags::ConformalMetric<DataType, 3, Frame::Inertial>,
                    tmpl::size_t<3>, Frame::Inertial>,
      Tags::ConformalChristoffelFirstKind<DataType, 3, Frame::Inertial>,
      Tags::ConformalChristoffelSecondKind<DataType, 3, Frame::Inertial>,
      Tags::ConformalChristoffelContracted<DataType, 3, Frame::Inertial>,
      gr::Tags::TraceExtrinsicCurvature<DataType>,
      ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>>,
      Tags::ConformalFactor<DataType>,
      ::Tags::deriv<Tags::ConformalFactor<DataType>, tmpl::size_t<3>,
                    Frame::Inertial>,
      ::Tags::Flux<Tags::ConformalFactor<DataType>, tmpl::size_t<3>,
                   Frame::Inertial>,
      gr::Tags::Lapse<DataType>, Tags::LapseTimesConformalFactor<DataType>,
      ::Tags::deriv<Tags::LapseTimesConformalFactor<DataType>, tmpl::size_t<3>,
                    Frame::Inertial>,
      ::Tags::Flux<Tags::LapseTimesConformalFactor<DataType>, tmpl::size_t<3>,
                   Frame::Inertial>,
      Tags::ShiftBackground<DataType, 3, Frame::Inertial>,
      Tags::ShiftExcess<DataType, 3, Frame::Inertial>,
      gr::Tags::Shift<Dim, Frame::Inertial, DataType>,
      Tags::ShiftStrain<DataType, 3, Frame::Inertial>,
      Tags::LongitudinalShiftExcess<DataType, 3, Frame::Inertial>,
      Tags::LongitudinalShiftMinusDtConformalMetricOverLapseSquare<DataType>,
      Tags::LongitudinalShiftMinusDtConformalMetricSquare<DataType>,
      Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<DataType, 3,
                                                              Frame::Inertial>,
      ::Tags::div<Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
          DataType, 3, Frame::Inertial>>,
      gr::Tags::EnergyDensity<DataType>, gr::Tags::StressTrace<DataType>,
      gr::Tags::MomentumDensity<3, Frame::Inertial, DataType>,
      ::Tags::FixedSource<Tags::ConformalFactor<DataType>>,
      ::Tags::FixedSource<Tags::LapseTimesConformalFactor<DataType>>,
      ::Tags::FixedSource<Tags::ShiftExcess<DataType, 3, Frame::Inertial>>>;

  const tnsr::I<DataType, 3>& x;
  const gr::Solutions::KerrSchild& kerr_schild;

  void operator()(gsl::not_null<tnsr::ii<DataType, 3>*> conformal_metric,
                  gsl::not_null<Cache*> cache,
                  Tags::ConformalMetric<DataType, 3, Frame::Inertial> /*meta*/)
      const noexcept;
  void operator()(
      gsl::not_null<tnsr::II<DataType, 3>*> inv_conformal_metric,
      gsl::not_null<Cache*> cache,
      Tags::InverseConformalMetric<DataType, 3, Frame::Inertial> /*meta*/)
      const noexcept;
  void operator()(
      gsl::not_null<tnsr::ijj<DataType, 3>*> deriv_conformal_metric,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<Tags::ConformalMetric<DataType, 3, Frame::Inertial>,
                    tmpl::size_t<3>, Frame::Inertial> /*meta*/) const noexcept;
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
  void operator()(
      gsl::not_null<Scalar<DataType>*> trace_extrinsic_curvature,
      gsl::not_null<Cache*> cache,
      gr::Tags::TraceExtrinsicCurvature<DataType> /*meta*/) const noexcept;
  void operator()(
      gsl::not_null<Scalar<DataType>*> dt_trace_extrinsic_curvature,
      gsl::not_null<Cache*> cache,
      ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>> /*meta*/)
      const noexcept;
  void operator()(gsl::not_null<Scalar<DataType>*> conformal_factor,
                  gsl::not_null<Cache*> cache,
                  Tags::ConformalFactor<DataType> /*meta*/) const noexcept;
  void operator()(
      gsl::not_null<tnsr::i<DataType, 3>*> conformal_factor_gradient,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>, tmpl::size_t<3>,
                    Frame::Inertial> /*meta*/) const noexcept;
  void operator()(gsl::not_null<tnsr::I<DataType, 3>*> conformal_factor_flux,
                  gsl::not_null<Cache*> cache,
                  ::Tags::Flux<Tags::ConformalFactor<DataType>, tmpl::size_t<3>,
                               Frame::Inertial> /*meta*/) const noexcept;
  void operator()(gsl::not_null<Scalar<DataType>*> lapse,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::Lapse<DataType> /*meta*/) const noexcept;
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
  void operator()(gsl::not_null<tnsr::I<DataType, 3>*> shift_background,
                  gsl::not_null<Cache*> cache,
                  Tags::ShiftBackground<DataType, 3, Frame::Inertial> /*meta*/)
      const noexcept;
  void operator()(
      gsl::not_null<tnsr::I<DataType, 3>*> shift_excess,
      gsl::not_null<Cache*> cache,
      Tags::ShiftExcess<DataType, 3, Frame::Inertial> /*meta*/) const noexcept;
  void operator()(
      gsl::not_null<tnsr::I<DataType, 3>*> shift,
      gsl::not_null<Cache*> cache,
      gr::Tags::Shift<3, Frame::Inertial, DataType> /*meta*/) const noexcept;
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

}  // namespace detail

// The following implements the registration and factory-creation mechanism

/// \cond
template <typename Registrars>
struct Kerr;

namespace Registrars {
struct Kerr {
  template <typename Registrars>
  using f = Solutions::Kerr<Registrars>;
};
}  // namespace Registrars
/// \endcond

template <typename Registrars = tmpl::list<Solutions::Registrars::Kerr>>
class Kerr : public AnalyticSolution<Registrars>,
             public gr::Solutions::KerrSchild {
 public:
  using KerrSchild::KerrSchild;
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Kerr);

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<RequestedTags...> /*meta*/) const noexcept {
    using VarsComputer = detail::KerrVariables<DataType>;
    typename VarsComputer::Cache cache{get_size(*x.begin()),
                                       VarsComputer{x, *this}};
    return {cache.get_var(RequestedTags{})...};
  }
};

/// \cond
template <typename Registrars>
PUP::able::PUP_ID Kerr<Registrars>::my_PUP_ID = 0;  // NOLINT
/// \endcond

}  // namespace Xcts::Solutions
