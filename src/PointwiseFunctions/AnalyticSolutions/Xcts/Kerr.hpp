// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <ostream>

#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace Xcts::Solutions {

/// Various coordinate systems in which to express the Kerr solution
enum class KerrCoordinates { KerrSchild, Harmonic };

std::ostream& operator<<(std::ostream& os,
                         const KerrCoordinates& coords) noexcept;

/*!
 * \brief Kerr spacetime in general relativity
 *
 * This class implements the Kerr solution in various coordinate systems. See
 * the entries of the `Xcts::Solutions::KerrCoordinates` enum for the available
 * coordinate systems and for the solution variables in the respective
 * coordinates.
 */
template <KerrCoordinates Coords>
class Kerr {
 public:
  using options = typename gr::Solutions::KerrSchild::options;
  static constexpr Options::String help{"Kerr spacetime in general relativity"};

  Kerr() = default;
  Kerr(const Kerr&) noexcept = default;
  Kerr& operator=(const Kerr&) noexcept = default;
  Kerr(Kerr&&) noexcept = default;
  Kerr& operator=(Kerr&&) noexcept = default;
  ~Kerr() noexcept = default;

  Kerr(double mass, std::array<double, 3> dimensionless_spin,
       std::array<double, 3> center, const Options::Context& context = {});

  /// The radius of the event horizon in the given coordinates
  static double radius_at_event_horizon() noexcept;

  /// The radius of the Cauchy horizon in the given coordinates
  static double radius_at_cauchy_horizon() noexcept;

  // @{
  /// Retrieve variable at coordinates `x`

  // Missing quantities for the XCTS system (need numeric derivatives):
  // - Deriv(ExtrinsicCurvatureTrace)
  // - Conformal Ricci scalar

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<Xcts::Tags::ConformalMetric<
                     DataType, 3, Frame::Inertial>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<
          Xcts::Tags::ConformalMetric<DataType, 3, Frame::Inertial>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<Xcts::Tags::InverseConformalMetric<
                     DataType, 3, Frame::Inertial>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<
          Xcts::Tags::InverseConformalMetric<DataType, 3, Frame::Inertial>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<::Tags::deriv<
                     Xcts::Tags::ConformalMetric<DataType, 3, Frame::Inertial>,
                     tmpl::size_t<3>, Frame::Inertial>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<::Tags::deriv<
          Xcts::Tags::ConformalMetric<DataType, 3, Frame::Inertial>,
          tmpl::size_t<3>, Frame::Inertial>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<gr::Tags::TraceExtrinsicCurvature<DataType>> /*meta*/) const
      noexcept
      -> tuples::TaggedTuple<gr::Tags::TraceExtrinsicCurvature<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataType>,
                               tmpl::size_t<3>, Frame::Inertial>> /*meta*/)
      const noexcept -> tuples::TaggedTuple<
          ::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataType>,
                        tmpl::size_t<3>, Frame::Inertial>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<
          ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>>> /*meta*/)
      const noexcept -> tuples::TaggedTuple<
          ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<Xcts::Tags::ConformalFactor<DataType>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<Xcts::Tags::ConformalFactor<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>,
                               tmpl::size_t<3>, Frame::Inertial>> /*meta*/)
      const noexcept -> tuples::TaggedTuple<
          ::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>, tmpl::size_t<3>,
                        Frame::Inertial>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<Xcts::Tags::LapseTimesConformalFactor<DataType>> /*meta*/)
      const noexcept
      -> tuples::TaggedTuple<Xcts::Tags::LapseTimesConformalFactor<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
                               tmpl::size_t<3>, Frame::Inertial>> /*meta*/)
      const noexcept -> tuples::TaggedTuple<
          ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
                        tmpl::size_t<3>, Frame::Inertial>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<Xcts::Tags::ShiftBackground<
                     DataType, 3, Frame::Inertial>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<
          Xcts::Tags::ShiftBackground<DataType, 3, Frame::Inertial>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
          DataType, 3, Frame::Inertial>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<
          Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
              DataType, 3, Frame::Inertial>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<::Tags::div<
          Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
              DataType, 3, Frame::Inertial>>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<::Tags::div<
          Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
              DataType, 3, Frame::Inertial>>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<Xcts::Tags::ShiftExcess<
                     DataType, 3, Frame::Inertial>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<
          Xcts::Tags::ShiftExcess<DataType, 3, Frame::Inertial>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<
          Xcts::Tags::ShiftStrain<DataType, 3, Frame::Inertial>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<
          Xcts::Tags::ShiftStrain<DataType, 3, Frame::Inertial>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<::Tags::FixedSource<
                     Xcts::Tags::ConformalFactor<DataType>>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<
          ::Tags::FixedSource<Xcts::Tags::ConformalFactor<DataType>>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<::Tags::FixedSource<
                     Xcts::Tags::LapseTimesConformalFactor<DataType>>> /*meta*/)
      const noexcept -> tuples::TaggedTuple<
          ::Tags::FixedSource<Xcts::Tags::LapseTimesConformalFactor<DataType>>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<::Tags::FixedSource<Xcts::Tags::ShiftExcess<
                     DataType, 3, Frame::Inertial>>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<::Tags::FixedSource<
          Xcts::Tags::ShiftExcess<DataType, 3, Frame::Inertial>>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<gr::Tags::EnergyDensity<DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<gr::Tags::EnergyDensity<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<gr::Tags::StressTrace<DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<gr::Tags::StressTrace<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<gr::Tags::MomentumDensity<3, Frame::Inertial,
                                                      DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<
          gr::Tags::MomentumDensity<3, Frame::Inertial, DataType>>;
  // @}

  /// Retrieve a collection of variables at coordinates `x`
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataType, 3>& x,
                                         tmpl::list<Tags...> /*meta*/) const
      noexcept {
    static_assert(sizeof...(Tags) > 1, "The requested tag is not implemented.");
    return {tuples::get<Tags>(variables(x, tmpl::list<Tags>{}))...};
  }

  void pup(PUP::er& /* p */) noexcept {}  // NOLINT

 private:
  gr::Solutions::KerrSchild kerr_schild_solution_;
};

template <KerrCoordinates Coords>
SPECTRE_ALWAYS_INLINE bool operator==(const Kerr<Coords>& /*lhs*/,
                                      const Kerr<Coords>& /*rhs*/) noexcept {
  return true;
}

template <KerrCoordinates Coords>
SPECTRE_ALWAYS_INLINE bool operator!=(const Kerr<Coords>& /*lhs*/,
                                      const Kerr<Coords>& /*rhs*/) noexcept {
  return false;
}

}  // namespace Xcts::Solutions
