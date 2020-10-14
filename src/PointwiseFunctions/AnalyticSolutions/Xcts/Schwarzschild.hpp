// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <ostream>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Xcts::Solutions {

/// Various coordinate systems in which to express the Schwarzschild solution
enum class SchwarzschildCoordinates {
  /*!
   * \brief Isotropic Schwarzschild coordinates
   *
   * These arise from the canonical Schwarzschild coordinates by the radial
   * transformation
   *
   * \f{equation}
   * r = \bar{r}\left(1+\frac{M}{2\bar{r}}\right)^2
   * \f}
   *
   * (Eq. (1.61) in \cite BaumgarteShapiro) where \f$r\f$ is the canonical
   * Schwarzschild radius, also referred to as "areal" radius because it is
   * defined such that spheres with constant \f$r\f$ have the area \f$4\pi
   * r^2\f$, and \f$\bar{r}\f$ is the "isotropic" radius. In the isotropic
   * radius the Schwarzschild spatial metric is conformally flat
   *
   * \f{equation}
   * \gamma_{ij}=\psi^4\eta_{ij} \quad \text{with conformal factor} \quad
   * \psi=\left(1+\frac{M}{2\bar{r}}\right)^4
   * \f}
   *
   * (Table 2.1 in \cite BaumgarteShapiro). Its lapse transforms to
   *
   * \f{equation}
   * \alpha=\frac{1-M/(2\bar{r})}{1+M/(2\bar{r})}
   * \f}
   *
   * and the shift vanishes \f$\beta^i=0\f$ as it does in areal Schwarzschild
   * coordinates. The solution also remains maximally sliced \f$K=0\f$.
   *
   * The Schwarzschild horizon in these coordinates is at
   * \f$\bar{r}=\frac{M}{2}\f$ due to the radial transformation from \f$r=2M\f$.
   */
  Isotropic,

  /*!
   * \brief Painlevé-Gullstrand coordinates
   *
   * A coordinate system that arises from the canonical Schwarzschild
   * coordinates by a time coordinate transformation such that the lapse is just
   * \f$\alpha=1\f$ and the spatial metric is entirely flat
   * \f$\gamma_{ij}=\eta_{ij}\f$ (not only conformally flat). The shift vector
   * is
   *
   * \f{equation}
   * \beta^i=\left(\frac{2M}{r}\right)^{1/2} l^i
   * \f}
   *
   * with \f$l^i=l_i=\frac{x^i}{r}\f$ in Cartesian coordinates or
   * \f$l^i=l_i=\left(1,0,0\right)\f$ in spherical polar coordinates and the
   * spacetime is not maximally sliced but
   *
   * \f{equation}
   * K=\frac{3}{2}\left(\frac{2M}{r^3}\right)^{1/2}
   * \f}
   *
   * (Table 2.1 in \cite BaumgarteShapiro).
   *
   * Since only the time coordinate is transformed, the horizon remains at
   * \f$r=2M\f$.
   */
  PainleveGullstrand,

  /*!
   * \brief Kerr-Schild (or ingoing Eddington-Finkelstein) coordinates with an
   * isotropic radial transformation
   *
   * These coordinates arise from the Kerr-Schild coordinates (see e.g. Table
   * 2.1 in \cite BaumgarteShapiro) by a radial transformation that makes the
   * spatial metric conformally flat, just like the `Isotropic` coordinate
   * system makes the canonical Schwarzschild metric conformally flat. The
   * transformation from the "areal" Kerr-Schild radius \f$r\f$ to the
   * "isotropic" Kerr-Schild radius \f$\bar{r}\f$ is
   *
   * \f{equation}\label{eq:radial transform}
   * \bar{r}=\frac{r}{4}\left(1+\sqrt{1+\frac{2M}{r}}\right)^2
   * e^{2-2\sqrt{1+2M/r}}
   * \f}
   *
   * (Eq. (7.34) in \cite Pfeiffer2005zm). In the isotropic radius the spatial
   * metric is conformally flat \f$\gamma_{ij}=\psi^4 \eta_{ij}\f$ with the
   * conformal factor
   *
   * \f{equation}
   * \psi=\sqrt{\frac{r}{\bar{r}}}=\frac{2e^{\sqrt{1+2M/r}-1}}{1+\sqrt{1+2M/r}}
   * \f}
   *
   * (Eq. (7.35) in \cite Pfeiffer2005zm). The lapse and shift in these
   * coordinates are
   *
   * \f{align}
   * \alpha=\left(1+\frac{2M}{r(\bar{r})}\right)^{-1/2} \\
   * \beta^i=\frac{2M}{r(\bar{r})}\frac{\alpha}{\psi^2} l^i
   * \f}
   *
   * where we need to numerically invert the radial transformation
   * (\ref{eq:radial transform}) to find the areal radius \f$r\f$ at the given
   * isotropic radius \f$\bar{r}\f$ and where
   * \f$l^i=l_i=\frac{x^i}{\bar{r}}\f$ in Cartesian coordinates or
   * \f$l^i=l_i=\left(1,0,0\right)\f$ in spherical polar coordinates. Note that
   * these are the (isotropic) Cartesian coordinates that define the isotropic
   * radius \f$\bar{r}^2=x^2+y^2+z^2\f$.
   *
   * The extrinsic curvature is a scalar under the radial transformation
   * (\ref{eq:radial transform}) and therefore is
   *
   * \f{equation}
   * K=\frac{2M\alpha^3}{r(\bar{r})^2}\left(1+\frac{3M}{r(\bar{r})}\right)
   * \f}
   *
   * (Table 2.1 in \cite BaumgarteShapiro).
   *
   * In these coordinates the horizon is at an isotropic radius of
   * \f$\bar{r}=\frac{e^{2-2\sqrt{2}}}{2}\left(1+\sqrt{2}\right)^2
   * \approx 1.27274\f$ (Eq. (7.37) in \cite Pfeiffer2005zm).
   *
   * See section 7.4.1 in \cite Pfeiffer2005zm for more details.
   */
  KerrSchildIsotropic
};

std::ostream& operator<<(std::ostream& os,
                         const SchwarzschildCoordinates& coords) noexcept;

/*!
 * \brief Schwarzschild spacetime in general relativity
 *
 * This class implements the Schwarzschild solution with mass parameter
 * \f$M=1\f$ in various coordinate systems. See the entries of the
 * `Xcts::Solutions::SchwarzschildCoordinates` enum for the available coordinate
 * systems and for the solution variables in the respective coordinates.
 */
template <SchwarzschildCoordinates Coords>
class Schwarzschild {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Schwarzschild spacetime in general relativity"};

  Schwarzschild() = default;
  Schwarzschild(const Schwarzschild&) noexcept = delete;
  Schwarzschild& operator=(const Schwarzschild&) noexcept = delete;
  Schwarzschild(Schwarzschild&&) noexcept = default;
  Schwarzschild& operator=(Schwarzschild&&) noexcept = default;
  ~Schwarzschild() noexcept = default;

  /// The radius of the Schwarzschild horizon in the given coordinates.
  static double radius_at_horizon() noexcept;

  // @{
  /// Retrieve variable at coordinates `x`
  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<Xcts::Tags::ConformalMetric<
                     DataType, 3, Frame::Inertial>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<
          Xcts::Tags::ConformalMetric<DataType, 3, Frame::Inertial>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<gr::Tags::TraceExtrinsicCurvature<DataType>> /*meta*/) const
      noexcept
      -> tuples::TaggedTuple<gr::Tags::TraceExtrinsicCurvature<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataType>,
                               tmpl::size_t<3>, Frame::Inertial>> /*meta*/)
      const noexcept -> tuples::TaggedTuple<
          ::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataType>,
                        tmpl::size_t<3>, Frame::Inertial>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<
          ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>>> /*meta*/)
      const noexcept -> tuples::TaggedTuple<
          ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<Xcts::Tags::ConformalFactor<DataType>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<Xcts::Tags::ConformalFactor<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>,
                               tmpl::size_t<3>, Frame::Inertial>> /*meta*/)
      const noexcept -> tuples::TaggedTuple<
          ::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>, tmpl::size_t<3>,
                        Frame::Inertial>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<Xcts::Tags::LapseTimesConformalFactor<DataType>> /*meta*/)
      const noexcept
      -> tuples::TaggedTuple<Xcts::Tags::LapseTimesConformalFactor<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
                               tmpl::size_t<3>, Frame::Inertial>> /*meta*/)
      const noexcept -> tuples::TaggedTuple<
          ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
                        tmpl::size_t<3>, Frame::Inertial>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<Xcts::Tags::ShiftBackground<
                     DataType, 3, Frame::Inertial>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<
          Xcts::Tags::ShiftBackground<DataType, 3, Frame::Inertial>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
          DataType, 3, Frame::Inertial>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<
          Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
              DataType, 3, Frame::Inertial>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<::Tags::div<
          Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
              DataType, 3, Frame::Inertial>>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<::Tags::div<
          Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
              DataType, 3, Frame::Inertial>>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<Xcts::Tags::ShiftExcess<
                     DataType, 3, Frame::Inertial>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<
          Xcts::Tags::ShiftExcess<DataType, 3, Frame::Inertial>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<
          Xcts::Tags::ShiftStrain<DataType, 3, Frame::Inertial>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<
          Xcts::Tags::ShiftStrain<DataType, 3, Frame::Inertial>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<Xcts::Tags::LongitudinalShiftMinusDtConformalMetricSquare<
          DataType>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<
          Xcts::Tags::LongitudinalShiftMinusDtConformalMetricSquare<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<
          Xcts::Tags::LongitudinalShiftMinusDtConformalMetricOverLapseSquare<
              DataType>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<
          Xcts::Tags::LongitudinalShiftMinusDtConformalMetricOverLapseSquare<
              DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<Xcts::Tags::ShiftDotDerivExtrinsicCurvatureTrace<
                     DataType>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<
          Xcts::Tags::ShiftDotDerivExtrinsicCurvatureTrace<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<::Tags::FixedSource<
                     Xcts::Tags::ConformalFactor<DataType>>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<
          ::Tags::FixedSource<Xcts::Tags::ConformalFactor<DataType>>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<::Tags::FixedSource<
                     Xcts::Tags::LapseTimesConformalFactor<DataType>>> /*meta*/)
      const noexcept -> tuples::TaggedTuple<
          ::Tags::FixedSource<Xcts::Tags::LapseTimesConformalFactor<DataType>>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<::Tags::FixedSource<Xcts::Tags::ShiftExcess<
                     DataType, 3, Frame::Inertial>>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<::Tags::FixedSource<
          Xcts::Tags::ShiftExcess<DataType, 3, Frame::Inertial>>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<gr::Tags::EnergyDensity<DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<gr::Tags::EnergyDensity<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<gr::Tags::StressTrace<DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<gr::Tags::StressTrace<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<gr::Tags::MomentumDensity<3, Frame::Inertial,
                                                      DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<
          gr::Tags::MomentumDensity<3, Frame::Inertial, DataType>>;
  // @}

  /// Retrieve a collection of variables at coordinates `x`
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<Tags...> /*meta*/) const noexcept {
    static_assert(sizeof...(Tags) > 1, "The requested tag is not implemented.");
    return {tuples::get<Tags>(variables(x, tmpl::list<Tags>{}))...};
  }

  void pup(PUP::er& /* p */) noexcept {}  // NOLINT
};

template <SchwarzschildCoordinates Coords>
SPECTRE_ALWAYS_INLINE bool operator==(
    const Schwarzschild<Coords>& /*lhs*/,
    const Schwarzschild<Coords>& /*rhs*/) noexcept {
  return true;
}

template <SchwarzschildCoordinates Coords>
SPECTRE_ALWAYS_INLINE bool operator!=(
    const Schwarzschild<Coords>& /*lhs*/,
    const Schwarzschild<Coords>& /*rhs*/) noexcept {
  return false;
}

}  // namespace Xcts::Solutions
