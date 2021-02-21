// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Elliptic/Systems/Xcts/Equations.hpp"
#include "Elliptic/Systems/Xcts/Geometry.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace Xcts {

/*!
 * \brief The Extended Conformal Thin Sandwich (XCTS) decomposition of the
 * Einstein constraint equations, formulated as a set of coupled first-order
 * partial differential equations
 *
 * See \ref Xcts for details on the XCTS equations. This system introduces as
 * auxiliary variables the conformal factor gradient \f$v_i=\partial_i\psi\f$,
 * the symmetric shift strain \f$B_{ij}=\bar{D}_{(i}\beta_{j)}\f$ and the
 * gradient of the lapse times the conformal factor
 * \f$w_i=\partial_i\left(\alpha\psi\right)\f$. When we then cast the equations
 * in first-order flux-form
 *
 * \f[
 * -\partial_i F^i_A + S_A = f_A(x)
 * \f]
 *
 * (see also `Poisson::FirstOrderSystem`), the fluxes \f$F^i_A\f$, sources
 * \f$S_A\f$ and fixed-sources \f$f_A\f$ are:
 *
 * \f{align}
 * F^i_\psi &= \bar{\gamma}^{ij} v_j \\
 * S_\psi &= -\bar{Gamma}^i_{ij}\bar{\gamma}^{jk}v_k
 * + \frac{1}{8}\psi\bar{R} + \frac{1}{12}\psi^5 K^2
 * - \frac{1}{8}\psi^{-7}\bar{A}^2 - 2\pi\psi^5\rho \\
 * f_\psi &= 0 \\
 * F^i_{\beta^j} &= \left(\bar{L}\beta\right)^{ij} \\
 * S_{\beta^i} &= -\bar{\Gamma}^j_{jk} \left(\bar{L}\beta\right)^{ik}
 * - \bar{\Gamma}^i_{jk} \left(\bar{L}\beta\right)^{jk}
 * + \left( \left(\bar{L}\beta\right)^{ij}
 * + \left(\bar{L}\beta_\mathrm{background}\right)^{ij} - \bar{u}^{ij}\right)
 * \left(\frac{w_j}{\alpha\psi} - 7 \frac{v_j}{\psi}\right)
 * - \bar{D}_j\left(\left(\bar{L}\beta_\mathrm{background}\right)^{ij}
 * - \bar{u}^{ij}\right)
 * + \frac{4}{3}\frac{\alpha\psi}{\psi}\bar{D}^i K
 * + 16\pi\left(\alpha\psi\right)\psi^3 S^i \\
 * f_{\beta^i} &= 0 \\
 * F^i_{\alpha\psi} &= \bar{\gamma}^{ij} w^j \\
 * S_{\alpha\psi} &= -\bar{\Gamma}^i_{ij}\bar{\gamma}^{jk}w_k
 * + \alpha\psi \left( \frac{7}{8}\psi^{-8} \bar{A}^2
 * + \frac{5}{12} \psi^4 K^2 + \frac{1}{8}\bar{R}
 * + 2\pi\psi^4\left(\rho + 2S\right) \right)
 * - \psi^5\partial_t K + \psi^5\left(\beta^i\bar{D}_i K
 * + \beta_\mathrm{background}^i\bar{D}_i K\right) \\
 * f_{\alpha\psi} &= 0 \\
 * F^i_{v_j} &= \delta^i_j \psi \\
 * S_{v_j} &= v_j \\
 * f_{v_j} &= 0 \\
 * F^i_{B_{jk}} &= \delta^i_{(j} \beta_{k)} \\
 * S_{B_{jk}} &= B_{jk} + \bar{\Gamma}_{ijk}\beta^i \\
 * f_{B_{jk}} &= 0 \\
 * F^i_{w_j} &= \delta^i_j \alpha\psi \\
 * S_{w_j} &= w_j \\
 * f_{w_j} &= 0 \\
 * \text{with} \quad \bar{A}^{ij} &= \frac{\psi^7}{2\alpha\psi}\left(
 * \left(\bar{L}\beta\right)^{ij} +
 * \left(\bar{L}\beta_\mathrm{background}\right)^{ij} - \bar{u}^{ij} \right) \\
 * \text{and} \quad \left(\bar{L}\beta\right)^{ij} &=
 * 2\left(\bar{gamma}^{ik}\bar{gamma}^{jl} - \frac{1}{3}
 * \bar{gamma}^{jk}\bar{gamma}^{kl}\right) B_{kl}
 * \f}
 *
 * Note that the symbol \f$\beta\f$ in the equations above means
 * \f$\beta_\mathrm{excess}\f$. The full shift is \f$\beta_\mathrm{excess} +
 * \beta_\mathrm{background}\f$. Also note that the background shift is
 * degenerate with \f$\bar{u}\f$ so we treat the quantity
 * \f$\left(\bar{L}\beta_\mathrm{background}\right)^{ij} - \bar{u}^{ij}\f$
 * as a single background field. The covariant divergence of this quantity
 * w.r.t. the conformal metric is also a background field.
 *
 * \par Solving a subset of equations:
 * This system allows you to select a subset of `Xcts::Equations` so you don't
 * have to solve for all variables if some are analytically known. Specify the
 * set of enabled equations as the first template parameter. The set of required
 * background fields depends on your choice of equations.
 *
 * \par Conformal background geometry:
 * The equations simplify significantly if the conformal metric is flat
 * ("conformal flatness") and in Cartesian coordinates. In this case you can
 * specify `Xcts::Geometry::FlatCartesian` as the second template paramter so
 * computations are optimized for a flat background geometry and you don't have
 * to supply geometric background fields.
 */
template <Equations EnabledEquations, Geometry ConformalGeometry>
struct FirstOrderSystem {
 private:
  using conformal_factor = Tags::ConformalFactor<DataVector>;
  using conformal_factor_gradient =
      ::Tags::deriv<conformal_factor, tmpl::size_t<3>, Frame::Inertial>;
  using lapse_times_conformal_factor =
      Tags::LapseTimesConformalFactor<DataVector>;
  using lapse_times_conformal_factor_gradient =
      ::Tags::deriv<lapse_times_conformal_factor, tmpl::size_t<3>,
                    Frame::Inertial>;
  using shift_excess = Tags::ShiftExcess<DataVector, 3, Frame::Inertial>;
  using shift_strain = Tags::ShiftStrain<DataVector, 3, Frame::Inertial>;
  using longitudinal_shift_excess =
      Tags::LongitudinalShiftExcess<DataVector, 3, Frame::Inertial>;

 public:
  static constexpr size_t volume_dim = 3;

  // The physical fields to solve for
  using primal_fields = tmpl::flatten<tmpl::list<
      conformal_factor,
      tmpl::conditional_t<EnabledEquations == Equations::HamiltonianAndLapse or
                              EnabledEquations ==
                                  Equations::HamiltonianLapseAndShift,
                          lapse_times_conformal_factor, tmpl::list<>>,
      tmpl::conditional_t<EnabledEquations ==
                              Equations::HamiltonianLapseAndShift,
                          shift_excess, tmpl::list<>>>>;
  using auxiliary_fields = tmpl::flatten<tmpl::list<
      conformal_factor_gradient,
      tmpl::conditional_t<EnabledEquations == Equations::HamiltonianAndLapse or
                              EnabledEquations ==
                                  Equations::HamiltonianLapseAndShift,
                          lapse_times_conformal_factor_gradient, tmpl::list<>>,
      tmpl::conditional_t<EnabledEquations ==
                              Equations::HamiltonianLapseAndShift,
                          shift_strain, tmpl::list<>>>>;

  // For fluxes we use the gradients with raised indices for the conformal
  // factor and the lapse-times-conformal-factor and the longitudinal shift
  // excess for the momentum constraint. The gradient fluxes don't have
  // symmetries and no particular meaning so we use the standard `Flux` tags,
  // but for the symmetric longitudinal shift we use the appropriate tag.
  using primal_fluxes = tmpl::flatten<tmpl::list<
      ::Tags::Flux<conformal_factor, tmpl::size_t<3>, Frame::Inertial>,
      tmpl::conditional_t<EnabledEquations == Equations::HamiltonianAndLapse or
                              EnabledEquations ==
                                  Equations::HamiltonianLapseAndShift,
                          ::Tags::Flux<lapse_times_conformal_factor,
                                       tmpl::size_t<3>, Frame::Inertial>,
                          tmpl::list<>>,
      tmpl::conditional_t<EnabledEquations ==
                              Equations::HamiltonianLapseAndShift,
                          longitudinal_shift_excess, tmpl::list<>>>>;
  using auxiliary_fluxes = db::wrap_tags_in<::Tags::Flux, auxiliary_fields,
                                            tmpl::size_t<3>, Frame::Inertial>;

  using background_fields = tmpl::flatten<tmpl::list<
      // Quantities for Hamiltonian constraint
      gr::Tags::EnergyDensity<DataVector>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>,
      tmpl::conditional_t<ConformalGeometry == Geometry::Curved,
                          tmpl::list<Tags::InverseConformalMetric<
                                         DataVector, 3, Frame::Inertial>,
                                     Tags::ConformalRicciScalar<DataVector>,
                                     Tags::ConformalChristoffelContracted<
                                         DataVector, 3, Frame::Inertial>>,
                          tmpl::list<>>,
      tmpl::conditional_t<
          EnabledEquations == Equations::Hamiltonian,
          Tags::LongitudinalShiftMinusDtConformalMetricOverLapseSquare<
              DataVector>,
          tmpl::list<>>,
      // Additional quantities for lapse equation
      tmpl::conditional_t<
          EnabledEquations == Equations::HamiltonianAndLapse or
              EnabledEquations == Equations::HamiltonianLapseAndShift,
          tmpl::list<gr::Tags::StressTrace<DataVector>,
                     ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataVector>>>,
          tmpl::list<>>,
      tmpl::conditional_t<
          EnabledEquations ==
              Equations::HamiltonianAndLapse,
          tmpl::list<
              Tags::LongitudinalShiftMinusDtConformalMetricSquare<DataVector>,
              Tags::ShiftDotDerivExtrinsicCurvatureTrace<DataVector>>,
          tmpl::list<>>,
      // Additional quantities for momentum constraint
      tmpl::conditional_t<
          EnabledEquations ==
              Equations::HamiltonianLapseAndShift,
          tmpl::list<
              gr::Tags::MomentumDensity<3, Frame::Inertial, DataVector>,
              ::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataVector>,
                            tmpl::size_t<3>, Frame::Inertial>,
              Tags::ShiftBackground<DataVector, 3, Frame::Inertial>,
              Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
                  DataVector, 3, Frame::Inertial>,
              // Note that this is the plain divergence, i.e. with no
              // Christoffel symbol terms added
              ::Tags::div<
                  Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
                      DataVector, 3, Frame::Inertial>>,
              tmpl::conditional_t<
                  ConformalGeometry == Geometry::Curved,
                  tmpl::list<
                      Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
                      Tags::ConformalChristoffelFirstKind<DataVector, 3,
                                                          Frame::Inertial>,
                      Tags::ConformalChristoffelSecondKind<DataVector, 3,
                                                           Frame::Inertial>>,
                  tmpl::list<>>>,
          tmpl::list<>>>>;

  using fluxes_computer = Fluxes<EnabledEquations, ConformalGeometry>;
  using sources_computer = Sources<EnabledEquations, ConformalGeometry>;
  using sources_computer_linearized =
      LinearizedSources<EnabledEquations, ConformalGeometry>;

  // The tag of the operator to compute magnitudes on the manifold, e.g. to
  // normalize vectors on the faces of an element
  using inv_metric_tag = tmpl::conditional_t<
      ConformalGeometry == Geometry::FlatCartesian, void,
      Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>>;
  template <typename Tag>
  using magnitude_tag =
      tmpl::conditional_t<ConformalGeometry == Geometry::FlatCartesian,
                          ::Tags::EuclideanMagnitude<Tag>,
                          ::Tags::NonEuclideanMagnitude<Tag, inv_metric_tag>>;
};

}  // namespace Xcts
