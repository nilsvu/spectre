// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <string>

#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Elliptic/Systems/Elasticity/Equations.hpp"
#include "Elliptic/Systems/Poisson/Equations.hpp"
#include "Elliptic/Systems/Xcts/Geometry.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "PointwiseFunctions/Xcts/LongitudinalOperator.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace Xcts {

/// Indicates a subset of the XCTS equations
enum class Equations {
  /// Only the Hamiltonian constraint, solved for \f$\psi\f$
  Hamiltonian,
  /// Both the Hamiltonian constraint and the lapse equation, solved for
  /// \f$\psi\f$ and \f$\alpha\psi\f$
  HamiltonianAndLapse,
  /// The full XCTS equations, solved for \f$\psi\f$, \f$\alpha\psi\f$ and
  /// \f$\beta_\mathrm{excess}\f$
  HamiltonianLapseAndShift
};

namespace detail {
// Tensor-contraction helper functions that should be replaced by tensor
// expressions once those work
template <typename DataType>
void fully_contract(gsl::not_null<Scalar<DataType>*> result,
                    const tnsr::II<DataType, 3>& tensor1,
                    const tnsr::II<DataType, 3>& tensor2) noexcept;

template <typename DataType>
void fully_contract(gsl::not_null<Scalar<DataType>*> result,
                    const tnsr::II<DataType, 3>& tensor1,
                    const tnsr::II<DataType, 3>& tensor2,
                    const tnsr::ii<DataType, 3>& metric) noexcept;
}  // namespace detail

/// The fluxes \f$F^i\f$ for the first-order formulation of the XCTS equations.
///
/// \see Xcts::FirstOrderSystem
template <Equations EnabledEquations, Geometry ConformalGeometry>
struct Fluxes;

/// \cond
template <>
struct Fluxes<Equations::Hamiltonian, Geometry::FlatCartesian> {
  using argument_tags = tmpl::list<>;
  static void apply(
      const gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_conformal_factor,
      const tnsr::i<DataVector, 3>& conformal_factor_gradient) noexcept {
    Poisson::flat_cartesian_fluxes(flux_for_conformal_factor,
                                   conformal_factor_gradient);
  }
  static void apply(const gsl::not_null<tnsr::Ij<DataVector, 3>*>
                        flux_for_conformal_factor_gradient,
                    const Scalar<DataVector>& conformal_factor) noexcept {
    Poisson::auxiliary_fluxes(flux_for_conformal_factor_gradient,
                              conformal_factor);
  }
};

template <>
struct Fluxes<Equations::Hamiltonian, Geometry::Curved> {
  using argument_tags =
      tmpl::list<Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>>;
  static void apply(
      const gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_conformal_factor,
      const tnsr::II<DataVector, 3>& inv_conformal_metric,
      const tnsr::i<DataVector, 3>& conformal_factor_gradient) noexcept {
    Poisson::curved_fluxes(flux_for_conformal_factor, inv_conformal_metric,
                           conformal_factor_gradient);
  }
  static void apply(const gsl::not_null<tnsr::Ij<DataVector, 3>*>
                        flux_for_conformal_factor_gradient,
                    const tnsr::II<DataVector, 3>& /*inv_conformal_metric*/,
                    const Scalar<DataVector>& conformal_factor) noexcept {
    Poisson::auxiliary_fluxes(flux_for_conformal_factor_gradient,
                              conformal_factor);
  }
};

template <>
struct Fluxes<Equations::HamiltonianAndLapse, Geometry::FlatCartesian> {
  using argument_tags = tmpl::list<>;
  static void apply(
      const gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_conformal_factor,
      const gsl::not_null<tnsr::I<DataVector, 3>*>
          flux_for_lapse_times_conformal_factor,
      const tnsr::i<DataVector, 3>& conformal_factor_gradient,
      const tnsr::i<DataVector, 3>&
          lapse_times_conformal_factor_gradient) noexcept {
    Fluxes<Equations::Hamiltonian, Geometry::FlatCartesian>::apply(
        flux_for_conformal_factor, conformal_factor_gradient);
    Poisson::flat_cartesian_fluxes(flux_for_lapse_times_conformal_factor,
                                   lapse_times_conformal_factor_gradient);
  }
  static void apply(
      const gsl::not_null<tnsr::Ij<DataVector, 3>*>
          flux_for_conformal_factor_gradient,
      const gsl::not_null<tnsr::Ij<DataVector, 3>*>
          flux_for_lapse_times_conformal_factor_gradient,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& lapse_times_conformal_factor) noexcept {
    Fluxes<Equations::Hamiltonian, Geometry::FlatCartesian>::apply(
        flux_for_conformal_factor_gradient, conformal_factor);
    Poisson::auxiliary_fluxes(flux_for_lapse_times_conformal_factor_gradient,
                              lapse_times_conformal_factor);
  }
};

template <>
struct Fluxes<Equations::HamiltonianAndLapse, Geometry::Curved> {
  using argument_tags =
      tmpl::list<Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>>;
  static void apply(
      const gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_conformal_factor,
      const gsl::not_null<tnsr::I<DataVector, 3>*>
          flux_for_lapse_times_conformal_factor,
      const tnsr::II<DataVector, 3>& inv_conformal_metric,
      const tnsr::i<DataVector, 3>& conformal_factor_gradient,
      const tnsr::i<DataVector, 3>&
          lapse_times_conformal_factor_gradient) noexcept {
    Fluxes<Equations::Hamiltonian, Geometry::Curved>::apply(
        flux_for_conformal_factor, inv_conformal_metric,
        conformal_factor_gradient);
    Poisson::curved_fluxes(flux_for_lapse_times_conformal_factor,
                           inv_conformal_metric,
                           lapse_times_conformal_factor_gradient);
  }
  static void apply(
      const gsl::not_null<tnsr::Ij<DataVector, 3>*>
          flux_for_conformal_factor_gradient,
      const gsl::not_null<tnsr::Ij<DataVector, 3>*>
          flux_for_lapse_times_conformal_factor_gradient,
      const tnsr::II<DataVector, 3>& inv_conformal_metric,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& lapse_times_conformal_factor) noexcept {
    Fluxes<Equations::Hamiltonian, Geometry::Curved>::apply(
        flux_for_conformal_factor_gradient, inv_conformal_metric,
        conformal_factor);
    Poisson::auxiliary_fluxes(flux_for_lapse_times_conformal_factor_gradient,
                              lapse_times_conformal_factor);
  }
};

template <>
struct Fluxes<Equations::HamiltonianLapseAndShift, Geometry::FlatCartesian> {
  using argument_tags = tmpl::list<>;
  static void apply(
      const gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_conformal_factor,
      const gsl::not_null<tnsr::I<DataVector, 3>*>
          flux_for_lapse_times_conformal_factor,
      const gsl::not_null<tnsr::II<DataVector, 3>*> longitudinal_shift_excess,
      const tnsr::i<DataVector, 3>& conformal_factor_gradient,
      const tnsr::i<DataVector, 3>& lapse_times_conformal_factor_gradient,
      const tnsr::ii<DataVector, 3>& shift_strain) noexcept {
    Fluxes<Equations::HamiltonianAndLapse, Geometry::FlatCartesian>::apply(
        flux_for_conformal_factor, flux_for_lapse_times_conformal_factor,
        conformal_factor_gradient, lapse_times_conformal_factor_gradient);
    Xcts::longitudinal_operator_flat_cartesian(longitudinal_shift_excess,
                                               shift_strain);
  }
  static void apply(
      const gsl::not_null<tnsr::Ij<DataVector, 3>*>
          flux_for_conformal_factor_gradient,
      const gsl::not_null<tnsr::Ij<DataVector, 3>*>
          flux_for_lapse_times_conformal_factor_gradient,
      const gsl::not_null<tnsr::Ijj<DataVector, 3>*> flux_for_shift_strain,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& lapse_times_conformal_factor,
      const tnsr::I<DataVector, 3>& shift) noexcept {
    Fluxes<Equations::HamiltonianAndLapse, Geometry::FlatCartesian>::apply(
        flux_for_conformal_factor_gradient,
        flux_for_lapse_times_conformal_factor_gradient, conformal_factor,
        lapse_times_conformal_factor);
    Elasticity::auxiliary_fluxes(flux_for_shift_strain, shift);
  }
};

template <>
struct Fluxes<Equations::HamiltonianLapseAndShift, Geometry::Curved> {
  using argument_tags =
      tmpl::list<Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
                 Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>>;
  static void apply(
      const gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_conformal_factor,
      const gsl::not_null<tnsr::I<DataVector, 3>*>
          flux_for_lapse_times_conformal_factor,
      const gsl::not_null<tnsr::II<DataVector, 3>*> longitudinal_shift_excess,
      const tnsr::ii<DataVector, 3>& /*conformal_metric*/,
      const tnsr::II<DataVector, 3>& inv_conformal_metric,
      const tnsr::i<DataVector, 3>& conformal_factor_gradient,
      const tnsr::i<DataVector, 3>& lapse_times_conformal_factor_gradient,
      const tnsr::ii<DataVector, 3>& shift_strain) noexcept {
    Fluxes<Equations::HamiltonianAndLapse, Geometry::Curved>::apply(
        flux_for_conformal_factor, flux_for_lapse_times_conformal_factor,
        inv_conformal_metric, conformal_factor_gradient,
        lapse_times_conformal_factor_gradient);
    Xcts::longitudinal_operator(longitudinal_shift_excess, shift_strain,
                                inv_conformal_metric);
  }
  static void apply(
      const gsl::not_null<tnsr::Ij<DataVector, 3>*>
          flux_for_conformal_factor_gradient,
      const gsl::not_null<tnsr::Ij<DataVector, 3>*>
          flux_for_lapse_times_conformal_factor_gradient,
      const gsl::not_null<tnsr::Ijj<DataVector, 3>*> flux_for_shift_strain,
      const tnsr::ii<DataVector, 3>& conformal_metric,
      const tnsr::II<DataVector, 3>& inv_conformal_metric,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& lapse_times_conformal_factor,
      const tnsr::I<DataVector, 3>& shift_excess) noexcept {
    Fluxes<Equations::HamiltonianAndLapse, Geometry::Curved>::apply(
        flux_for_conformal_factor_gradient,
        flux_for_lapse_times_conformal_factor_gradient, inv_conformal_metric,
        conformal_factor, lapse_times_conformal_factor);
    Elasticity::curved_auxiliary_fluxes(flux_for_shift_strain, conformal_metric,
                                        shift_excess);
  }
};
/// \endcond

/*!
 * \brief Add the nonlinear source to the Hamiltonian constraint on a flat
 * conformal background in Cartesian coordinates and with
 * \f$\bar{u}_{ij}=0=\beta^i\f$.
 *
 * Adds \f$\frac{1}{12}\psi^5 K^2 - 2\pi\psi^5\rho\f$. Additional sources can be
 * added with `add_distortion_hamiltonian_sources` and
 * `add_curved_hamiltonian_or_lapse_sources`.
 *
 * \see `Xcts::FirstOrderSystem`
 */
void add_hamiltonian_sources(
    gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
    const Scalar<DataVector>& energy_density,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const Scalar<DataVector>& conformal_factor) noexcept;

/*!
 * \brief The linearization of `add_hamiltonian_sources`
 *
 * \see `Xcts::FirstOrderSystem`
 */
void add_linearized_hamiltonian_sources(
    gsl::not_null<Scalar<DataVector>*> linearized_hamiltonian_constraint,
    const Scalar<DataVector>& energy_density,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& conformal_factor_correction) noexcept;

/*!
 * \brief Add the "distortion" source term to the Hamiltonian constraint.
 *
 * Adds \f$-\frac{1}{8}\psi^{-7}\bar{A}^2\f$.
 *
 * \see `Xcts::FirstOrderSystem`
 */
void add_distortion_hamiltonian_sources(
    gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
    const Scalar<DataVector>&
        longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
    const Scalar<DataVector>& conformal_factor) noexcept;

/*!
 * \brief The linearization of `add_distortion_hamiltonian_sources`
 *
 * Note that this linearization is only w.r.t. \f$\psi\f$.
 *
 * \see `Xcts::FirstOrderSystem`
 */
void add_linearized_distortion_hamiltonian_sources(
    gsl::not_null<Scalar<DataVector>*> linearized_hamiltonian_constraint,
    const Scalar<DataVector>&
        longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& conformal_factor_correction) noexcept;

/*!
 * \brief Add the contributions from a curved background geometry to the
 * Hamiltonian constraint or lapse equation
 *
 * Adds \f$\frac{1}{8}\psi\bar{R}\f$. This term appears both in the Hamiltonian
 * constraint and the lapse equation (where in the latter \f$\psi\f$ is replaced
 * by \f$\alpha\psi\f$).
 *
 * This term is linear.
 *
 * \see `Xcts::FirstOrderSystem`
 */
void add_curved_hamiltonian_or_lapse_sources(
    gsl::not_null<Scalar<DataVector>*> hamiltonian_or_lapse_equation,
    const Scalar<DataVector>& conformal_ricci_scalar,
    const Scalar<DataVector>& field) noexcept;

/*!
 * \brief Add the nonlinear source to the lapse equation on a flat conformal
 * background in Cartesian coordinates and with \f$\bar{u}_{ij}=0=\beta^i\f$.
 *
 * Adds \f$(\alpha\psi)\psi^4\left(\frac{5}{12}K^2 + 2\pi\left(\rho +
 * 2S\right)\right) + \psi^5 \left(\beta^i\partial_i K - \partial_t K\right)\f$.
 * Additional sources can be added with
 * `add_distortion_hamiltonian_and_lapse_sources` and
 * `add_curved_hamiltonian_or_lapse_sources`.
 *
 * \see `Xcts::FirstOrderSystem`
 */
void add_lapse_sources(
    gsl::not_null<Scalar<DataVector>*> lapse_equation,
    const Scalar<DataVector>& energy_density,
    const Scalar<DataVector>& stress_trace,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const Scalar<DataVector>& dt_extrinsic_curvature_trace,
    const Scalar<DataVector>& shift_dot_deriv_extrinsic_curvature_trace,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor) noexcept;

/*!
 * \brief The linearization of `add_lapse_sources`
 *
 * Note that this linearization is only w.r.t. \f$\psi\f$ and \f$\alpha\psi\f$.
 * The linearization w.r.t. \f$\beta^i\f$ is added in
 * `add_curved_linearized_momentum_sources` /
 * `add_flat_cartesian_linearized_momentum_sources`.
 *
 * \see `Xcts::FirstOrderSystem`
 */
void add_linearized_lapse_sources(
    gsl::not_null<Scalar<DataVector>*> linearized_lapse_equation,
    const Scalar<DataVector>& energy_density,
    const Scalar<DataVector>& stress_trace,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const Scalar<DataVector>& dt_extrinsic_curvature_trace,
    const Scalar<DataVector>& shift_dot_deriv_extrinsic_curvature_trace,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>& lapse_times_conformal_factor_correction) noexcept;

/*!
 * \brief Add the "distortion" source term to the Hamiltonian constraint and the
 * lapse equation.
 *
 * Adds \f$-\frac{1}{8}\psi^{-7}\bar{A}^2\f$ to the Hamiltonian constraint and
 * \f$\frac{7}{8}\alpha\psi^{-7}\bar{A}^2\f$ to the lapse equation.
 *
 * \see `Xcts::FirstOrderSystem`
 */
void add_distortion_hamiltonian_and_lapse_sources(
    gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
    gsl::not_null<Scalar<DataVector>*> lapse_equation,
    const Scalar<DataVector>&
        longitudinal_shift_minus_dt_conformal_metric_square,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor) noexcept;

/*!
 * \brief The linearization of `add_distortion_hamiltonian_and_lapse_sources`
 *
 * Note that this linearization is only w.r.t. \f$\psi\f$ and \f$\alpha\psi\f$.
 *
 * \see `Xcts::FirstOrderSystem`
 */
void add_linearized_distortion_hamiltonian_and_lapse_sources(
    gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
    gsl::not_null<Scalar<DataVector>*> lapse_equation,
    const Scalar<DataVector>&
        longitudinal_shift_minus_dt_conformal_metric_square,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>& lapse_times_conformal_factor_correction) noexcept;

/*!
 * \brief Add the nonlinear source to the momentum constraint and add the
 * "distortion" source term to the Hamiltonian constraint and lapse equation.
 *
 * Adds \f$\left((\bar{L}\beta)^{ij} -
 * \bar{u}^{ij}\right)\left(\frac{w_j}{\alpha\psi} - 7 \frac{v_j}{\psi}\right)
 * + \partial_j\bar{u}^{ij}
 * + \frac{4}{3}\frac{\alpha\psi}{\psi}\bar{\gamma}^{ij}\partial_j K
 * + 16\pi\left(\alpha\psi\right)\psi^3 S^i\f$ to the momentum constraint.
 *
 * Note that the \f$\partial_j\bar{u}^{ij}\f$ term is not the full covariant
 * divergence, but only the partial-derivatives part of it. The curved
 * contribution to this term can be added together with the curved contribution
 * to the flux divergence of the dynamic shift variable with the
 * `Elasticity::add_curved_sources` function.
 *
 * Also adds \f$-\frac{1}{8}\psi^{-7}\bar{A}^2\f$ to the Hamiltonian constraint
 * and \f$\frac{7}{8}\alpha\psi^{-7}\bar{A}^2\f$ to the lapse equation.
 *
 * \see `Xcts::FirstOrderSystem`
 */
//@{
void add_curved_momentum_sources(
    gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
    gsl::not_null<Scalar<DataVector>*> lapse_equation,
    gsl::not_null<tnsr::I<DataVector, 3>*> momentum_constraint,
    const tnsr::I<DataVector, 3>& momentum_density,
    const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::I<DataVector, 3>& minus_div_dt_conformal_metric,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, 3>& conformal_factor_flux,
    const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,
    const tnsr::II<DataVector, 3>&
        longitudinal_shift_minus_dt_conformal_metric) noexcept;

void add_flat_cartesian_momentum_sources(
    gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
    gsl::not_null<Scalar<DataVector>*> lapse_equation,
    gsl::not_null<tnsr::I<DataVector, 3>*> momentum_constraint,
    const tnsr::I<DataVector, 3>& momentum_density,
    const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,
    const tnsr::I<DataVector, 3>& minus_div_dt_conformal_metric,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, 3>& conformal_factor_flux,
    const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,
    const tnsr::II<DataVector, 3>&
        longitudinal_shift_minus_dt_conformal_metric) noexcept;
//@}

/*!
 * \brief The linearization of `add_curved_momentum_sources` /
 * `add_flat_cartesian_momentum_sources`
 *
 * \see `Xcts::FirstOrderSystem`
 */
//@{
void add_curved_linearized_momentum_sources(
    gsl::not_null<Scalar<DataVector>*> linearized_hamiltonian_constraint,
    gsl::not_null<Scalar<DataVector>*> linearized_lapse_equation,
    gsl::not_null<tnsr::I<DataVector, 3>*> linearized_momentum_constraint,
    const tnsr::I<DataVector, 3>& momentum_density,
    const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, 3>& conformal_factor_flux,
    const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,
    const tnsr::II<DataVector, 3>& longitudinal_shift_minus_dt_conformal_metric,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>& lapse_times_conformal_factor_correction,
    const tnsr::I<DataVector, 3>& shift_correction,
    const tnsr::I<DataVector, 3>& conformal_factor_flux_correction,
    const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux_correction,
    const tnsr::II<DataVector, 3>& longitudinal_shift_correction) noexcept;

void add_flat_cartesian_linearized_momentum_sources(
    gsl::not_null<Scalar<DataVector>*> linearized_hamiltonian_constraint,
    gsl::not_null<Scalar<DataVector>*> linearized_lapse_equation,
    gsl::not_null<tnsr::I<DataVector, 3>*> linearized_momentum_constraint,
    const tnsr::I<DataVector, 3>& momentum_density,
    const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, 3>& conformal_factor_flux,
    const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,
    const tnsr::II<DataVector, 3>& longitudinal_shift_minus_dt_conformal_metric,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>& lapse_times_conformal_factor_correction,
    const tnsr::I<DataVector, 3>& shift_correction,
    const tnsr::I<DataVector, 3>& conformal_factor_flux_correction,
    const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux_correction,
    const tnsr::II<DataVector, 3>& longitudinal_shift_correction) noexcept;
//@}

/// The sources \f$S\f$ for the first-order formulation of the XCTS equations.
///
/// \see Xcts::FirstOrderSystem
template <Equations EnabledEquations, Geometry ConformalGeometry>
struct Sources;

/// \cond
template <>
struct Sources<Equations::Hamiltonian, Geometry::FlatCartesian> {
  using argument_tags = tmpl::list<
      gr::Tags::EnergyDensity<DataVector>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>,
      Tags::LongitudinalShiftMinusDtConformalMetricOverLapseSquare<DataVector>>;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
      const Scalar<DataVector>& energy_density,
      const Scalar<DataVector>& extrinsic_curvature_trace,
      const Scalar<DataVector>&
          longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
      const Scalar<DataVector>& conformal_factor,
      const tnsr::I<DataVector, 3>& /*conformal_factor_flux*/) noexcept {
    add_hamiltonian_sources(hamiltonian_constraint, energy_density,
                            extrinsic_curvature_trace, conformal_factor);
    add_distortion_hamiltonian_sources(
        hamiltonian_constraint,
        longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
        conformal_factor);
  }
  static void apply(
      const gsl::not_null<tnsr::i<DataVector, 3>*>
      /*equation_for_conformal_factor_gradient*/,
      const Scalar<DataVector>& /*energy_density*/,
      const Scalar<DataVector>& /*extrinsic_curvature_trace*/,
      const Scalar<DataVector>&
      /*longitudinal_shift_minus_dt_conformal_metric_over_lapse_square*/,
      const Scalar<DataVector>& /*conformal_factor*/) noexcept {}
};

template <>
struct Sources<Equations::Hamiltonian, Geometry::Curved> {
  using argument_tags = tmpl::push_back<
      Sources<Equations::Hamiltonian, Geometry::FlatCartesian>::argument_tags,
      Tags::ConformalChristoffelContracted<DataVector, 3, Frame::Inertial>,
      Tags::ConformalRicciScalar<DataVector>>;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
      const Scalar<DataVector>& energy_density,
      const Scalar<DataVector>& extrinsic_curvature_trace,
      const Scalar<DataVector>&
          longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
      const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
      const Scalar<DataVector>& conformal_ricci_scalar,
      const Scalar<DataVector>& conformal_factor,
      const tnsr::I<DataVector, 3>& conformal_factor_flux) noexcept {
    Sources<Equations::Hamiltonian, Geometry::FlatCartesian>::apply(
        hamiltonian_constraint, energy_density, extrinsic_curvature_trace,
        longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
        conformal_factor, conformal_factor_flux);
    add_curved_hamiltonian_or_lapse_sources(
        hamiltonian_constraint, conformal_ricci_scalar, conformal_factor);
    Poisson::add_curved_sources(hamiltonian_constraint,
                                conformal_christoffel_contracted,
                                conformal_factor_flux);
  }
  static void apply(
      const gsl::not_null<tnsr::i<DataVector, 3>*>
      /*equation_for_conformal_factor_gradient*/,
      const Scalar<DataVector>& /*energy_density*/,
      const Scalar<DataVector>& /*extrinsic_curvature_trace*/,
      const Scalar<DataVector>&
      /*longitudinal_shift_minus_dt_conformal_metric_over_lapse_square*/,
      const tnsr::i<DataVector, 3>& /*conformal_christoffel_contracted*/,
      const Scalar<DataVector>& /*conformal_ricci_scalar*/,
      const Scalar<DataVector>& /*conformal_factor*/) noexcept {}
};

template <>
struct Sources<Equations::HamiltonianAndLapse, Geometry::FlatCartesian> {
  using argument_tags = tmpl::list<
      gr::Tags::EnergyDensity<DataVector>, gr::Tags::StressTrace<DataVector>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>,
      ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataVector>>,
      Tags::LongitudinalShiftMinusDtConformalMetricSquare<DataVector>,
      Tags::ShiftDotDerivExtrinsicCurvatureTrace<DataVector>>;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
      const gsl::not_null<Scalar<DataVector>*> lapse_equation,
      const Scalar<DataVector>& energy_density,
      const Scalar<DataVector>& stress_trace,
      const Scalar<DataVector>& extrinsic_curvature_trace,
      const Scalar<DataVector>& dt_extrinsic_curvature_trace,
      const Scalar<DataVector>&
          longitudinal_shift_minus_dt_conformal_metric_square,
      const Scalar<DataVector>& shift_dot_deriv_extrinsic_curvature_trace,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& lapse_times_conformal_factor,
      const tnsr::I<DataVector, 3>& /*conformal_factor_flux*/,
      const tnsr::I<DataVector, 3>&
      /*lapse_times_conformal_factor_flux*/) noexcept {
    add_hamiltonian_sources(hamiltonian_constraint, energy_density,
                            extrinsic_curvature_trace, conformal_factor);
    add_lapse_sources(lapse_equation, energy_density, stress_trace,
                      extrinsic_curvature_trace, dt_extrinsic_curvature_trace,
                      shift_dot_deriv_extrinsic_curvature_trace,
                      conformal_factor, lapse_times_conformal_factor);
    add_distortion_hamiltonian_and_lapse_sources(
        hamiltonian_constraint, lapse_equation,
        longitudinal_shift_minus_dt_conformal_metric_square, conformal_factor,
        lapse_times_conformal_factor);
  }
  static void apply(
      const gsl::not_null<tnsr::i<DataVector, 3>*>
      /*equation_for_conformal_factor_gradient*/,
      const gsl::not_null<tnsr::i<DataVector, 3>*>
      /*equation_for_lapse_times_conformal_factor_gradient*/,
      const Scalar<DataVector>& /*energy_density*/,
      const Scalar<DataVector>& /*stress_trace*/,
      const Scalar<DataVector>& /*extrinsic_curvature_trace*/,
      const Scalar<DataVector>& /*dt_extrinsic_curvature_trace*/,
      const Scalar<DataVector>&
      /*longitudinal_shift_minus_dt_conformal_metric_square*/,
      const Scalar<DataVector>& /*shift_dot_deriv_extrinsic_curvature_trace*/,
      const Scalar<DataVector>& /*conformal_factor*/,
      const Scalar<DataVector>& /*lapse_times_conformal_factor*/) noexcept {}
};

template <>
struct Sources<Equations::HamiltonianAndLapse, Geometry::Curved> {
  using argument_tags = tmpl::push_back<
      typename Sources<Equations::HamiltonianAndLapse,
                       Geometry::FlatCartesian>::argument_tags,
      Tags::ConformalChristoffelContracted<DataVector, 3, Frame::Inertial>,
      Tags::ConformalRicciScalar<DataVector>>;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
      const gsl::not_null<Scalar<DataVector>*> lapse_equation,
      const Scalar<DataVector>& energy_density,
      const Scalar<DataVector>& stress_trace,
      const Scalar<DataVector>& extrinsic_curvature_trace,
      const Scalar<DataVector>& dt_extrinsic_curvature_trace,
      const Scalar<DataVector>&
          longitudinal_shift_minus_dt_conformal_metric_square,
      const Scalar<DataVector>& shift_dot_deriv_extrinsic_curvature_trace,
      const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
      const Scalar<DataVector>& conformal_ricci_scalar,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& lapse_times_conformal_factor,
      const tnsr::I<DataVector, 3>& conformal_factor_flux,
      const tnsr::I<DataVector, 3>&
          lapse_times_conformal_factor_flux) noexcept {
    Sources<Equations::HamiltonianAndLapse, Geometry::FlatCartesian>::apply(
        hamiltonian_constraint, lapse_equation, energy_density, stress_trace,
        extrinsic_curvature_trace, dt_extrinsic_curvature_trace,
        longitudinal_shift_minus_dt_conformal_metric_square,
        shift_dot_deriv_extrinsic_curvature_trace, conformal_factor,
        lapse_times_conformal_factor, conformal_factor_flux,
        lapse_times_conformal_factor_flux);
    add_curved_hamiltonian_or_lapse_sources(
        hamiltonian_constraint, conformal_ricci_scalar, conformal_factor);
    Poisson::add_curved_sources(hamiltonian_constraint,
                                conformal_christoffel_contracted,
                                conformal_factor_flux);
    add_curved_hamiltonian_or_lapse_sources(
        lapse_equation, conformal_ricci_scalar, lapse_times_conformal_factor);
    Poisson::add_curved_sources(lapse_equation,
                                conformal_christoffel_contracted,
                                lapse_times_conformal_factor_flux);
  }
  static void apply(
      const gsl::not_null<tnsr::i<DataVector, 3>*>
      /*equation_for_conformal_factor_gradient*/,
      const gsl::not_null<tnsr::i<DataVector, 3>*>
      /*equation_for_lapse_times_conformal_factor_gradient*/,
      const Scalar<DataVector>& /*energy_density*/,
      const Scalar<DataVector>& /*stress_trace*/,
      const Scalar<DataVector>& /*extrinsic_curvature_trace*/,
      const Scalar<DataVector>& /*dt_extrinsic_curvature_trace*/,
      const Scalar<DataVector>&
      /*longitudinal_shift_minus_dt_conformal_metric_square*/,
      const Scalar<DataVector>& /*shift_dot_deriv_extrinsic_curvature_trace*/,
      const tnsr::i<DataVector, 3>& /*conformal_christoffel_contracted*/,
      const Scalar<DataVector>& /*conformal_ricci_scalar*/,
      const Scalar<DataVector>& /*conformal_factor*/,
      const Scalar<DataVector>& /*lapse_times_conformal_factor*/) noexcept {}
};

template <>
struct Sources<Equations::HamiltonianLapseAndShift, Geometry::FlatCartesian> {
  using argument_tags = tmpl::list<
      gr::Tags::EnergyDensity<DataVector>, gr::Tags::StressTrace<DataVector>,
      gr::Tags::MomentumDensity<3, Frame::Inertial, DataVector>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>,
      ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataVector>>,
      ::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataVector>,
                    tmpl::size_t<3>, Frame::Inertial>,
      Tags::ShiftBackground<DataVector, 3, Frame::Inertial>,
      Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<DataVector, 3,
                                                              Frame::Inertial>,
      ::Tags::div<Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
          DataVector, 3, Frame::Inertial>>>;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
      const gsl::not_null<Scalar<DataVector>*> lapse_equation,
      const gsl::not_null<tnsr::I<DataVector, 3>*> momentum_constraint,
      const Scalar<DataVector>& energy_density,
      const Scalar<DataVector>& stress_trace,
      const tnsr::I<DataVector, 3>& momentum_density,
      const Scalar<DataVector>& extrinsic_curvature_trace,
      const Scalar<DataVector>& dt_extrinsic_curvature_trace,
      const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,
      const tnsr::I<DataVector, 3>& shift_background,
      const tnsr::II<DataVector, 3>&
          longitudinal_shift_background_minus_dt_conformal_metric,
      const tnsr::I<DataVector, 3>&
          div_longitudinal_shift_background_minus_dt_conformal_metric,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& lapse_times_conformal_factor,
      const tnsr::I<DataVector, 3>& shift_excess,
      const tnsr::I<DataVector, 3>& conformal_factor_flux,
      const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,
      const tnsr::II<DataVector, 3>& longitudinal_shift_excess) noexcept {
    auto shift = shift_background;
    for (size_t i = 0; i < 3; ++i) {
      shift.get(i) += shift_excess.get(i);
    }
    auto longitudinal_shift_minus_dt_conformal_metric =
        longitudinal_shift_background_minus_dt_conformal_metric;
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j <= i; ++j) {
        longitudinal_shift_minus_dt_conformal_metric.get(i, j) +=
            longitudinal_shift_excess.get(i, j);
      }
    }
    add_hamiltonian_sources(hamiltonian_constraint, energy_density,
                            extrinsic_curvature_trace, conformal_factor);
    add_lapse_sources(lapse_equation, energy_density, stress_trace,
                      extrinsic_curvature_trace, dt_extrinsic_curvature_trace,
                      dot_product(shift, extrinsic_curvature_trace_gradient),
                      conformal_factor, lapse_times_conformal_factor);
    add_flat_cartesian_momentum_sources(
        hamiltonian_constraint, lapse_equation, momentum_constraint,
        momentum_density, extrinsic_curvature_trace_gradient,
        div_longitudinal_shift_background_minus_dt_conformal_metric,
        conformal_factor, lapse_times_conformal_factor, conformal_factor_flux,
        lapse_times_conformal_factor_flux,
        longitudinal_shift_minus_dt_conformal_metric);
  }
  static void apply(
      const gsl::not_null<tnsr::i<DataVector, 3>*>
      /*equation_for_conformal_factor_gradient*/,
      const gsl::not_null<tnsr::i<DataVector, 3>*>
      /*equation_for_lapse_times_conformal_factor_gradient*/,
      const gsl::not_null<tnsr::ii<DataVector, 3>*>
      /*equation_for_shift_strain*/,
      const Scalar<DataVector>& /*energy_density*/,
      const Scalar<DataVector>& /*stress_trace*/,
      const tnsr::I<DataVector, 3>& /*momentum_density*/,
      const Scalar<DataVector>& /*extrinsic_curvature_trace*/,
      const Scalar<DataVector>& /*dt_extrinsic_curvature_trace*/,
      const tnsr::i<DataVector, 3>& /*extrinsic_curvature_trace_gradient*/,
      const tnsr::I<DataVector, 3>& /*shift_background*/,
      const tnsr::II<DataVector, 3>&
      /*longitudinal_shift_background_minus_dt_conformal_metric*/,
      const tnsr::I<DataVector, 3>&
      /*div_longitudinal_shift_background_minus_dt_conformal_metric*/,
      const Scalar<DataVector>& /*conformal_factor*/,
      const Scalar<DataVector>& /*lapse_times_conformal_factor*/,
      const tnsr::I<DataVector, 3>& /*shift_excess*/) noexcept {}
};

template <>
struct Sources<Equations::HamiltonianLapseAndShift, Geometry::Curved> {
  using argument_tags = tmpl::push_back<
      typename Sources<Equations::HamiltonianLapseAndShift,
                       Geometry::FlatCartesian>::argument_tags,
      Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
      Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>,
      Tags::ConformalChristoffelFirstKind<DataVector, 3, Frame::Inertial>,
      Tags::ConformalChristoffelSecondKind<DataVector, 3, Frame::Inertial>,
      Tags::ConformalChristoffelContracted<DataVector, 3, Frame::Inertial>,
      Tags::ConformalRicciScalar<DataVector>>;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
      const gsl::not_null<Scalar<DataVector>*> lapse_equation,
      const gsl::not_null<tnsr::I<DataVector, 3>*> momentum_constraint,
      const Scalar<DataVector>& energy_density,
      const Scalar<DataVector>& stress_trace,
      const tnsr::I<DataVector, 3>& momentum_density,
      const Scalar<DataVector>& extrinsic_curvature_trace,
      const Scalar<DataVector>& dt_extrinsic_curvature_trace,
      const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,
      const tnsr::I<DataVector, 3>& shift_background,
      const tnsr::II<DataVector, 3>&
          longitudinal_shift_background_minus_dt_conformal_metric,
      const tnsr::I<DataVector, 3>&
          div_longitudinal_shift_background_minus_dt_conformal_metric,
      const tnsr::ii<DataVector, 3>& conformal_metric,
      const tnsr::II<DataVector, 3>& inv_conformal_metric,
      const tnsr::ijj<DataVector, 3>& /*conformal_christoffel_first_kind*/,
      const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind,
      const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
      const Scalar<DataVector>& conformal_ricci_scalar,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& lapse_times_conformal_factor,
      const tnsr::I<DataVector, 3>& shift_excess,
      const tnsr::I<DataVector, 3>& conformal_factor_flux,
      const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,
      const tnsr::II<DataVector, 3>& longitudinal_shift_excess) noexcept {
    auto shift = shift_background;
    for (size_t i = 0; i < 3; ++i) {
      shift.get(i) += shift_excess.get(i);
    }
    auto longitudinal_shift_minus_dt_conformal_metric =
        longitudinal_shift_background_minus_dt_conformal_metric;
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j <= i; ++j) {
        longitudinal_shift_minus_dt_conformal_metric.get(i, j) +=
            longitudinal_shift_excess.get(i, j);
      }
    }
    add_hamiltonian_sources(hamiltonian_constraint, energy_density,
                            extrinsic_curvature_trace, conformal_factor);
    add_curved_hamiltonian_or_lapse_sources(
        hamiltonian_constraint, conformal_ricci_scalar, conformal_factor);
    Poisson::add_curved_sources(hamiltonian_constraint,
                                conformal_christoffel_contracted,
                                conformal_factor_flux);
    add_lapse_sources(lapse_equation, energy_density, stress_trace,
                      extrinsic_curvature_trace, dt_extrinsic_curvature_trace,
                      dot_product(shift, extrinsic_curvature_trace_gradient),
                      conformal_factor, lapse_times_conformal_factor);
    add_curved_hamiltonian_or_lapse_sources(
        lapse_equation, conformal_ricci_scalar, lapse_times_conformal_factor);
    Poisson::add_curved_sources(lapse_equation,
                                conformal_christoffel_contracted,
                                lapse_times_conformal_factor_flux);
    add_curved_momentum_sources(
        hamiltonian_constraint, lapse_equation, momentum_constraint,
        momentum_density, extrinsic_curvature_trace_gradient, conformal_metric,
        inv_conformal_metric,
        div_longitudinal_shift_background_minus_dt_conformal_metric,
        conformal_factor, lapse_times_conformal_factor, conformal_factor_flux,
        lapse_times_conformal_factor_flux,
        longitudinal_shift_minus_dt_conformal_metric);
    Elasticity::add_curved_sources(
        momentum_constraint, conformal_christoffel_second_kind,
        conformal_christoffel_contracted,
        longitudinal_shift_minus_dt_conformal_metric);
  }
  static void apply(
      const gsl::not_null<tnsr::i<DataVector, 3>*>
      /*equation_for_conformal_factor_gradient*/,
      const gsl::not_null<tnsr::i<DataVector, 3>*>
      /*equation_for_lapse_times_conformal_factor_gradient*/,
      const gsl::not_null<tnsr::ii<DataVector, 3>*> equation_for_shift_strain,
      const Scalar<DataVector>& /*energy_density*/,
      const Scalar<DataVector>& /*stress_trace*/,
      const tnsr::I<DataVector, 3>& /*momentum_density*/,
      const Scalar<DataVector>& /*extrinsic_curvature_trace*/,
      const Scalar<DataVector>& /*dt_extrinsic_curvature_trace*/,
      const tnsr::i<DataVector, 3>& /*extrinsic_curvature_trace_gradient*/,
      const tnsr::I<DataVector, 3>& /*shift_background*/,
      const tnsr::II<DataVector, 3>&
      /*longitudinal_shift_background_minus_dt_conformal_metric*/,
      const tnsr::I<DataVector, 3>&
      /*div_longitudinal_shift_background_minus_dt_conformal_metric*/,
      const tnsr::ii<DataVector, 3>& /*conformal_metric*/,
      const tnsr::II<DataVector, 3>& /*inv_conformal_metric*/,
      const tnsr::ijj<DataVector, 3>& conformal_christoffel_first_kind,
      const tnsr::Ijj<DataVector, 3>& /*conformal_christoffel_second_kind*/,
      const tnsr::i<DataVector, 3>& /*conformal_christoffel_contracted*/,
      const Scalar<DataVector>& /*conformal_ricci_scalar*/,
      const Scalar<DataVector>& /*conformal_factor*/,
      const Scalar<DataVector>& /*lapse_times_conformal_factor*/,
      const tnsr::I<DataVector, 3>& shift_excess) noexcept {
    Elasticity::add_curved_auxiliary_sources(equation_for_shift_strain,
                                             conformal_christoffel_first_kind,
                                             shift_excess);
  }
};
/// \endcond

/// The linearization of the sources \f$S\f$ for the first-order formulation of
/// the XCTS equations.
///
/// \see Xcts::FirstOrderSystem
template <Equations EnabledEquations, Geometry ConformalGeometry>
struct LinearizedSources;

/// \cond
template <>
struct LinearizedSources<Equations::Hamiltonian, Geometry::FlatCartesian> {
  using argument_tags =
      tmpl::push_back<typename Sources<Equations::Hamiltonian,
                                       Geometry::FlatCartesian>::argument_tags,
                      Tags::ConformalFactor<DataVector>>;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*>
          linearized_hamiltonian_constraint,
      const Scalar<DataVector>& energy_density,
      const Scalar<DataVector>& extrinsic_curvature_trace,
      const Scalar<DataVector>&
          longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& conformal_factor_correction,
      const tnsr::I<DataVector, 3>&
      /*conformal_factor_flux_correction*/) noexcept {
    add_linearized_hamiltonian_sources(
        linearized_hamiltonian_constraint, energy_density,
        extrinsic_curvature_trace, conformal_factor,
        conformal_factor_correction);
    add_linearized_distortion_hamiltonian_sources(
        linearized_hamiltonian_constraint,
        longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
        conformal_factor, conformal_factor_correction);
  }
  static void apply(
      const gsl::not_null<tnsr::i<DataVector, 3>*>
      /*source_for_conformal_factor_gradient_correction*/,
      const Scalar<DataVector>& /*energy_density*/,
      const Scalar<DataVector>& /*extrinsic_curvature_trace*/,
      const Scalar<DataVector>&
      /*longitudinal_shift_minus_dt_conformal_metric_over_lapse_square*/,
      const Scalar<DataVector>& /*conformal_factor*/,
      const Scalar<DataVector>& /*conformal_factor_correction*/) noexcept {}
};

template <>
struct LinearizedSources<Equations::Hamiltonian, Geometry::Curved> {
  using argument_tags = tmpl::push_back<
      typename Sources<Equations::Hamiltonian, Geometry::Curved>::argument_tags,
      Tags::ConformalFactor<DataVector>>;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*>
          linearized_hamiltonian_constraint,
      const Scalar<DataVector>& energy_density,
      const Scalar<DataVector>& extrinsic_curvature_trace,
      const Scalar<DataVector>&
          longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
      const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
      const Scalar<DataVector>& conformal_ricci_scalar,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& conformal_factor_correction,
      const tnsr::I<DataVector, 3>& conformal_factor_flux_correction) noexcept {
    LinearizedSources<Equations::Hamiltonian, Geometry::FlatCartesian>::apply(
        linearized_hamiltonian_constraint, energy_density,
        extrinsic_curvature_trace,
        longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
        conformal_factor, conformal_factor_correction,
        conformal_factor_flux_correction);
    add_curved_hamiltonian_or_lapse_sources(linearized_hamiltonian_constraint,
                                            conformal_ricci_scalar,
                                            conformal_factor_correction);
    Poisson::add_curved_sources(linearized_hamiltonian_constraint,
                                conformal_christoffel_contracted,
                                conformal_factor_flux_correction);
  }
  static void apply(
      const gsl::not_null<tnsr::i<DataVector, 3>*>
      /*equation_for_conformal_factor_gradient_correction*/,
      const Scalar<DataVector>& /*energy_density*/,
      const Scalar<DataVector>& /*extrinsic_curvature_trace*/,
      const Scalar<DataVector>&
      /*longitudinal_shift_minus_dt_conformal_metric_over_lapse_square*/,
      const tnsr::i<DataVector, 3>& /*conformal_christoffel_contracted*/,
      const Scalar<DataVector>& /*conformal_ricci_scalar*/,
      const Scalar<DataVector>& /*conformal_factor*/,
      const Scalar<DataVector>& /*conformal_factor_correction*/) noexcept {}
};

template <>
struct LinearizedSources<Equations::HamiltonianAndLapse,
                         Geometry::FlatCartesian> {
  using argument_tags =
      tmpl::push_back<typename Sources<Equations::HamiltonianAndLapse,
                                       Geometry::FlatCartesian>::argument_tags,
                      Tags::ConformalFactor<DataVector>,
                      Tags::LapseTimesConformalFactor<DataVector>>;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*>
          linearized_hamiltonian_constraint,
      const gsl::not_null<Scalar<DataVector>*> linearized_lapse_equation,
      const Scalar<DataVector>& energy_density,
      const Scalar<DataVector>& stress_trace,
      const Scalar<DataVector>& extrinsic_curvature_trace,
      const Scalar<DataVector>& dt_extrinsic_curvature_trace,
      const Scalar<DataVector>&
          longitudinal_shift_minus_dt_conformal_metric_square,
      const Scalar<DataVector>& shift_dot_deriv_extrinsic_curvature_trace,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& lapse_times_conformal_factor,
      const Scalar<DataVector>& conformal_factor_correction,
      const Scalar<DataVector>& lapse_times_conformal_factor_correction,
      const tnsr::I<DataVector, 3>& /*conformal_factor_flux_correction*/,
      const tnsr::I<DataVector, 3>&
      /*lapse_times_conformal_factor_flux_correction*/) noexcept {
    add_linearized_hamiltonian_sources(
        linearized_hamiltonian_constraint, energy_density,
        extrinsic_curvature_trace, conformal_factor,
        conformal_factor_correction);
    add_linearized_lapse_sources(
        linearized_lapse_equation, energy_density, stress_trace,
        extrinsic_curvature_trace, dt_extrinsic_curvature_trace,
        shift_dot_deriv_extrinsic_curvature_trace, conformal_factor,
        lapse_times_conformal_factor, conformal_factor_correction,
        lapse_times_conformal_factor_correction);
    add_linearized_distortion_hamiltonian_and_lapse_sources(
        linearized_hamiltonian_constraint, linearized_lapse_equation,
        longitudinal_shift_minus_dt_conformal_metric_square, conformal_factor,
        lapse_times_conformal_factor, conformal_factor_correction,
        lapse_times_conformal_factor_correction);
  }
  static void apply(
      const gsl::not_null<tnsr::i<DataVector, 3>*>
      /*equation_for_conformal_factor_gradient_correction*/,
      const gsl::not_null<tnsr::i<DataVector, 3>*>
      /*equation_for_lapse_times_conformal_factor_gradient_correction*/,
      const Scalar<DataVector>& /*energy_density*/,
      const Scalar<DataVector>& /*stress_trace*/,
      const Scalar<DataVector>& /*extrinsic_curvature_trace*/,
      const Scalar<DataVector>& /*dt_extrinsic_curvature_trace*/,
      const Scalar<DataVector>&
      /*longitudinal_shift_minus_dt_conformal_metric_square*/,
      const Scalar<DataVector>& /*shift_dot_deriv_extrinsic_curvature_trace*/,
      const Scalar<DataVector>& /*conformal_factor*/,
      const Scalar<DataVector>& /*lapse_times_conformal_factor*/,
      const Scalar<DataVector>& /*conformal_factor_correction*/,
      const Scalar<
          DataVector>& /*lapse_times_conformal_factor_correction*/) noexcept {}
};

template <>
struct LinearizedSources<Equations::HamiltonianAndLapse, Geometry::Curved> {
  using argument_tags =
      tmpl::push_back<typename Sources<Equations::HamiltonianAndLapse,
                                       Geometry::Curved>::argument_tags,
                      Tags::ConformalFactor<DataVector>,
                      Tags::LapseTimesConformalFactor<DataVector>>;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*>
          linearized_hamiltonian_constraint,
      const gsl::not_null<Scalar<DataVector>*> linearized_lapse_equation,
      const Scalar<DataVector>& energy_density,
      const Scalar<DataVector>& stress_trace,
      const Scalar<DataVector>& extrinsic_curvature_trace,
      const Scalar<DataVector>& dt_extrinsic_curvature_trace,
      const Scalar<DataVector>&
          longitudinal_shift_minus_dt_conformal_metric_square,
      const Scalar<DataVector>& shift_dot_deriv_extrinsic_curvature_trace,
      const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
      const Scalar<DataVector>& conformal_ricci_scalar,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& lapse_times_conformal_factor,
      const Scalar<DataVector>& conformal_factor_correction,
      const Scalar<DataVector>& lapse_times_conformal_factor_correction,
      const tnsr::I<DataVector, 3>& conformal_factor_flux_correction,
      const tnsr::I<DataVector, 3>&
          lapse_times_conformal_factor_flux_correction) noexcept {
    LinearizedSources<Equations::HamiltonianAndLapse, Geometry::FlatCartesian>::
        apply(linearized_hamiltonian_constraint, linearized_lapse_equation,
              energy_density, stress_trace, extrinsic_curvature_trace,
              dt_extrinsic_curvature_trace,
              longitudinal_shift_minus_dt_conformal_metric_square,
              shift_dot_deriv_extrinsic_curvature_trace, conformal_factor,
              lapse_times_conformal_factor, conformal_factor_correction,
              lapse_times_conformal_factor_correction,
              conformal_factor_flux_correction,
              lapse_times_conformal_factor_flux_correction);
    add_curved_hamiltonian_or_lapse_sources(linearized_hamiltonian_constraint,
                                            conformal_ricci_scalar,
                                            conformal_factor_correction);
    Poisson::add_curved_sources(linearized_hamiltonian_constraint,
                                conformal_christoffel_contracted,
                                conformal_factor_flux_correction);
    add_curved_hamiltonian_or_lapse_sources(
        linearized_lapse_equation, conformal_ricci_scalar,
        lapse_times_conformal_factor_correction);
    Poisson::add_curved_sources(linearized_lapse_equation,
                                conformal_christoffel_contracted,
                                lapse_times_conformal_factor_flux_correction);
  }
  static void apply(
      const gsl::not_null<tnsr::i<DataVector, 3>*>
      /*equation_for_conformal_factor_gradient_correction*/,
      const gsl::not_null<tnsr::i<DataVector, 3>*>
      /*equation_for_lapse_times_conformal_factor_gradient_correction*/,
      const Scalar<DataVector>& /*energy_density*/,
      const Scalar<DataVector>& /*stress_trace*/,
      const Scalar<DataVector>& /*extrinsic_curvature_trace*/,
      const Scalar<DataVector>& /*dt_extrinsic_curvature_trace*/,
      const Scalar<DataVector>&
      /*longitudinal_shift_minus_dt_conformal_metric_square*/,
      const Scalar<DataVector>& /*shift_dot_deriv_extrinsic_curvature_trace*/,
      const tnsr::i<DataVector, 3>& /*conformal_christoffel_contracted*/,
      const Scalar<DataVector>& /*conformal_ricci_scalar*/,
      const Scalar<DataVector>& /*conformal_factor*/,
      const Scalar<DataVector>& /*lapse_times_conformal_factor*/,
      const Scalar<DataVector>& /*conformal_factor_correction*/,
      const Scalar<
          DataVector>& /*lapse_times_conformal_factor_correction*/) noexcept {}
};

template <>
struct LinearizedSources<Equations::HamiltonianLapseAndShift,
                         Geometry::FlatCartesian> {
  using argument_tags = tmpl::push_back<
      typename Sources<Equations::HamiltonianLapseAndShift,
                       Geometry::FlatCartesian>::argument_tags,
      Tags::ConformalFactor<DataVector>,
      Tags::LapseTimesConformalFactor<DataVector>,
      Tags::ShiftExcess<DataVector, 3, Frame::Inertial>,
      ::Tags::Flux<Tags::ConformalFactor<DataVector>, tmpl::size_t<3>,
                   Frame::Inertial>,
      ::Tags::Flux<Tags::LapseTimesConformalFactor<DataVector>, tmpl::size_t<3>,
                   Frame::Inertial>,
      Tags::LongitudinalShiftExcess<DataVector, 3, Frame::Inertial>>;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*>
          linearized_hamiltonian_constraint,
      const gsl::not_null<Scalar<DataVector>*> linearized_lapse_equation,
      const gsl::not_null<tnsr::I<DataVector, 3>*>
          linearized_momentum_constraint,
      const Scalar<DataVector>& energy_density,
      const Scalar<DataVector>& stress_trace,
      const tnsr::I<DataVector, 3>& momentum_density,
      const Scalar<DataVector>& extrinsic_curvature_trace,
      const Scalar<DataVector>& dt_extrinsic_curvature_trace,
      const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,
      const tnsr::I<DataVector, 3>& shift_background,
      const tnsr::II<DataVector, 3>&
          longitudinal_shift_background_minus_dt_conformal_metric,
      const tnsr::I<DataVector, 3>&
      /*div_longitudinal_shift_background_minus_dt_conformal_metric*/,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& lapse_times_conformal_factor,
      const tnsr::I<DataVector, 3>& shift_excess,
      const tnsr::I<DataVector, 3>& conformal_factor_flux,
      const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,
      const tnsr::II<DataVector, 3>& longitudinal_shift_excess,
      const Scalar<DataVector>& conformal_factor_correction,
      const Scalar<DataVector>& lapse_times_conformal_factor_correction,
      const tnsr::I<DataVector, 3>& shift_excess_correction,
      const tnsr::I<DataVector, 3>& conformal_factor_flux_correction,
      const tnsr::I<DataVector, 3>&
          lapse_times_conformal_factor_flux_correction,
      const tnsr::II<DataVector, 3>&
          longitudinal_shift_excess_correction) noexcept {
    auto shift = shift_background;
    for (size_t i = 0; i < 3; ++i) {
      shift.get(i) += shift_excess.get(i);
    }
    auto longitudinal_shift_minus_dt_conformal_metric =
        longitudinal_shift_background_minus_dt_conformal_metric;
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j <= i; ++j) {
        longitudinal_shift_minus_dt_conformal_metric.get(i, j) +=
            longitudinal_shift_excess.get(i, j);
      }
    }
    add_linearized_hamiltonian_sources(
        linearized_hamiltonian_constraint, energy_density,
        extrinsic_curvature_trace, conformal_factor,
        conformal_factor_correction);
    add_linearized_lapse_sources(
        linearized_lapse_equation, energy_density, stress_trace,
        extrinsic_curvature_trace, dt_extrinsic_curvature_trace,
        dot_product(shift, extrinsic_curvature_trace_gradient),
        conformal_factor, lapse_times_conformal_factor,
        conformal_factor_correction, lapse_times_conformal_factor_correction);
    add_flat_cartesian_linearized_momentum_sources(
        linearized_hamiltonian_constraint, linearized_lapse_equation,
        linearized_momentum_constraint, momentum_density,
        extrinsic_curvature_trace_gradient, conformal_factor,
        lapse_times_conformal_factor, conformal_factor_flux,
        lapse_times_conformal_factor_flux,
        longitudinal_shift_minus_dt_conformal_metric,
        conformal_factor_correction, lapse_times_conformal_factor_correction,
        shift_excess_correction, conformal_factor_flux_correction,
        lapse_times_conformal_factor_flux_correction,
        longitudinal_shift_excess_correction);
  }
  static void apply(
      const gsl::not_null<tnsr::i<DataVector, 3>*>
      /*equation_for_conformal_factor_gradient_correction*/,
      const gsl::not_null<tnsr::i<DataVector, 3>*>
      /*equation_for_lapse_times_conformal_factor_gradient_correction*/,
      const gsl::not_null<tnsr::ii<DataVector, 3>*>
      /*equation_for_shift_strain_correction*/,
      const Scalar<DataVector>& /*energy_density*/,
      const Scalar<DataVector>& /*stress_trace*/,
      const tnsr::I<DataVector, 3>& /*momentum_density*/,
      const Scalar<DataVector>& /*extrinsic_curvature_trace*/,
      const Scalar<DataVector>& /*dt_extrinsic_curvature_trace*/,
      const tnsr::i<DataVector, 3>& /*extrinsic_curvature_trace_gradient*/,
      const tnsr::I<DataVector, 3>& /*shift_background*/,
      const tnsr::II<DataVector, 3>&
      /*longitudinal_shift_background_minus_dt_conformal_metric*/,
      const tnsr::I<DataVector, 3>&
      /*div_longitudinal_shift_background_minus_dt_conformal_metric*/,
      const Scalar<DataVector>& /*conformal_factor*/,
      const Scalar<DataVector>& /*lapse_times_conformal_factor*/,
      const tnsr::I<DataVector, 3>& /*shift_excess*/,
      const tnsr::I<DataVector, 3>& /*conformal_factor_flux*/,
      const tnsr::I<DataVector, 3>& /*lapse_times_conformal_factor_flux*/,
      const tnsr::II<DataVector, 3>& /*longitudinal_shift_excess*/,
      const Scalar<DataVector>& /*conformal_factor_correction*/,
      const Scalar<DataVector>& /*lapse_times_conformal_factor_correction*/,
      const tnsr::I<DataVector, 3>& /*shift_excess_correction*/) noexcept {}
};

template <>
struct LinearizedSources<Equations::HamiltonianLapseAndShift,
                         Geometry::Curved> {
  using argument_tags = tmpl::push_back<
      typename Sources<Equations::HamiltonianLapseAndShift,
                       Geometry::Curved>::argument_tags,
      Tags::ConformalFactor<DataVector>,
      Tags::LapseTimesConformalFactor<DataVector>,
      Tags::ShiftExcess<DataVector, 3, Frame::Inertial>,
      ::Tags::Flux<Tags::ConformalFactor<DataVector>, tmpl::size_t<3>,
                   Frame::Inertial>,
      ::Tags::Flux<Tags::LapseTimesConformalFactor<DataVector>, tmpl::size_t<3>,
                   Frame::Inertial>,
      Tags::LongitudinalShiftExcess<DataVector, 3, Frame::Inertial>>;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*>
          linearized_hamiltonian_constraint,
      const gsl::not_null<Scalar<DataVector>*> linearized_lapse_equation,
      const gsl::not_null<tnsr::I<DataVector, 3>*>
          linearized_momentum_constraint,
      const Scalar<DataVector>& energy_density,
      const Scalar<DataVector>& stress_trace,
      const tnsr::I<DataVector, 3>& momentum_density,
      const Scalar<DataVector>& extrinsic_curvature_trace,
      const Scalar<DataVector>& dt_extrinsic_curvature_trace,
      const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,
      const tnsr::I<DataVector, 3>& shift_background,
      const tnsr::II<DataVector, 3>&
          longitudinal_shift_background_minus_dt_conformal_metric,
      const tnsr::I<DataVector, 3>&
      /*div_longitudinal_shift_background_minus_dt_conformal_metric*/,
      const tnsr::ii<DataVector, 3>& conformal_metric,
      const tnsr::II<DataVector, 3>& inv_conformal_metric,
      const tnsr::ijj<DataVector, 3>& /*conformal_christoffel_first_kind*/,
      const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind,
      const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
      const Scalar<DataVector>& conformal_ricci_scalar,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& lapse_times_conformal_factor,
      const tnsr::I<DataVector, 3>& shift_excess,
      const tnsr::I<DataVector, 3>& conformal_factor_flux,
      const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,
      const tnsr::II<DataVector, 3>& longitudinal_shift_excess,
      const Scalar<DataVector>& conformal_factor_correction,
      const Scalar<DataVector>& lapse_times_conformal_factor_correction,
      const tnsr::I<DataVector, 3>& shift_excess_correction,
      const tnsr::I<DataVector, 3>& conformal_factor_flux_correction,
      const tnsr::I<DataVector, 3>&
          lapse_times_conformal_factor_flux_correction,
      const tnsr::II<DataVector, 3>&
          longitudinal_shift_excess_correction) noexcept {
    auto shift = shift_background;
    for (size_t i = 0; i < 3; ++i) {
      shift.get(i) += shift_excess.get(i);
    }
    auto longitudinal_shift_minus_dt_conformal_metric =
        longitudinal_shift_background_minus_dt_conformal_metric;
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j <= i; ++j) {
        longitudinal_shift_minus_dt_conformal_metric.get(i, j) +=
            longitudinal_shift_excess.get(i, j);
      }
    }
    add_linearized_hamiltonian_sources(
        linearized_hamiltonian_constraint, energy_density,
        extrinsic_curvature_trace, conformal_factor,
        conformal_factor_correction);
    add_curved_hamiltonian_or_lapse_sources(linearized_hamiltonian_constraint,
                                            conformal_ricci_scalar,
                                            conformal_factor_correction);
    Poisson::add_curved_sources(linearized_hamiltonian_constraint,
                                conformal_christoffel_contracted,
                                conformal_factor_flux_correction);
    add_linearized_lapse_sources(
        linearized_lapse_equation, energy_density, stress_trace,
        extrinsic_curvature_trace, dt_extrinsic_curvature_trace,
        dot_product(shift, extrinsic_curvature_trace_gradient),
        conformal_factor, lapse_times_conformal_factor,
        conformal_factor_correction, lapse_times_conformal_factor_correction);
    add_curved_hamiltonian_or_lapse_sources(
        linearized_lapse_equation, conformal_ricci_scalar,
        lapse_times_conformal_factor_correction);
    Poisson::add_curved_sources(linearized_lapse_equation,
                                conformal_christoffel_contracted,
                                lapse_times_conformal_factor_flux_correction);
    add_curved_linearized_momentum_sources(
        linearized_hamiltonian_constraint, linearized_lapse_equation,
        linearized_momentum_constraint, momentum_density,
        extrinsic_curvature_trace_gradient, conformal_metric,
        inv_conformal_metric, conformal_factor, lapse_times_conformal_factor,
        conformal_factor_flux, lapse_times_conformal_factor_flux,
        longitudinal_shift_minus_dt_conformal_metric,
        conformal_factor_correction, lapse_times_conformal_factor_correction,
        shift_excess_correction, conformal_factor_flux_correction,
        lapse_times_conformal_factor_flux_correction,
        longitudinal_shift_excess_correction);
    Elasticity::add_curved_sources(
        linearized_momentum_constraint, conformal_christoffel_second_kind,
        conformal_christoffel_contracted, longitudinal_shift_excess_correction);
  }
  static void apply(
      const gsl::not_null<tnsr::i<DataVector, 3>*>
      /*equation_for_conformal_factor_gradient_correction*/,
      const gsl::not_null<tnsr::i<DataVector, 3>*>
      /*equation_for_lapse_times_conformal_factor_gradient_correction*/,
      const gsl::not_null<tnsr::ii<DataVector, 3>*>
          equation_for_shift_strain_correction,
      const Scalar<DataVector>& /*energy_density*/,
      const Scalar<DataVector>& /*stress_trace*/,
      const tnsr::I<DataVector, 3>& /*momentum_density*/,
      const Scalar<DataVector>& /*extrinsic_curvature_trace*/,
      const Scalar<DataVector>& /*dt_extrinsic_curvature_trace*/,
      const tnsr::i<DataVector, 3>& /*extrinsic_curvature_trace_gradient*/,
      const tnsr::I<DataVector, 3>& /*shift_background*/,
      const tnsr::II<DataVector, 3>&
      /*longitudinal_shift_background_minus_dt_conformal_metric*/,
      const tnsr::I<DataVector, 3>&
      /*div_longitudinal_shift_background_minus_dt_conformal_metric*/,
      const tnsr::ii<DataVector, 3>& /*conformal_metric*/,
      const tnsr::II<DataVector, 3>& /*inv_conformal_metric*/,
      const tnsr::ijj<DataVector, 3>& conformal_christoffel_first_kind,
      const tnsr::Ijj<DataVector, 3>& /*conformal_christoffel_second_kind*/,
      const tnsr::i<DataVector, 3>& /*conformal_christoffel_contracted*/,
      const Scalar<DataVector>& /*conformal_ricci_scalar*/,
      const Scalar<DataVector>& /*conformal_factor*/,
      const Scalar<DataVector>& /*lapse_times_conformal_factor*/,
      const tnsr::I<DataVector, 3>& /*shift_excess*/,
      const tnsr::I<DataVector, 3>& /*conformal_factor_flux*/,
      const tnsr::I<DataVector, 3>& /*lapse_times_conformal_factor_flux*/,
      const tnsr::II<DataVector, 3>& /*longitudinal_shift_excess*/,
      const Scalar<DataVector>& /*conformal_factor_correction*/,
      const Scalar<DataVector>& /*lapse_times_conformal_factor_correction*/,
      const tnsr::I<DataVector, 3>& shift_excess_correction) noexcept {
    Elasticity::add_curved_auxiliary_sources(
        equation_for_shift_strain_correction, conformal_christoffel_first_kind,
        shift_excess_correction);
  }
};
/// \endcond

}  // namespace Xcts
