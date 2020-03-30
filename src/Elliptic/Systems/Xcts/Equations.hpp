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

/*!
 * \brief The conformal longitudinal shift \f$\left(\bar{L}\beta\right)^{ij}\f$
 *
 * Computes the conformal longitudinal shift
 *
 * \f{equation}
 * (\bar{L}\beta)^{ij} = 2\left(\bar{gamma}^{ik}\bar{gamma}^{jl} - \frac{1}{3}
 * \bar{gamma}^{jk}\bar{gamma}^{kl}\right) B_{kl}
 * \f}
 *
 * where \f$B_{ij}=\left(\partial_{(i}\bar{\gamma}_{j)k} -
 * \bar{\Gamma}_{kij}\right)\beta^k\f$ is the
 * `Xcts::Tags::ShiftStrain`.
 *
 * \see `Xcts::FirstOrderSystem`
 */
template <typename DataType, typename Symm>
void longitudinal_shift(
    // We provide this function for a symmetric or non-symmetric result tensor
    // because fluxes currently can't be symmetric in their first index. Making
    // them symmetric would require changes to the way flux tags are constructed
    // Once fluxes can be symmetric this function should only allow a symmetric
    // result tensor.
    gsl::not_null<
        Tensor<DataType, Symm,
               index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                          SpatialIndex<3, UpLo::Up, Frame::Inertial>>>*>
        result,
    const tnsr::II<DataType, 3>& inv_conformal_metric,
    const tnsr::ii<DataType, 3>& shift_strain) noexcept;

/*!
 * \brief The conformal longitudinal shift \f$\left(\bar{L}\beta\right)^{ij}\f$
 * on a Euclidean conformal metric \f$\bar{\gamma}_{ij}=\delta_{ij}\f$
 *
 * \see `Xcts::longitudinal_shift`
 */
template <typename DataType, typename Symm>
void euclidean_longitudinal_shift(
    gsl::not_null<
        Tensor<DataType, Symm,
               index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                          SpatialIndex<3, UpLo::Up, Frame::Inertial>>>*>
        result,
    const tnsr::ii<DataType, 3>& shift_strain) noexcept;

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
}

/// The fluxes \f$F^i\f$ for the first-order formulation of the XCTS equations.
///
/// \see Xcts::FirstOrderSystem
template <Equations EnabledEquations, Geometry ConformalGeometry>
struct Fluxes;

/// \cond
template <>
struct Fluxes<Equations::Hamiltonian, Geometry::Euclidean> {
  using argument_tags = tmpl::list<>;
  static void apply(
      const gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_conformal_factor,
      const tnsr::i<DataVector, 3>& conformal_factor_gradient) noexcept {
    Poisson::euclidean_fluxes(flux_for_conformal_factor,
                              conformal_factor_gradient);
  }
  static void apply(const gsl::not_null<tnsr::Ij<DataVector, 3>*>
                        flux_for_conformal_factor_gradient,
                    const Scalar<DataVector>& conformal_factor) noexcept {
    Poisson::auxiliary_fluxes(flux_for_conformal_factor_gradient,
                              conformal_factor);
  }
  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

template <>
struct Fluxes<Equations::Hamiltonian, Geometry::NonEuclidean> {
  using argument_tags =
      tmpl::list<Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>>;
  static void apply(
      const gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_conformal_factor,
      const tnsr::II<DataVector, 3>& inv_conformal_metric,
      const tnsr::i<DataVector, 3>& conformal_factor_gradient) noexcept {
    Poisson::non_euclidean_fluxes(flux_for_conformal_factor,
                                  inv_conformal_metric,
                                  conformal_factor_gradient);
  }
  static void apply(const gsl::not_null<tnsr::Ij<DataVector, 3>*>
                        flux_for_conformal_factor_gradient,
                    const tnsr::II<DataVector, 3>& /*inv_conformal_metric*/,
                    const Scalar<DataVector>& conformal_factor) noexcept {
    Poisson::auxiliary_fluxes(flux_for_conformal_factor_gradient,
                              conformal_factor);
  }
  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

template <>
struct Fluxes<Equations::HamiltonianAndLapse, Geometry::Euclidean> {
  using argument_tags = tmpl::list<>;
  static void apply(
      const gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_conformal_factor,
      const gsl::not_null<tnsr::I<DataVector, 3>*>
          flux_for_lapse_times_conformal_factor,
      const tnsr::i<DataVector, 3>& conformal_factor_gradient,
      const tnsr::i<DataVector, 3>&
          lapse_times_conformal_factor_gradient) noexcept {
    Fluxes<Equations::Hamiltonian, Geometry::Euclidean>::apply(
        flux_for_conformal_factor, conformal_factor_gradient);
    Poisson::euclidean_fluxes(flux_for_lapse_times_conformal_factor,
                              lapse_times_conformal_factor_gradient);
  }
  static void apply(
      const gsl::not_null<tnsr::Ij<DataVector, 3>*>
          flux_for_conformal_factor_gradient,
      const gsl::not_null<tnsr::Ij<DataVector, 3>*>
          flux_for_lapse_times_conformal_factor_gradient,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& lapse_times_conformal_factor) noexcept {
    Fluxes<Equations::Hamiltonian, Geometry::Euclidean>::apply(
        flux_for_conformal_factor_gradient, conformal_factor);
    Poisson::auxiliary_fluxes(flux_for_lapse_times_conformal_factor_gradient,
                              lapse_times_conformal_factor);
  }
  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

template <>
struct Fluxes<Equations::HamiltonianAndLapse, Geometry::NonEuclidean> {
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
    Fluxes<Equations::Hamiltonian, Geometry::NonEuclidean>::apply(
        flux_for_conformal_factor, inv_conformal_metric,
        conformal_factor_gradient);
    Poisson::non_euclidean_fluxes(flux_for_lapse_times_conformal_factor,
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
    Fluxes<Equations::Hamiltonian, Geometry::NonEuclidean>::apply(
        flux_for_conformal_factor_gradient, inv_conformal_metric,
        conformal_factor);
    Poisson::auxiliary_fluxes(flux_for_lapse_times_conformal_factor_gradient,
                              lapse_times_conformal_factor);
  }
  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

template <>
struct Fluxes<Equations::HamiltonianLapseAndShift, Geometry::Euclidean> {
  using argument_tags = tmpl::list<>;
  static void apply(
      const gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_conformal_factor,
      const gsl::not_null<tnsr::I<DataVector, 3>*>
          flux_for_lapse_times_conformal_factor,
      const gsl::not_null<tnsr::IJ<DataVector, 3>*> flux_for_shift,
      const tnsr::i<DataVector, 3>& conformal_factor_gradient,
      const tnsr::i<DataVector, 3>& lapse_times_conformal_factor_gradient,
      const tnsr::ii<DataVector, 3>& shift_strain) noexcept {
    Fluxes<Equations::HamiltonianAndLapse, Geometry::Euclidean>::apply(
        flux_for_conformal_factor, flux_for_lapse_times_conformal_factor,
        conformal_factor_gradient, lapse_times_conformal_factor_gradient);
    Xcts::euclidean_longitudinal_shift(flux_for_shift, shift_strain);
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
    Fluxes<Equations::HamiltonianAndLapse, Geometry::Euclidean>::apply(
        flux_for_conformal_factor_gradient,
        flux_for_lapse_times_conformal_factor_gradient, conformal_factor,
        lapse_times_conformal_factor);
    Elasticity::auxiliary_fluxes(flux_for_shift_strain, shift);
  }
  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

template <>
struct Fluxes<Equations::HamiltonianLapseAndShift, Geometry::NonEuclidean> {
  using argument_tags =
      tmpl::list<Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
                 Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>>;
  static void apply(
      const gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_conformal_factor,
      const gsl::not_null<tnsr::I<DataVector, 3>*>
          flux_for_lapse_times_conformal_factor,
      const gsl::not_null<tnsr::IJ<DataVector, 3>*> flux_for_shift,
      const tnsr::ii<DataVector, 3>& /*conformal_metric*/,
      const tnsr::II<DataVector, 3>& inv_conformal_metric,
      const tnsr::i<DataVector, 3>& conformal_factor_gradient,
      const tnsr::i<DataVector, 3>& lapse_times_conformal_factor_gradient,
      const tnsr::ii<DataVector, 3>& shift_strain) noexcept {
    Fluxes<Equations::HamiltonianAndLapse, Geometry::NonEuclidean>::apply(
        flux_for_conformal_factor, flux_for_lapse_times_conformal_factor,
        inv_conformal_metric, conformal_factor_gradient,
        lapse_times_conformal_factor_gradient);
    Xcts::longitudinal_shift(flux_for_shift, inv_conformal_metric,
                             shift_strain);
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
      const tnsr::I<DataVector, 3>& shift) noexcept {
    Fluxes<Equations::HamiltonianAndLapse, Geometry::NonEuclidean>::apply(
        flux_for_conformal_factor_gradient,
        flux_for_lapse_times_conformal_factor_gradient, inv_conformal_metric,
        conformal_factor, lapse_times_conformal_factor);
    Elasticity::non_euclidean_auxiliary_fluxes(flux_for_shift_strain,
                                               conformal_metric, shift);
  }
  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};
/// \endcond

/*!
 * \brief The nonlinear source to the Hamiltonian constraint on a Euclidean
 * conformal background and with \f$\bar{u}_{ij}=0=\beta^i\f$.
 *
 * Computes \f$\frac{1}{12}\psi^5 K^2 - 2\pi\psi^5\rho\f$. Additional sources
 * can be added with `add_distortion_hamiltonian_sources` and
 * `add_non_euclidean_hamiltonian_or_lapse_sources`.
 *
 * \see `Xcts::FirstOrderSystem`
 */
void hamiltonian_sources(
    gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
    const Scalar<DataVector>& energy_density,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const Scalar<DataVector>& conformal_factor) noexcept;

/*!
 * \brief The linearization of `hamiltonian_sources`
 *
 * \see `Xcts::LinearizedFirstOrderSystem`
 */
void linearized_hamiltonian_sources(
    const gsl::not_null<Scalar<DataVector>*>
        source_for_conformal_factor_correction,
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
    gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
    const Scalar<DataVector>&
        longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
    const Scalar<DataVector>& conformal_factor) noexcept;

/*!
 * \brief The linearization of `add_distortion_hamiltonian_sources`
 *
 * Note that this linearization is only w.r.t. \f$\psi\f$.
 *
 * \see `Xcts::LinearizedFirstOrderSystem`
 */
void add_linearized_distortion_hamiltonian_sources(
    gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
    const Scalar<DataVector>&
        longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& conformal_factor_correction) noexcept;

/*!
 * \brief Add the contributions from a non-Euclidean background geometry to the
 * source of the Hamiltonian constraint or lapse equation
 *
 * Adds \f$\frac{1}{8}\psi\bar{R}\f$. This term appears both in the Hamiltonian
 * constraint and the lapse equation (where in the latter \f$\psi\f$ is
 * replaced by \f$\alpha\psi\f$).
 *
 * This term is linear.
 *
 * \see `Xcts::FirstOrderSystem`
 */
void add_non_euclidean_hamiltonian_or_lapse_sources(
    gsl::not_null<Scalar<DataVector>*> source,
    const Scalar<DataVector>& conformal_ricci_scalar,
    const Scalar<DataVector>& field) noexcept;

/*!
 * \brief The nonlinear source to the lapse equation on a Euclidean conformal
 * background and with \f$\bar{u}_{ij}=0=\beta^i\f$.
 *
 * Computes \f$(\alpha\psi)\psi^4\left(\frac{5}{12}K^2 + 2\pi\left(\rho +
 * 2S\right)\right) + \psi^5 \left(\beta^i\partial_i K - \partial_t K\right)\f$.
 * Additional sources can be added with
 * `add_distortion_hamiltonian_and_lapse_sources` and
 * `add_non_euclidean_hamiltonian_or_lapse_sources`.
 *
 * \see `Xcts::FirstOrderSystem`
 */
void lapse_sources(
    const gsl::not_null<Scalar<DataVector>*>
        source_for_lapse_times_conformal_factor,
    const Scalar<DataVector>& energy_density,
    const Scalar<DataVector>& stress_trace,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const Scalar<DataVector>& dt_extrinsic_curvature_trace,
    const Scalar<DataVector>& shift_dot_deriv_extrinsic_curvature_trace,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor) noexcept;

/*!
 * \brief The linearization of `lapse_sources`
 *
 * Note that this linearization is only w.r.t. \f$\psi\f$ and \f$\alpha\psi\f$.
 * The linearization w.r.t. \f$\beta^i\f$ is added in
 * `linearized_momentum_sources`.
 *
 * \see `Xcts::LinearizedFirstOrderSystem`
 */
void linearized_lapse_sources(
    const gsl::not_null<Scalar<DataVector>*>
        source_for_lapse_times_conformal_factor_correction,
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
    gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
    gsl::not_null<Scalar<DataVector>*> source_for_lapse_times_conformal_factor,
    const Scalar<DataVector>&
        longitudinal_shift_minus_dt_conformal_metric_square,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor) noexcept;

/*!
 * \brief The linearization of `add_distortion_hamiltonian_and_lapse_sources`
 *
 * Note that this linearization is only w.r.t. \f$\psi\f$ and \f$\alpha\psi\f$.
 *
 * \see `Xcts::LinearizedFirstOrderSystem`
 */
void add_linearized_distortion_hamiltonian_and_lapse_sources(
    gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
    gsl::not_null<Scalar<DataVector>*> source_for_lapse_times_conformal_factor,
    const Scalar<DataVector>&
        longitudinal_shift_minus_dt_conformal_metric_square,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>& lapse_times_conformal_factor_correction) noexcept;

/*!
 * \brief Compute the nonlinear source to the momentum constraint and add the
 * "distortion" source term to the Hamiltonian constraint and lapse equation.
 *
 * Computes \f$\left((\bar{L}\beta)^{ij} - \bar{u}^{ij}\right)\left(
 * \frac{w_j}{\alpha\psi} - 7 \frac{v_j}{\psi}\right)
 * + \partial_j\bar{u}^{ij}
 * + \frac{4}{3}\frac{\alpha\psi}{\psi}\bar{\gamma}^{ij}\partial_j K
 * + 16\pi\left(\alpha\psi\right)\psi^3 S^i\f$ for the momentum constraint.
 *
 * Note that the \f$\partial_j\bar{u}^{ij}\f$ term is not the full covariant
 * divergence, but only the Euclidean part of it. The non-Euclidean contribution
 * to this term can be added together with the non-Euclidean contribution to the
 * flux divergence of the dynamic shift variable with the
 * `Elasticity::add_non_euclidean_sources` function.
 *
 * Also adds \f$-\frac{1}{8}\psi^{-7}\bar{A}^2\f$ to the Hamiltonian constraint
 * and \f$\frac{7}{8}\alpha\psi^{-7}\bar{A}^2\f$ to the lapse equation.
 *
 * \see `Xcts::FirstOrderSystem`
 */
template <Geometry ConformalGeometry>
void momentum_sources(
    const gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
    const gsl::not_null<Scalar<DataVector>*>
        source_for_lapse_times_conformal_factor,
    const gsl::not_null<tnsr::I<DataVector, 3>*> source_for_shift,
    const tnsr::I<DataVector, 3>& momentum_density,
    const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,
    std::optional<std::reference_wrapper<const tnsr::ii<DataVector, 3>>>
        conformal_metric,
    std::optional<std::reference_wrapper<const tnsr::II<DataVector, 3>>>
        inv_conformal_metric,
    const tnsr::I<DataVector, 3>& minus_div_dt_conformal_metric,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, 3>& conformal_factor_flux,
    const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,
    const tnsr::II<DataVector, 3>&
        longitudinal_shift_minus_dt_conformal_metric) noexcept;

/*!
 * \brief Compute the linearization of `momentum_sources`
 *
 * \see `Xcts::LinearizedFirstOrderSystem`
 */
template <Geometry ConformalGeometry>
void linearized_momentum_sources(
    const gsl::not_null<Scalar<DataVector>*>
        source_for_conformal_factor_correction,
    const gsl::not_null<Scalar<DataVector>*>
        source_for_lapse_times_conformal_factor_correction,
    const gsl::not_null<tnsr::I<DataVector, 3>*> source_for_shift_correction,
    const tnsr::I<DataVector, 3>& momentum_density,
    const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,
    std::optional<std::reference_wrapper<const tnsr::ii<DataVector, 3>>>
        conformal_metric,
    std::optional<std::reference_wrapper<const tnsr::II<DataVector, 3>>>
        inv_conformal_metric,
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

/// The sources \f$S\f$ for the first-order formulation of the XCTS equations.
///
/// \see Xcts::FirstOrderSystem
template <Equations EnabledEquations, Geometry ConformalGeometry>
struct Sources;

/// \cond
template <>
struct Sources<Equations::Hamiltonian, Geometry::Euclidean> {
  using argument_tags = tmpl::list<
      gr::Tags::EnergyDensity<DataVector>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>,
      Tags::LongitudinalShiftMinusDtConformalMetricOverLapseSquare<DataVector>>;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
      const gsl::not_null<tnsr::i<DataVector, 3>*>
      /*source_for_conformal_factor_gradient*/,
      const Scalar<DataVector>& energy_density,
      const Scalar<DataVector>& extrinsic_curvature_trace,
      const Scalar<DataVector>&
          longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
      const Scalar<DataVector>& conformal_factor,
      const tnsr::I<DataVector, 3>& /*conformal_factor_flux*/) noexcept {
    hamiltonian_sources(source_for_conformal_factor, energy_density,
                        extrinsic_curvature_trace, conformal_factor);
    add_distortion_hamiltonian_sources(
        source_for_conformal_factor,
        longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
        conformal_factor);
  }
};

template <>
struct Sources<Equations::Hamiltonian, Geometry::NonEuclidean> {
  using argument_tags = tmpl::push_back<
      Sources<Equations::Hamiltonian, Geometry::Euclidean>::argument_tags,
      Tags::ConformalChristoffelContracted<DataVector, 3, Frame::Inertial>,
      Tags::ConformalRicciScalar<DataVector>>;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
      const gsl::not_null<tnsr::i<DataVector, 3>*>
          source_for_conformal_factor_gradient,
      const Scalar<DataVector>& energy_density,
      const Scalar<DataVector>& extrinsic_curvature_trace,
      const Scalar<DataVector>&
          longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
      const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
      const Scalar<DataVector>& conformal_ricci_scalar,
      const Scalar<DataVector>& conformal_factor,
      const tnsr::I<DataVector, 3>& conformal_factor_flux) noexcept {
    Sources<Equations::Hamiltonian, Geometry::Euclidean>::apply(
        source_for_conformal_factor, source_for_conformal_factor_gradient,
        energy_density, extrinsic_curvature_trace,
        longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
        conformal_factor, conformal_factor_flux);
    add_non_euclidean_hamiltonian_or_lapse_sources(
        source_for_conformal_factor, conformal_ricci_scalar, conformal_factor);
    Poisson::add_non_euclidean_sources(source_for_conformal_factor,
                                       conformal_christoffel_contracted,
                                       conformal_factor_flux);
  }
};

template <>
struct Sources<Equations::HamiltonianAndLapse, Geometry::Euclidean> {
  using argument_tags = tmpl::list<
      gr::Tags::EnergyDensity<DataVector>, gr::Tags::StressTrace<DataVector>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>,
      ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataVector>>,
      Tags::LongitudinalShiftMinusDtConformalMetricSquare<DataVector>,
      Tags::ShiftDotDerivExtrinsicCurvatureTrace<DataVector>>;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
      const gsl::not_null<Scalar<DataVector>*>
          source_for_lapse_times_conformal_factor,
      const gsl::not_null<tnsr::i<DataVector, 3>*>
      /*source_for_conformal_factor_gradient*/,
      const gsl::not_null<tnsr::i<DataVector, 3>*>
      /*source_for_lapse_times_conformal_factor_gradient*/,
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
    hamiltonian_sources(source_for_conformal_factor, energy_density,
                        extrinsic_curvature_trace, conformal_factor);
    lapse_sources(source_for_lapse_times_conformal_factor, energy_density,
                  stress_trace, extrinsic_curvature_trace,
                  dt_extrinsic_curvature_trace,
                  shift_dot_deriv_extrinsic_curvature_trace, conformal_factor,
                  lapse_times_conformal_factor);
    add_distortion_hamiltonian_and_lapse_sources(
        source_for_conformal_factor, source_for_lapse_times_conformal_factor,
        longitudinal_shift_minus_dt_conformal_metric_square, conformal_factor,
        lapse_times_conformal_factor);
  }
};

template <>
struct Sources<Equations::HamiltonianAndLapse, Geometry::NonEuclidean> {
  using argument_tags = tmpl::push_back<
      typename Sources<Equations::HamiltonianAndLapse,
                       Geometry::Euclidean>::argument_tags,
      Tags::ConformalChristoffelContracted<DataVector, 3, Frame::Inertial>,
      Tags::ConformalRicciScalar<DataVector>>;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
      const gsl::not_null<Scalar<DataVector>*>
          source_for_lapse_times_conformal_factor,
      const gsl::not_null<tnsr::i<DataVector, 3>*>
          source_for_conformal_factor_gradient,
      const gsl::not_null<tnsr::i<DataVector, 3>*>
          source_for_lapse_times_conformal_factor_gradient,
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
    Sources<Equations::HamiltonianAndLapse, Geometry::Euclidean>::apply(
        source_for_conformal_factor, source_for_lapse_times_conformal_factor,
        source_for_conformal_factor_gradient,
        source_for_lapse_times_conformal_factor_gradient, energy_density,
        stress_trace, extrinsic_curvature_trace, dt_extrinsic_curvature_trace,
        longitudinal_shift_minus_dt_conformal_metric_square,
        shift_dot_deriv_extrinsic_curvature_trace, conformal_factor,
        lapse_times_conformal_factor, conformal_factor_flux,
        lapse_times_conformal_factor_flux);
    add_non_euclidean_hamiltonian_or_lapse_sources(
        source_for_conformal_factor, conformal_ricci_scalar, conformal_factor);
    Poisson::add_non_euclidean_sources(source_for_conformal_factor,
                                       conformal_christoffel_contracted,
                                       conformal_factor_flux);
    add_non_euclidean_hamiltonian_or_lapse_sources(
        source_for_lapse_times_conformal_factor, conformal_ricci_scalar,
        lapse_times_conformal_factor);
    Poisson::add_non_euclidean_sources(source_for_lapse_times_conformal_factor,
                                       conformal_christoffel_contracted,
                                       lapse_times_conformal_factor_flux);
  }
};

template <>
struct Sources<Equations::HamiltonianLapseAndShift, Geometry::Euclidean> {
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
      const gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
      const gsl::not_null<Scalar<DataVector>*>
          source_for_lapse_times_conformal_factor,
      const gsl::not_null<tnsr::I<DataVector, 3>*> source_for_shift,
      const gsl::not_null<tnsr::i<DataVector, 3>*>
      /*source_for_conformal_factor_gradient*/,
      const gsl::not_null<tnsr::i<DataVector, 3>*>
      /*source_for_lapse_times_conformal_factor_gradient*/,
      const gsl::not_null<tnsr::ii<DataVector, 3>*>
      /*source_for_shift_strain*/,
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
      const tnsr::IJ<DataVector, 3>& longitudinal_shift_excess) noexcept {
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
    hamiltonian_sources(source_for_conformal_factor, energy_density,
                        extrinsic_curvature_trace, conformal_factor);
    lapse_sources(source_for_lapse_times_conformal_factor, energy_density,
                  stress_trace, extrinsic_curvature_trace,
                  dt_extrinsic_curvature_trace,
                  dot_product(shift, extrinsic_curvature_trace_gradient),
                  conformal_factor, lapse_times_conformal_factor);
    momentum_sources<Geometry::Euclidean>(
        source_for_conformal_factor, source_for_lapse_times_conformal_factor,
        source_for_shift, momentum_density, extrinsic_curvature_trace_gradient,
        std::nullopt, std::nullopt,
        div_longitudinal_shift_background_minus_dt_conformal_metric,
        conformal_factor, lapse_times_conformal_factor, conformal_factor_flux,
        lapse_times_conformal_factor_flux,
        longitudinal_shift_minus_dt_conformal_metric);
  }
};

template <>
struct Sources<Equations::HamiltonianLapseAndShift, Geometry::NonEuclidean> {
  using argument_tags = tmpl::push_back<
      typename Sources<Equations::HamiltonianLapseAndShift,
                       Geometry::Euclidean>::argument_tags,
      Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
      Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>,
      Tags::ConformalChristoffelFirstKind<DataVector, 3, Frame::Inertial>,
      Tags::ConformalChristoffelSecondKind<DataVector, 3, Frame::Inertial>,
      Tags::ConformalChristoffelContracted<DataVector, 3, Frame::Inertial>,
      Tags::ConformalRicciScalar<DataVector>>;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
      const gsl::not_null<Scalar<DataVector>*>
          source_for_lapse_times_conformal_factor,
      const gsl::not_null<tnsr::I<DataVector, 3>*> source_for_shift,
      const gsl::not_null<tnsr::i<DataVector, 3>*>
      /*source_for_conformal_factor_gradient*/,
      const gsl::not_null<tnsr::i<DataVector, 3>*>
      /*source_for_lapse_times_conformal_factor_gradient*/,
      const gsl::not_null<tnsr::ii<DataVector, 3>*> source_for_shift_strain,
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
      const tnsr::ijj<DataVector, 3>& conformal_christoffel_first_kind,
      const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind,
      const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
      const Scalar<DataVector>& conformal_ricci_scalar,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& lapse_times_conformal_factor,
      const tnsr::I<DataVector, 3>& shift_excess,
      const tnsr::I<DataVector, 3>& conformal_factor_flux,
      const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,
      const tnsr::IJ<DataVector, 3>& longitudinal_shift_excess) noexcept {
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
    hamiltonian_sources(source_for_conformal_factor, energy_density,
                        extrinsic_curvature_trace, conformal_factor);
    lapse_sources(source_for_lapse_times_conformal_factor, energy_density,
                  stress_trace, extrinsic_curvature_trace,
                  dt_extrinsic_curvature_trace,
                  dot_product(shift, extrinsic_curvature_trace_gradient),
                  conformal_factor, lapse_times_conformal_factor);
    add_non_euclidean_hamiltonian_or_lapse_sources(
        source_for_conformal_factor, conformal_ricci_scalar, conformal_factor);
    Poisson::add_non_euclidean_sources(source_for_conformal_factor,
                                       conformal_christoffel_contracted,
                                       conformal_factor_flux);
    add_non_euclidean_hamiltonian_or_lapse_sources(
        source_for_lapse_times_conformal_factor, conformal_ricci_scalar,
        lapse_times_conformal_factor);
    Poisson::add_non_euclidean_sources(source_for_lapse_times_conformal_factor,
                                       conformal_christoffel_contracted,
                                       lapse_times_conformal_factor_flux);
    momentum_sources<Geometry::NonEuclidean>(
        source_for_conformal_factor, source_for_lapse_times_conformal_factor,
        source_for_shift, momentum_density, extrinsic_curvature_trace_gradient,
        conformal_metric, inv_conformal_metric,
        div_longitudinal_shift_background_minus_dt_conformal_metric,
        conformal_factor, lapse_times_conformal_factor, conformal_factor_flux,
        lapse_times_conformal_factor_flux,
        longitudinal_shift_minus_dt_conformal_metric);
    Elasticity::add_non_euclidean_sources(
        source_for_shift, source_for_shift_strain,
        conformal_christoffel_first_kind, conformal_christoffel_second_kind,
        conformal_christoffel_contracted, shift_excess,
        longitudinal_shift_minus_dt_conformal_metric);
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
struct LinearizedSources<Equations::Hamiltonian, Geometry::Euclidean> {
  using argument_tags =
      tmpl::push_back<typename Sources<Equations::Hamiltonian,
                                       Geometry::Euclidean>::argument_tags,
                      Tags::ConformalFactor<DataVector>>;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*>
          source_for_conformal_factor_correction,
      const gsl::not_null<tnsr::i<
          DataVector, 3>*> /*source_for_conformal_factor_gradient_correction*/,
      const Scalar<DataVector>& energy_density,
      const Scalar<DataVector>& extrinsic_curvature_trace,
      const Scalar<DataVector>&
          longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& conformal_factor_correction,
      const tnsr::I<DataVector, 3>&
      /*conformal_factor_flux_correction*/) noexcept {
    linearized_hamiltonian_sources(source_for_conformal_factor_correction,
                                   energy_density, extrinsic_curvature_trace,
                                   conformal_factor,
                                   conformal_factor_correction);
    add_linearized_distortion_hamiltonian_sources(
        source_for_conformal_factor_correction,
        longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
        conformal_factor, conformal_factor_correction);
  }
};

template <>
struct LinearizedSources<Equations::Hamiltonian, Geometry::NonEuclidean> {
  using argument_tags =
      tmpl::push_back<typename Sources<Equations::Hamiltonian,
                                       Geometry::NonEuclidean>::argument_tags,
                      Tags::ConformalFactor<DataVector>>;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*>
          source_for_conformal_factor_correction,
      const gsl::not_null<tnsr::i<DataVector, 3>*>
          source_for_conformal_factor_gradient_correction,
      const Scalar<DataVector>& energy_density,
      const Scalar<DataVector>& extrinsic_curvature_trace,
      const Scalar<DataVector>&
          longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
      const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
      const Scalar<DataVector>& conformal_ricci_scalar,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& conformal_factor_correction,
      const tnsr::I<DataVector, 3>& conformal_factor_flux_correction) noexcept {
    LinearizedSources<Equations::Hamiltonian, Geometry::Euclidean>::apply(
        source_for_conformal_factor_correction,
        source_for_conformal_factor_gradient_correction, energy_density,
        extrinsic_curvature_trace,
        longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
        conformal_factor, conformal_factor_correction,
        conformal_factor_flux_correction);
    add_non_euclidean_hamiltonian_or_lapse_sources(
        source_for_conformal_factor_correction, conformal_ricci_scalar,
        conformal_factor_correction);
    Poisson::add_non_euclidean_sources(source_for_conformal_factor_correction,
                                       conformal_christoffel_contracted,
                                       conformal_factor_flux_correction);
  }
};

template <>
struct LinearizedSources<Equations::HamiltonianAndLapse, Geometry::Euclidean> {
  using argument_tags =
      tmpl::push_back<typename Sources<Equations::HamiltonianAndLapse,
                                       Geometry::Euclidean>::argument_tags,
                      Tags::ConformalFactor<DataVector>,
                      Tags::LapseTimesConformalFactor<DataVector>>;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*>
          source_for_conformal_factor_correction,
      const gsl::not_null<Scalar<DataVector>*>
          source_for_lapse_times_conformal_factor_correction,
      const gsl::not_null<tnsr::i<
          DataVector, 3>*> /*source_for_conformal_factor_gradient_correction*/,
      const gsl::not_null<tnsr::i<DataVector, 3>*>
      /*source_for_lapse_times_conformal_factor_gradient_correction*/,
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
    linearized_hamiltonian_sources(source_for_conformal_factor_correction,
                                   energy_density, extrinsic_curvature_trace,
                                   conformal_factor,
                                   conformal_factor_correction);
    linearized_lapse_sources(
        source_for_lapse_times_conformal_factor_correction, energy_density,
        stress_trace, extrinsic_curvature_trace, dt_extrinsic_curvature_trace,
        shift_dot_deriv_extrinsic_curvature_trace, conformal_factor,
        lapse_times_conformal_factor, conformal_factor_correction,
        lapse_times_conformal_factor_correction);
    add_linearized_distortion_hamiltonian_and_lapse_sources(
        source_for_conformal_factor_correction,
        source_for_lapse_times_conformal_factor_correction,
        longitudinal_shift_minus_dt_conformal_metric_square, conformal_factor,
        lapse_times_conformal_factor, conformal_factor_correction,
        lapse_times_conformal_factor_correction);
  }
};

template <>
struct LinearizedSources<Equations::HamiltonianAndLapse,
                         Geometry::NonEuclidean> {
  using argument_tags =
      tmpl::push_back<typename Sources<Equations::HamiltonianAndLapse,
                                       Geometry::NonEuclidean>::argument_tags,
                      Tags::ConformalFactor<DataVector>,
                      Tags::LapseTimesConformalFactor<DataVector>>;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*>
          source_for_conformal_factor_correction,
      const gsl::not_null<Scalar<DataVector>*>
          source_for_lapse_times_conformal_factor_correction,
      const gsl::not_null<tnsr::i<DataVector, 3>*>
          source_for_conformal_factor_gradient_correction,
      const gsl::not_null<tnsr::i<DataVector, 3>*>
          source_for_lapse_times_conformal_factor_gradient_correction,
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
    LinearizedSources<Equations::HamiltonianAndLapse, Geometry::Euclidean>::
        apply(source_for_conformal_factor_correction,
              source_for_lapse_times_conformal_factor_correction,
              source_for_conformal_factor_gradient_correction,
              source_for_lapse_times_conformal_factor_gradient_correction,
              energy_density, stress_trace, extrinsic_curvature_trace,
              dt_extrinsic_curvature_trace,
              longitudinal_shift_minus_dt_conformal_metric_square,
              shift_dot_deriv_extrinsic_curvature_trace, conformal_factor,
              lapse_times_conformal_factor, conformal_factor_correction,
              lapse_times_conformal_factor_correction,
              conformal_factor_flux_correction,
              lapse_times_conformal_factor_flux_correction);
    add_non_euclidean_hamiltonian_or_lapse_sources(
        source_for_conformal_factor_correction, conformal_ricci_scalar,
        conformal_factor_correction);
    Poisson::add_non_euclidean_sources(source_for_conformal_factor_correction,
                                       conformal_christoffel_contracted,
                                       conformal_factor_flux_correction);
    add_non_euclidean_hamiltonian_or_lapse_sources(
        source_for_lapse_times_conformal_factor_correction,
        conformal_ricci_scalar, lapse_times_conformal_factor_correction);
    Poisson::add_non_euclidean_sources(
        source_for_lapse_times_conformal_factor_correction,
        conformal_christoffel_contracted,
        lapse_times_conformal_factor_flux_correction);
  }
};

template <>
struct LinearizedSources<Equations::HamiltonianLapseAndShift,
                         Geometry::Euclidean> {
  using argument_tags = tmpl::push_back<
      typename Sources<Equations::HamiltonianLapseAndShift,
                       Geometry::Euclidean>::argument_tags,
      Tags::ConformalFactor<DataVector>,
      Tags::LapseTimesConformalFactor<DataVector>,
      Tags::ShiftExcess<DataVector, 3, Frame::Inertial>,
      ::Tags::Flux<Tags::ConformalFactor<DataVector>, tmpl::size_t<3>,
                   Frame::Inertial>,
      ::Tags::Flux<Tags::LapseTimesConformalFactor<DataVector>, tmpl::size_t<3>,
                   Frame::Inertial>,
      ::Tags::Flux<Tags::ShiftExcess<DataVector, 3, Frame::Inertial>,
                   tmpl::size_t<3>, Frame::Inertial>>;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*>
          source_for_conformal_factor_correction,
      const gsl::not_null<Scalar<DataVector>*>
          source_for_lapse_times_conformal_factor_correction,
      const gsl::not_null<tnsr::I<DataVector, 3>*> source_for_shift_correction,
      const gsl::not_null<tnsr::i<
          DataVector, 3>*> /*source_for_conformal_factor_gradient_correction*/,
      const gsl::not_null<tnsr::i<DataVector, 3>*>
      /*source_for_lapse_times_conformal_factor_gradient_correction*/,
      const gsl::not_null<
          tnsr::ii<DataVector, 3>*> /*source_for_shift_strain_correction*/,
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
      const tnsr::IJ<DataVector, 3>& longitudinal_shift_excess,
      const Scalar<DataVector>& conformal_factor_correction,
      const Scalar<DataVector>& lapse_times_conformal_factor_correction,
      const tnsr::I<DataVector, 3>& shift_correction,
      const tnsr::I<DataVector, 3>& conformal_factor_flux_correction,
      const tnsr::I<DataVector, 3>&
          lapse_times_conformal_factor_flux_correction,
      const tnsr::IJ<DataVector, 3>&
          longitudinal_shift_excess_correction) noexcept {
    auto shift = shift_background;
    for (size_t i = 0; i < 3; ++i) {
      shift.get(i) += shift_excess.get(i);
    }
    auto longitudinal_shift_minus_dt_conformal_metric =
        longitudinal_shift_background_minus_dt_conformal_metric;
    auto longitudinal_shift_excess_correction_symm =
        make_with_value<tnsr::II<DataVector, 3>>(
            longitudinal_shift_excess_correction, 0.);
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j <= i; ++j) {
        longitudinal_shift_minus_dt_conformal_metric.get(i, j) +=
            longitudinal_shift_excess.get(i, j);
        longitudinal_shift_excess_correction_symm.get(i, j) =
            longitudinal_shift_excess_correction.get(i, j);
      }
    }
    linearized_hamiltonian_sources(source_for_conformal_factor_correction,
                                   energy_density, extrinsic_curvature_trace,
                                   conformal_factor,
                                   conformal_factor_correction);
    linearized_lapse_sources(
        source_for_lapse_times_conformal_factor_correction, energy_density,
        stress_trace, extrinsic_curvature_trace, dt_extrinsic_curvature_trace,
        dot_product(shift, extrinsic_curvature_trace_gradient),
        conformal_factor, lapse_times_conformal_factor,
        conformal_factor_correction, lapse_times_conformal_factor_correction);
    linearized_momentum_sources<Geometry::Euclidean>(
        source_for_conformal_factor_correction,
        source_for_lapse_times_conformal_factor_correction,
        source_for_shift_correction, momentum_density,
        extrinsic_curvature_trace_gradient, std::nullopt, std::nullopt,
        conformal_factor, lapse_times_conformal_factor, conformal_factor_flux,
        lapse_times_conformal_factor_flux,
        longitudinal_shift_minus_dt_conformal_metric,
        conformal_factor_correction, lapse_times_conformal_factor_correction,
        shift_correction, conformal_factor_flux_correction,
        lapse_times_conformal_factor_flux_correction,
        longitudinal_shift_excess_correction_symm);
  }
};

template <>
struct LinearizedSources<Equations::HamiltonianLapseAndShift,
                         Geometry::NonEuclidean> {
  using argument_tags = tmpl::push_back<
      typename Sources<Equations::HamiltonianLapseAndShift,
                       Geometry::NonEuclidean>::argument_tags,
      Tags::ConformalFactor<DataVector>,
      Tags::LapseTimesConformalFactor<DataVector>,
      Tags::ShiftExcess<DataVector, 3, Frame::Inertial>,
      ::Tags::Flux<Tags::ConformalFactor<DataVector>, tmpl::size_t<3>,
                   Frame::Inertial>,
      ::Tags::Flux<Tags::LapseTimesConformalFactor<DataVector>, tmpl::size_t<3>,
                   Frame::Inertial>,
      ::Tags::Flux<Tags::ShiftExcess<DataVector, 3, Frame::Inertial>,
                   tmpl::size_t<3>, Frame::Inertial>>;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*>
          source_for_conformal_factor_correction,
      const gsl::not_null<Scalar<DataVector>*>
          source_for_lapse_times_conformal_factor_correction,
      const gsl::not_null<tnsr::I<DataVector, 3>*> source_for_shift_correction,
      const gsl::not_null<tnsr::i<
          DataVector, 3>*> /*source_for_conformal_factor_gradient_correction*/,
      const gsl::not_null<tnsr::i<DataVector, 3>*>
      /*source_for_lapse_times_conformal_factor_gradient_correction*/,
      const gsl::not_null<tnsr::ii<DataVector, 3>*>
          source_for_shift_strain_correction,
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
      const tnsr::ijj<DataVector, 3>& conformal_christoffel_first_kind,
      const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind,
      const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
      const Scalar<DataVector>& conformal_ricci_scalar,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& lapse_times_conformal_factor,
      const tnsr::I<DataVector, 3>& shift_excess,
      const tnsr::I<DataVector, 3>& conformal_factor_flux,
      const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,
      const tnsr::IJ<DataVector, 3>& longitudinal_shift_excess,
      const Scalar<DataVector>& conformal_factor_correction,
      const Scalar<DataVector>& lapse_times_conformal_factor_correction,
      const tnsr::I<DataVector, 3>& shift_correction,
      const tnsr::I<DataVector, 3>& conformal_factor_flux_correction,
      const tnsr::I<DataVector, 3>&
          lapse_times_conformal_factor_flux_correction,
      const tnsr::IJ<DataVector, 3>& longitudinal_shift_correction) noexcept {
    auto shift = shift_background;
    for (size_t i = 0; i < 3; ++i) {
      shift.get(i) += shift_excess.get(i);
    }
    auto longitudinal_shift_minus_dt_conformal_metric =
        longitudinal_shift_background_minus_dt_conformal_metric;
    auto longitudinal_shift_correction_symm =
        make_with_value<tnsr::II<DataVector, 3>>(longitudinal_shift_correction,
                                                 0.);
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j <= i; ++j) {
        longitudinal_shift_minus_dt_conformal_metric.get(i, j) +=
            longitudinal_shift_excess.get(i, j);
        longitudinal_shift_correction_symm.get(i, j) =
            longitudinal_shift_correction.get(i, j);
      }
    }
    linearized_hamiltonian_sources(source_for_conformal_factor_correction,
                                   energy_density, extrinsic_curvature_trace,
                                   conformal_factor,
                                   conformal_factor_correction);
    linearized_lapse_sources(
        source_for_lapse_times_conformal_factor_correction, energy_density,
        stress_trace, extrinsic_curvature_trace, dt_extrinsic_curvature_trace,
        dot_product(shift, extrinsic_curvature_trace_gradient),
        conformal_factor, lapse_times_conformal_factor,
        conformal_factor_correction, lapse_times_conformal_factor_correction);
    add_non_euclidean_hamiltonian_or_lapse_sources(
        source_for_conformal_factor_correction, conformal_ricci_scalar,
        conformal_factor_correction);
    Poisson::add_non_euclidean_sources(source_for_conformal_factor_correction,
                                       conformal_christoffel_contracted,
                                       conformal_factor_flux_correction);
    add_non_euclidean_hamiltonian_or_lapse_sources(
        source_for_lapse_times_conformal_factor_correction,
        conformal_ricci_scalar, lapse_times_conformal_factor_correction);
    Poisson::add_non_euclidean_sources(
        source_for_lapse_times_conformal_factor_correction,
        conformal_christoffel_contracted,
        lapse_times_conformal_factor_flux_correction);
    linearized_momentum_sources<Geometry::NonEuclidean>(
        source_for_conformal_factor_correction,
        source_for_lapse_times_conformal_factor_correction,
        source_for_shift_correction, momentum_density,
        extrinsic_curvature_trace_gradient, conformal_metric,
        inv_conformal_metric, conformal_factor, lapse_times_conformal_factor,
        conformal_factor_flux, lapse_times_conformal_factor_flux,
        longitudinal_shift_minus_dt_conformal_metric,
        conformal_factor_correction, lapse_times_conformal_factor_correction,
        shift_correction, conformal_factor_flux_correction,
        lapse_times_conformal_factor_flux_correction,
        longitudinal_shift_correction_symm);
    Elasticity::add_non_euclidean_sources(
        source_for_shift_correction, source_for_shift_strain_correction,
        conformal_christoffel_first_kind, conformal_christoffel_second_kind,
        conformal_christoffel_contracted, shift_correction,
        longitudinal_shift_correction_symm);
  }
};
/// \endcond

}  // namespace Xcts
