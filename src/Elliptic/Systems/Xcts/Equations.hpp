// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Elliptic/Systems/Elasticity/Equations.hpp"
#include "Elliptic/Systems/Poisson/Equations.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace Xcts {

enum class Equations {
  Hamiltonian,
  HamiltonianAndLapse,
  HamiltonianLapseAndShift
};

/*!
 * \brief Compute the longitudinal shift \f$\left(L\beta\right)^{ij}=2B^{ij} -
 * \frac{2}{3}\gamma^{ij}\mathrm{Tr}(B)\f$
 *
 * \see `Xcts::FirstOrderSystem`
 */
void longitudinal_shift(
    const gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*>
        flux_for_shift,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& shift_strain) noexcept;

template <Equations EnabledEquations>
struct Fluxes;

template <>
struct Fluxes<Equations::Hamiltonian> {
  using argument_tags = tmpl::list<>;
  static void apply(
      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          flux_for_conformal_factor,
      const tnsr::i<DataVector, 3, Frame::Inertial>&
          conformal_factor_gradient) noexcept {
    Poisson::euclidean_fluxes(flux_for_conformal_factor,
                              conformal_factor_gradient);
  }
  static void apply(
      const gsl::not_null<tnsr::Ij<DataVector, 3, Frame::Inertial>*>
          flux_for_conformal_factor_gradient,
      const Scalar<DataVector>& conformal_factor) noexcept {
    Poisson::auxiliary_fluxes(flux_for_conformal_factor_gradient,
                              conformal_factor);
  }
  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

template <>
struct Fluxes<Equations::HamiltonianAndLapse> {
  using argument_tags = tmpl::list<>;
  static void apply(
      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          flux_for_conformal_factor,
      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          flux_for_lapse_times_conformal_factor,
      const tnsr::i<DataVector, 3, Frame::Inertial>& conformal_factor_gradient,
      const tnsr::i<DataVector, 3, Frame::Inertial>&
          lapse_times_conformal_factor_gradient) noexcept {
    Poisson::euclidean_fluxes(flux_for_conformal_factor,
                              conformal_factor_gradient);
    Poisson::euclidean_fluxes(flux_for_lapse_times_conformal_factor,
                              lapse_times_conformal_factor_gradient);
  }
  static void apply(
      const gsl::not_null<tnsr::Ij<DataVector, 3, Frame::Inertial>*>
          flux_for_conformal_factor_gradient,
      const gsl::not_null<tnsr::Ij<DataVector, 3, Frame::Inertial>*>
          flux_for_lapse_times_conformal_factor_gradient,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& lapse_times_conformal_factor) noexcept {
    Poisson::auxiliary_fluxes(flux_for_conformal_factor_gradient,
                              conformal_factor);
    Poisson::auxiliary_fluxes(flux_for_lapse_times_conformal_factor_gradient,
                              lapse_times_conformal_factor);
  }
  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

template <>
struct Fluxes<Equations::HamiltonianLapseAndShift> {
  using argument_tags = tmpl::list<>;
  static void apply(
      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          flux_for_conformal_factor,
      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          flux_for_lapse_times_conformal_factor,
      const gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*>
          flux_for_shift,
      const tnsr::i<DataVector, 3, Frame::Inertial>& conformal_factor_gradient,
      const tnsr::i<DataVector, 3, Frame::Inertial>&
          lapse_times_conformal_factor_gradient,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& shift_strain) noexcept {
    Poisson::euclidean_fluxes(flux_for_conformal_factor,
                              conformal_factor_gradient);
    Poisson::euclidean_fluxes(flux_for_lapse_times_conformal_factor,
                              lapse_times_conformal_factor_gradient);
    longitudinal_shift(flux_for_shift, shift_strain);
  }
  static void apply(
      const gsl::not_null<tnsr::Ij<DataVector, 3, Frame::Inertial>*>
          flux_for_conformal_factor_gradient,
      const gsl::not_null<tnsr::Ij<DataVector, 3, Frame::Inertial>*>
          flux_for_lapse_times_conformal_factor_gradient,
      const gsl::not_null<tnsr::Ijj<DataVector, 3, Frame::Inertial>*>
          flux_for_shift_strain,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& lapse_times_conformal_factor,
      const tnsr::I<DataVector, 3, Frame::Inertial>& shift) noexcept {
    Poisson::auxiliary_fluxes(flux_for_conformal_factor_gradient,
                              conformal_factor);
    Poisson::auxiliary_fluxes(flux_for_lapse_times_conformal_factor_gradient,
                              lapse_times_conformal_factor);
    Elasticity::auxiliary_fluxes(flux_for_shift_strain, shift);
  }
  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

/*!
 * \brief Compute the nonlinear source to the Hamiltonian constraint, ignoring
 * contributions from a non-vanishing shift
 *
 * Computes \f$- 2\pi\psi^5\rho\f$. Contributions from a non-vanishing shift
 * are added in `momentum_sources`.
 *
 * \see `Xcts::FirstOrderSystem`
 */
void hamiltonian_sources(
    gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
    const Scalar<DataVector>& energy_density,
    const Scalar<DataVector>& conformal_factor) noexcept;

/*!
 * \brief Compute the linearization of `hamiltonian_sources`
 *
 * \see `Xcts::LinearizedFirstOrderSystem`
 */
void linearized_hamiltonian_sources(
    const gsl::not_null<Scalar<DataVector>*>
        source_for_conformal_factor_correction,
    const Scalar<DataVector>& energy_density,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& conformal_factor_correction) noexcept;

/*!
 * \brief Compute the nonlinear source to the lapse equation, ignoring
 * contributions from a non-vanishing shift
 *
 * Computes \f$2\pi\left(\alpha\psi\right)\psi^4\left(\rho + 2S\right)\f$.
 * Contributions from a non-vanishing shift are added in `momentum_sources`.
 *
 * \see `Xcts::FirstOrderSystem`
 */
void lapse_sources(
    const gsl::not_null<Scalar<DataVector>*>
        source_for_lapse_times_conformal_factor,
    const Scalar<DataVector>& energy_density,
    const Scalar<DataVector>& stress_trace,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor) noexcept;

/*!
 * \brief Compute the linearization of `lapse_sources`
 *
 * \see `Xcts::LinearizedFirstOrderSystem`
 */
void linearized_lapse_sources(
    const gsl::not_null<Scalar<DataVector>*>
        source_for_lapse_times_conformal_factor_correction,
    const Scalar<DataVector>& energy_density,
    const Scalar<DataVector>& stress_trace,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>& lapse_times_conformal_factor_correction) noexcept;

/*!
 * \brief Compute the nonlinear source to the momentum constraint and adds
 * contributions from a non-vanishing shift to the Hamiltonian constraint and
 * the lapse equation
 *
 * Computes \f$\left(L\beta\right)^{ij}\left(
 * \frac{w_j}{\alpha\psi} - 7 \frac{v_j}{\psi}\right)
 * + 16\pi\left(\alpha\psi\right)\psi^3 S^i\f$ for the momentum constraint.
 *
 * Also adds \f$-\frac{1}{8}\frac{\psi^7}{\left(\alpha\psi\right)^2}
 * \mathrm{Tr}(B)^2\f$ to the Hamiltonian constraint and
 * \f$\frac{7}{8}\frac{\psi^6}{\alpha\psi}\mathrm{Tr}(B)^2\f$ to the lapse
 * equation.
 *
 * \see `Xcts::FirstOrderSystem`
 */
void momentum_sources(
    const gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
    const gsl::not_null<Scalar<DataVector>*>
        source_for_lapse_times_conformal_factor,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        source_for_shift,
    const tnsr::I<DataVector, 3, Frame::Inertial>& momentum_density,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::i<DataVector, 3, Frame::Inertial>& conformal_factor_gradient,
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        lapse_times_conformal_factor_gradient,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& shift_strain) noexcept;

/*!
 * \brief Compute the linearization of `momentum_sources`
 *
 * \see `Xcts::LinearizedFirstOrderSystem`
 */
void linearized_momentum_sources(
    const gsl::not_null<Scalar<DataVector>*>
        source_for_conformal_factor_correction,
    const gsl::not_null<Scalar<DataVector>*>
        source_for_lapse_times_conformal_factor_correction,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        source_for_shift_correction,
    const tnsr::I<DataVector, 3, Frame::Inertial>& momentum_density,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::i<DataVector, 3, Frame::Inertial>& conformal_factor_gradient,
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        lapse_times_conformal_factor_gradient,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& shift_strain,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>& lapse_times_conformal_factor_correction,
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        conformal_factor_gradient_correction,
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        lapse_times_conformal_factor_gradient_correction,
    const tnsr::ii<DataVector, 3, Frame::Inertial>&
        shift_strain_correction) noexcept;

template <Equations EnabledEquations>
struct Sources;

template <>
struct Sources<Equations::Hamiltonian> {
  using argument_tags = tmpl::list<gr::Tags::EnergyDensity<DataVector>>;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
      const Scalar<DataVector>& energy_density,
      const Scalar<DataVector>& conformal_factor,
      const tnsr::i<DataVector, 3,
                    Frame::Inertial>& /*conformal_factor_gradient*/) noexcept {
    hamiltonian_sources(source_for_conformal_factor, energy_density,
                        conformal_factor);
  }
};

template <>
struct Sources<Equations::HamiltonianAndLapse> {
  using argument_tags = tmpl::list<gr::Tags::EnergyDensity<DataVector>,
                                   gr::Tags::StressTrace<DataVector>>;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
      const gsl::not_null<Scalar<DataVector>*>
          source_for_lapse_times_conformal_factor,
      const Scalar<DataVector>& energy_density,
      const Scalar<DataVector>& stress_trace,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& lapse_times_conformal_factor,
      const tnsr::i<DataVector, 3, Frame::Inertial>&
      /*conformal_factor_gradient*/,
      const tnsr::i<DataVector, 3, Frame::Inertial>&
      /*lapse_times_conformal_factor_gradient*/) noexcept {
    hamiltonian_sources(source_for_conformal_factor, energy_density,
                        conformal_factor);
    lapse_sources(source_for_lapse_times_conformal_factor, energy_density,
                  stress_trace, conformal_factor, lapse_times_conformal_factor);
  }
};

template <>
struct Sources<Equations::HamiltonianLapseAndShift> {
  using argument_tags =
      tmpl::list<gr::Tags::EnergyDensity<DataVector>,
                 gr::Tags::StressTrace<DataVector>,
                 gr::Tags::MomentumDensity<3, Frame::Inertial, DataVector>>;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
      const gsl::not_null<Scalar<DataVector>*>
          source_for_lapse_times_conformal_factor,
      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          source_for_shift,
      const Scalar<DataVector>& energy_density,
      const Scalar<DataVector>& stress_trace,
      const tnsr::I<DataVector, 3, Frame::Inertial>& momentum_density,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& lapse_times_conformal_factor,
      const tnsr::I<DataVector, 3, Frame::Inertial>& /*shift*/,
      const tnsr::i<DataVector, 3, Frame::Inertial>& conformal_factor_gradient,
      const tnsr::i<DataVector, 3, Frame::Inertial>&
          lapse_times_conformal_factor_gradient,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& shift_strain) noexcept {
    hamiltonian_sources(source_for_conformal_factor, energy_density,
                        conformal_factor);
    lapse_sources(source_for_lapse_times_conformal_factor, energy_density,
                  stress_trace, conformal_factor, lapse_times_conformal_factor);
    momentum_sources(source_for_conformal_factor,
                     source_for_lapse_times_conformal_factor, source_for_shift,
                     momentum_density, conformal_factor,
                     lapse_times_conformal_factor, conformal_factor_gradient,
                     lapse_times_conformal_factor_gradient, shift_strain);
  }
};

template <Equations EnabledEquations>
struct LinearizedSources;

template <>
struct LinearizedSources<Equations::Hamiltonian> {
  using argument_tags = tmpl::list<gr::Tags::EnergyDensity<DataVector>,
                                   Xcts::Tags::ConformalFactor<DataVector>>;
  static constexpr auto apply = linearized_hamiltonian_sources;
};

template <>
struct LinearizedSources<Equations::HamiltonianAndLapse> {
  using argument_tags =
      tmpl::list<gr::Tags::EnergyDensity<DataVector>,
                 gr::Tags::StressTrace<DataVector>,
                 Xcts::Tags::ConformalFactor<DataVector>,
                 Xcts::Tags::LapseTimesConformalFactor<DataVector>>;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*>
          source_for_conformal_factor_correction,
      const gsl::not_null<Scalar<DataVector>*>
          source_for_lapse_times_conformal_factor_correction,
      const Scalar<DataVector>& energy_density,
      const Scalar<DataVector>& stress_trace,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& lapse_times_conformal_factor,
      const Scalar<DataVector>& conformal_factor_correction,
      const Scalar<DataVector>& lapse_times_conformal_factor_correction,
      const tnsr::i<DataVector, 3, Frame::Inertial>&
      /*conformal_factor_gradient_correction*/,
      const tnsr::i<DataVector, 3, Frame::Inertial>&
      /*lapse_times_conformal_factor_gradient_correction*/) noexcept {
    linearized_hamiltonian_sources(source_for_conformal_factor_correction,
                                   energy_density, conformal_factor,
                                   conformal_factor_correction);
    linearized_lapse_sources(
        source_for_lapse_times_conformal_factor_correction, energy_density,
        stress_trace, conformal_factor, lapse_times_conformal_factor,
        conformal_factor_correction, lapse_times_conformal_factor_correction);
  }
};

template <>
struct LinearizedSources<Equations::HamiltonianLapseAndShift> {
  using argument_tags = tmpl::list<
      gr::Tags::EnergyDensity<DataVector>, gr::Tags::StressTrace<DataVector>,
      gr::Tags::MomentumDensity<3, Frame::Inertial, DataVector>,
      Xcts::Tags::ConformalFactor<DataVector>,
      Xcts::Tags::LapseTimesConformalFactor<DataVector>,
      ::Tags::deriv<Xcts::Tags::ConformalFactor<DataVector>, tmpl::size_t<3>,
                    Frame::Inertial>,
      ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataVector>,
                    tmpl::size_t<3>, Frame::Inertial>,
      Xcts::Tags::ShiftStrain<3, Frame::Inertial, DataVector>>;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*>
          source_for_conformal_factor_correction,
      const gsl::not_null<Scalar<DataVector>*>
          source_for_lapse_times_conformal_factor_correction,
      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          source_for_shift_correction,
      const Scalar<DataVector>& energy_density,
      const Scalar<DataVector>& stress_trace,
      const tnsr::I<DataVector, 3, Frame::Inertial>& momentum_density,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& lapse_times_conformal_factor,
      const tnsr::i<DataVector, 3, Frame::Inertial>& conformal_factor_gradient,
      const tnsr::i<DataVector, 3, Frame::Inertial>&
          lapse_times_conformal_factor_gradient,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& shift_strain,
      const Scalar<DataVector>& conformal_factor_correction,
      const Scalar<DataVector>& lapse_times_conformal_factor_correction,
      const tnsr::I<DataVector, 3, Frame::Inertial>& /*shift_correction*/,
      const tnsr::i<DataVector, 3, Frame::Inertial>&
          conformal_factor_gradient_correction,
      const tnsr::i<DataVector, 3, Frame::Inertial>&
          lapse_times_conformal_factor_gradient_correction,
      const tnsr::ii<DataVector, 3, Frame::Inertial>&
          shift_strain_correction) noexcept {
    linearized_hamiltonian_sources(source_for_conformal_factor_correction,
                                   energy_density, conformal_factor,
                                   conformal_factor_correction);
    linearized_lapse_sources(
        source_for_lapse_times_conformal_factor_correction, energy_density,
        stress_trace, conformal_factor, lapse_times_conformal_factor,
        conformal_factor_correction, lapse_times_conformal_factor_correction);
    linearized_momentum_sources(
        source_for_conformal_factor_correction,
        source_for_lapse_times_conformal_factor_correction,
        source_for_shift_correction, momentum_density, conformal_factor,
        lapse_times_conformal_factor, conformal_factor_gradient,
        lapse_times_conformal_factor_gradient, shift_strain,
        conformal_factor_correction, lapse_times_conformal_factor_correction,
        conformal_factor_gradient_correction,
        lapse_times_conformal_factor_gradient_correction,
        shift_strain_correction);
  }
};

}  // namespace Xcts
