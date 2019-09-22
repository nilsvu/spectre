// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/Tensor/TypeAliases.hpp"
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

template <size_t Dim>
void longitudinal_shift(
    const gsl::not_null<tnsr::IJ<DataVector, Dim, Frame::Inertial>*>
        flux_for_shift,
    const tnsr::ii<DataVector, Dim, Frame::Inertial>& shift_strain) noexcept;

template <size_t Dim>
void momentum_auxiliary_fluxes(
    const gsl::not_null<tnsr::Ijj<DataVector, Dim, Frame::Inertial>*>
        flux_for_shift_strain,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift) noexcept;

template <size_t Dim, Equations EnabledEquations>
struct Fluxes;

template <size_t Dim>
struct Fluxes<Dim, Equations::Hamiltonian> {
  using argument_tags = tmpl::list<>;
  static void apply(
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          flux_for_conformal_factor,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          conformal_factor_gradient) noexcept {
    Poisson::euclidean_fluxes(flux_for_conformal_factor,
                              conformal_factor_gradient);
  }
  static void apply(
      const gsl::not_null<tnsr::Ij<DataVector, Dim, Frame::Inertial>*>
          flux_for_conformal_factor_gradient,
      const Scalar<DataVector>& conformal_factor) noexcept {
    Poisson::auxiliary_fluxes(flux_for_conformal_factor_gradient,
                              conformal_factor);
  }
  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

template <size_t Dim>
struct Fluxes<Dim, Equations::HamiltonianAndLapse> {
  using argument_tags = tmpl::list<>;
  static void apply(
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          flux_for_conformal_factor,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          flux_for_lapse_times_conformal_factor,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          conformal_factor_gradient,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          lapse_times_conformal_factor_gradient) noexcept {
    Poisson::euclidean_fluxes(flux_for_conformal_factor,
                              conformal_factor_gradient);
    Poisson::euclidean_fluxes(flux_for_lapse_times_conformal_factor,
                              lapse_times_conformal_factor_gradient);
  }
  static void apply(
      const gsl::not_null<tnsr::Ij<DataVector, Dim, Frame::Inertial>*>
          flux_for_conformal_factor_gradient,
      const gsl::not_null<tnsr::Ij<DataVector, Dim, Frame::Inertial>*>
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

template <size_t Dim>
struct Fluxes<Dim, Equations::HamiltonianLapseAndShift> {
  using argument_tags = tmpl::list<>;
  static void apply(
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          flux_for_conformal_factor,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          flux_for_lapse_times_conformal_factor,
      const gsl::not_null<tnsr::IJ<DataVector, Dim, Frame::Inertial>*>
          flux_for_shift,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          conformal_factor_gradient,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          lapse_times_conformal_factor_gradient,
      const tnsr::ii<DataVector, Dim, Frame::Inertial>& shift_strain) noexcept {
    Poisson::euclidean_fluxes(flux_for_conformal_factor,
                              conformal_factor_gradient);
    Poisson::euclidean_fluxes(flux_for_lapse_times_conformal_factor,
                              lapse_times_conformal_factor_gradient);
    longitudinal_shift(flux_for_shift, shift_strain);
  }
  static void apply(
      const gsl::not_null<tnsr::Ij<DataVector, Dim, Frame::Inertial>*>
          flux_for_conformal_factor_gradient,
      const gsl::not_null<tnsr::Ij<DataVector, Dim, Frame::Inertial>*>
          flux_for_lapse_times_conformal_factor_gradient,
      const gsl::not_null<tnsr::Ijj<DataVector, Dim, Frame::Inertial>*>
          flux_for_shift_strain,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& lapse_times_conformal_factor,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& shift) noexcept {
    Poisson::auxiliary_fluxes(flux_for_conformal_factor_gradient,
                              conformal_factor);
    Poisson::auxiliary_fluxes(flux_for_lapse_times_conformal_factor_gradient,
                              lapse_times_conformal_factor);
    momentum_auxiliary_fluxes(flux_for_shift_strain, shift);
  }
  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

void hamiltonian_sources(
    gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
    const Scalar<DataVector>& energy_density,
    const Scalar<DataVector>& conformal_factor) noexcept;

void linearized_hamiltonian_sources(
    const gsl::not_null<Scalar<DataVector>*>
        source_for_conformal_factor_correction,
    const Scalar<DataVector>& energy_density,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& conformal_factor_correction) noexcept;

void lapse_sources(
    const gsl::not_null<Scalar<DataVector>*>
        source_for_lapse_times_conformal_factor,
    const Scalar<DataVector>& energy_density,
    const Scalar<DataVector>& stress_trace,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor) noexcept;

void linearized_lapse_sources(
    const gsl::not_null<Scalar<DataVector>*>
        source_for_lapse_times_conformal_factor_correction,
    const Scalar<DataVector>& energy_density,
    const Scalar<DataVector>& stress_trace,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>& lapse_times_conformal_factor_correction) noexcept;

template <size_t Dim>
void momentum_sources(
    const gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
    const gsl::not_null<Scalar<DataVector>*>
        source_for_lapse_times_conformal_factor,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        source_for_shift,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& momentum_density,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& conformal_factor_gradient,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        lapse_times_conformal_factor_gradient,
    const tnsr::ii<DataVector, Dim, Frame::Inertial>& shift_strain) noexcept;

template <size_t Dim>
void linearized_momentum_sources(
    const gsl::not_null<Scalar<DataVector>*>
        source_for_conformal_factor_correction,
    const gsl::not_null<Scalar<DataVector>*>
        source_for_lapse_times_conformal_factor_correction,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        source_for_shift_correction,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& momentum_density,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& conformal_factor_gradient,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        lapse_times_conformal_factor_gradient,
    const tnsr::ii<DataVector, Dim, Frame::Inertial>& shift_strain,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>& lapse_times_conformal_factor_correction,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        conformal_factor_gradient_correction,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        lapse_times_conformal_factor_gradient_correction,
    const tnsr::ii<DataVector, Dim, Frame::Inertial>&
        shift_strain_correction) noexcept;

template <size_t Dim, Equations EnabledEquations>
struct Sources;

template <size_t Dim>
struct Sources<Dim, Equations::Hamiltonian> {
  using argument_tags = tmpl::list<gr::Tags::EnergyDensity<DataVector>>;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
      const Scalar<DataVector>& energy_density,
      const Scalar<DataVector>& conformal_factor,
      const tnsr::i<DataVector, Dim,
                    Frame::Inertial>& /*conformal_factor_gradient*/) noexcept {
    hamiltonian_sources(source_for_conformal_factor, energy_density,
                        conformal_factor);
  }
};

template <size_t Dim>
struct Sources<Dim, Equations::HamiltonianAndLapse> {
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
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
      /*conformal_factor_gradient*/,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
      /*lapse_times_conformal_factor_gradient*/) noexcept {
    hamiltonian_sources(source_for_conformal_factor, energy_density,
                        conformal_factor);
    lapse_sources(source_for_lapse_times_conformal_factor, energy_density,
                  stress_trace, conformal_factor, lapse_times_conformal_factor);
  }
};

template <size_t Dim>
struct Sources<Dim, Equations::HamiltonianLapseAndShift> {
  using argument_tags =
      tmpl::list<gr::Tags::EnergyDensity<DataVector>,
                 gr::Tags::StressTrace<DataVector>,
                 gr::Tags::MomentumDensity<Dim, Frame::Inertial, DataVector>>;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
      const gsl::not_null<Scalar<DataVector>*>
          source_for_lapse_times_conformal_factor,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          source_for_shift,
      const Scalar<DataVector>& energy_density,
      const Scalar<DataVector>& stress_trace,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& momentum_density,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& lapse_times_conformal_factor,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& /*shift*/,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          conformal_factor_gradient,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          lapse_times_conformal_factor_gradient,
      const tnsr::ii<DataVector, Dim, Frame::Inertial>& shift_strain) noexcept {
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

template <size_t Dim, Equations EnabledEquations>
struct LinearizedSources;

template <size_t Dim>
struct LinearizedSources<Dim, Equations::Hamiltonian> {
  using argument_tags = tmpl::list<gr::Tags::EnergyDensity<DataVector>,
                                   Xcts::Tags::ConformalFactor<DataVector>>;
  static constexpr auto apply = linearized_hamiltonian_sources;
};

template <size_t Dim>
struct LinearizedSources<Dim, Equations::HamiltonianAndLapse> {
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
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
      /*conformal_factor_gradient_correction*/,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
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

template <size_t Dim>
struct LinearizedSources<Dim, Equations::HamiltonianLapseAndShift> {
  using argument_tags = tmpl::list<
      gr::Tags::EnergyDensity<DataVector>, gr::Tags::StressTrace<DataVector>,
      gr::Tags::MomentumDensity<Dim, Frame::Inertial, DataVector>,
      Xcts::Tags::ConformalFactor<DataVector>,
      Xcts::Tags::LapseTimesConformalFactor<DataVector>,
      ::Tags::deriv<Xcts::Tags::ConformalFactor<DataVector>, tmpl::size_t<Dim>,
                    Frame::Inertial>,
      ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataVector>,
                    tmpl::size_t<Dim>, Frame::Inertial>,
      Xcts::Tags::ShiftStrain<Dim, Frame::Inertial, DataVector>>;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*>
          source_for_conformal_factor_correction,
      const gsl::not_null<Scalar<DataVector>*>
          source_for_lapse_times_conformal_factor_correction,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          source_for_shift_correction,
      const Scalar<DataVector>& energy_density,
      const Scalar<DataVector>& stress_trace,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& momentum_density,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& lapse_times_conformal_factor,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          conformal_factor_gradient,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          lapse_times_conformal_factor_gradient,
      const tnsr::ii<DataVector, Dim, Frame::Inertial>& shift_strain,
      const Scalar<DataVector>& conformal_factor_correction,
      const Scalar<DataVector>& lapse_times_conformal_factor_correction,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& /*shift_correction*/,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          conformal_factor_gradient_correction,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          lapse_times_conformal_factor_gradient_correction,
      const tnsr::ii<DataVector, Dim, Frame::Inertial>&
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
