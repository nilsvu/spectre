// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesHelpers.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Systems/Poisson/Equations.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace Xcts {

// @{
/*!
 * \brief Compute the fluxes \f$F^i_A\f$ for the first-order formulation of the
 * XCTS equations.
 *
 * The divergence of the fluxes computed here is taken to build the XCTS
 * operator.
 *
 * \note This compute item can be used both in the volume and on an interface
 * (using `Tags::InterfaceComputeItem`).
 *
 * \see `Xcts::FirstOrderSystem`
 */
template <size_t Dim, typename VarsTag, typename ConformalFactorTag,
          typename ConformalFactorGradientTag>
using ComputeFirstOrderHamiltonianFluxes =
    Poisson::ComputeFirstOrderFluxes<Dim, VarsTag, ConformalFactorTag,
                                     ConformalFactorGradientTag>;

template <size_t Dim, typename VarsTag, typename ConformalFactorTag,
          typename ConformalFactorGradientTag,
          typename LapseTimesConformalFactorTag,
          typename LapseTimesConformalFactorGradientTag>
struct ComputeFirstOrderHamiltonianAndLapseFluxes
    : db::add_tag_prefix<::Tags::Flux, VarsTag, tmpl::size_t<Dim>,
                         Frame::Inertial>,
      db::ComputeTag {
  using base = db::add_tag_prefix<::Tags::Flux, VarsTag, tmpl::size_t<Dim>,
                                  Frame::Inertial>;
  using argument_tags = tmpl::list<VarsTag>;
  static constexpr auto function(const db::item_type<VarsTag>& vars) noexcept {
    auto fluxes = make_with_value<db::item_type<db::add_tag_prefix<
        ::Tags::Flux, VarsTag, tmpl::size_t<Dim>, Frame::Inertial>>>(vars, 0.);
    Poisson::first_order_fluxes(
        make_not_null(&get<::Tags::Flux<ConformalFactorTag, tmpl::size_t<Dim>,
                                        Frame::Inertial>>(fluxes)),
        make_not_null(
            &get<::Tags::Flux<ConformalFactorGradientTag, tmpl::size_t<Dim>,
                              Frame::Inertial>>(fluxes)),
        get<ConformalFactorTag>(vars), get<ConformalFactorGradientTag>(vars));
    Poisson::first_order_fluxes(
        make_not_null(
            &get<::Tags::Flux<LapseTimesConformalFactorTag, tmpl::size_t<Dim>,
                              Frame::Inertial>>(fluxes)),
        make_not_null(
            &get<::Tags::Flux<LapseTimesConformalFactorGradientTag,
                              tmpl::size_t<Dim>, Frame::Inertial>>(fluxes)),
        get<LapseTimesConformalFactorTag>(vars),
        get<LapseTimesConformalFactorGradientTag>(vars));
    return fluxes;
  }
};

template <size_t Dim, typename VarsTag, typename ConformalFactorTag,
          typename ConformalFactorGradientTag,
          typename LapseTimesConformalFactorTag,
          typename LapseTimesConformalFactorGradientTag, typename ShiftTag,
          typename ShiftStrainTag>
struct ComputeFirstOrderFluxes
    : db::add_tag_prefix<::Tags::Flux, VarsTag, tmpl::size_t<Dim>,
                         Frame::Inertial>,
      db::ComputeTag {
  using base = db::add_tag_prefix<::Tags::Flux, VarsTag, tmpl::size_t<Dim>,
                                  Frame::Inertial>;
  using argument_tags = tmpl::list<VarsTag>;
  static constexpr auto function(const db::item_type<VarsTag>& vars) noexcept {
    auto fluxes = make_with_value<db::item_type<db::add_tag_prefix<
        ::Tags::Flux, VarsTag, tmpl::size_t<Dim>, Frame::Inertial>>>(vars, 0.);
    Poisson::first_order_fluxes(
        make_not_null(&get<::Tags::Flux<ConformalFactorTag, tmpl::size_t<Dim>,
                                        Frame::Inertial>>(fluxes)),
        make_not_null(
            &get<::Tags::Flux<ConformalFactorGradientTag, tmpl::size_t<Dim>,
                              Frame::Inertial>>(fluxes)),
        get<ConformalFactorTag>(vars), get<ConformalFactorGradientTag>(vars));
    Poisson::first_order_fluxes(
        make_not_null(
            &get<::Tags::Flux<LapseTimesConformalFactorTag, tmpl::size_t<Dim>,
                              Frame::Inertial>>(fluxes)),
        make_not_null(
            &get<::Tags::Flux<LapseTimesConformalFactorGradientTag,
                              tmpl::size_t<Dim>, Frame::Inertial>>(fluxes)),
        get<LapseTimesConformalFactorTag>(vars),
        get<LapseTimesConformalFactorGradientTag>(vars));
    // Flux for shift
    const auto& shift_strain = get<ShiftStrainTag>(vars);
    auto shift_strain_trace = get<0, 0>(shift_strain);
    for (size_t d = 1; d < Dim; d++) {
      shift_strain_trace += shift_strain.get(d, d);
    }
    auto& flux_for_shift =
        get<::Tags::Flux<ShiftTag, tmpl::size_t<Dim>, Frame::Inertial>>(fluxes);
    flux_for_shift = shift_strain;
    for (size_t d = 0; d < Dim; d++) {
      flux_for_shift.get(d, d) -= shift_strain_trace / 3.;
    }
    // Flux for shift strain
    auto& flux_for_shift_strain =
        get<::Tags::Flux<ShiftStrainTag, tmpl::size_t<Dim>, Frame::Inertial>>(
            fluxes);
    for (size_t i = 0; i < Dim; i++) {
      for (size_t j = 0; j < Dim; j++) {
        flux_for_shift_strain.get(i, i, j) =
            0.5 * (shift.get(i) + shift.get(j));
      }
    }
    return fluxes;
  }
};
// @}

// @{
/*!
 * \brief Compute the sources \f$S_A\f$ for the first-order formulation of the
 * XCTS equations.
 *
 * These are the quantities that source the divergence of the fluxes computed in
 * `Xcts::first_order_fluxes`. Terms that are independent of the system
 * variables are not included here.
 *
 * \see `Xcts::FirstOrderSystem`
 */
template <size_t Dim>
void first_order_hamiltonian_sources(
    const gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        source_for_conformal_factor_gradient,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& conformal_factor_gradient,
    const Scalar<DataVector>& energy_density) noexcept;

template <size_t Dim, typename VarsTag, typename ConformalFactorTag,
          typename ConformalFactorGradientTag>
struct ComputeFirstOrderHamiltonianSources
    : db::add_tag_prefix<::Tags::Source, VarsTag>,
      db::ComputeTag {
  using argument_tags =
      tmpl::list<VarsTag, gr::Tags::EnergyDensity<DataVector>>;
  static constexpr auto function(
      const db::item_type<VarsTag>& vars,
      const Scalar<DataVector>& energy_density) noexcept {
    auto sources = make_with_value<
        db::item_type<db::add_tag_prefix<::Tags::Source, VarsTag>>>(vars, 0.);
    first_order_hamiltonian_sources(
        make_not_null(&get<::Tags::Source<ConformalFactorTag>>(sources)),
        make_not_null(
            &get<::Tags::Source<ConformalFactorGradientTag>>(sources)),
        get<ConformalFactorTag>(vars), get<ConformalFactorGradientTag>(vars),
        energy_density);
    return sources;
  }
};

template <size_t Dim>
void first_order_lapse_sources(
    const gsl::not_null<Scalar<DataVector>*>
        source_for_lapse_times_conformal_factor,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        source_for_lapse_times_conformal_factor_gradient,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, Dim, Frame::Inertial>&
        lapse_times_conformal_factor_gradient,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& energy_density,
    const Scalar<DataVector>& stress_trace) noexcept;

template <size_t Dim, typename VarsTag, typename ConformalFactorTag,
          typename ConformalFactorGradientTag,
          typename LapseTimesConformalFactorTag,
          typename LapseTimesConformalFactorGradientTag>
struct ComputeFirstOrderHamiltonianAndLapseSources
    : db::add_tag_prefix<::Tags::Source, VarsTag>,
      db::ComputeTag {
  using argument_tags = tmpl::list<VarsTag, gr::Tags::EnergyDensity<DataVector>,
                                   gr::Tags::StressTrace<DataVector>>;
  static constexpr auto function(
      const db::item_type<VarsTag>& vars,
      const Scalar<DataVector>& energy_density,
      const Scalar<DataVector>& stress_trace) noexcept {
    auto sources = make_with_value<
        db::item_type<db::add_tag_prefix<::Tags::Source, VarsTag>>>(vars, 0.);
    first_order_hamiltonian_sources(
        make_not_null(&get<::Tags::Source<ConformalFactorTag>>(sources)),
        make_not_null(
            &get<::Tags::Source<ConformalFactorGradientTag>>(sources)),
        get<ConformalFactorTag>(vars), get<ConformalFactorGradientTag>(vars),
        energy_density);
    first_order_lapse_sources(
        make_not_null(
            &get<::Tags::Source<LapseTimesConformalFactorTag>>(sources)),
        make_not_null(
            &get<::Tags::Source<LapseTimesConformalFactorGradientTag>>(
                sources)),
        get<LapseTimesConformalFactorTag>(vars),
        get<LapseTimesConformalFactorGradientTag>(vars),
        get<ConformalFactorTag>(vars), energy_density, stress_trace);
    return sources;
  }
};
// @}

template <size_t Dim, typename VarsTag, typename ConformalFactorTag,
          typename ConformalFactorGradientTag,
          typename LapseTimesConformalFactorTag,
          typename LapseTimesConformalFactorGradientTag, typename ShiftTag,
          typename ShiftStrainTag>
struct ComputeFirstOrderSources : db::add_tag_prefix<::Tags::Source, VarsTag>,
                                  db::ComputeTag {
  using argument_tags = tmpl::list<
      VarsTag, ::Tags::Flux<ShiftTag, tmpl::size_t<Dim>, Frame::Inertial>,
      gr::Tags::EnergyDensity<DataVector>, gr::Tags::StressTrace<DataVector>,
      gr::Tags::MomentumDensity<Dim, Frame::Inertial, DataVector>>;
  static constexpr auto function(
      const db::item_type<VarsTag>& vars,
      const tnsr::IJ<DataVector, Dim, Frame::Inertial>& longitudinal_shift,
      const Scalar<DataVector>& energy_density,
      const Scalar<DataVector>& stress_trace,
      const tnsr::I<DataVector, Dim, Frame::Inertial>&
          momentum_density) noexcept {
    auto sources = make_with_value<
        db::item_type<db::add_tag_prefix<::Tags::Source, VarsTag>>>(vars, 0.);
    first_order_hamiltonian_sources(
        make_not_null(&get<::Tags::Source<ConformalFactorTag>>(sources)),
        make_not_null(
            &get<::Tags::Source<ConformalFactorGradientTag>>(sources)),
        get<ConformalFactorTag>(vars), get<ConformalFactorGradientTag>(vars),
        energy_density);
    first_order_lapse_sources(
        make_not_null(
            &get<::Tags::Source<LapseTimesConformalFactorTag>>(sources)),
        make_not_null(
            &get<::Tags::Source<LapseTimesConformalFactorGradientTag>>(
                sources)),
        get<LapseTimesConformalFactorTag>(vars),
        get<LapseTimesConformalFactorGradientTag>(vars),
        get<ConformalFactorTag>(vars), energy_density, stress_trace);
    // Add shift terms
    DataVector longitudinal_shift_square{vars.number_of_grid_points(), 0.};
    for (size_t i = 0; i < Dim; i++) {
      for (size_t j = 0; j < Dim; j++) {
        longitudinal_shift_square += square(longitudinal_shift.get(i, j));
      }
    }
    get(get<::Tags::Source<ConformalFactorTag>>(sources)) -=
        pow<7>(get(conformal_factor)) /
        square(get(lapse_times_conformal_factor)) * longitudinal_shift_square /
        32.;
    get(get<::Tags::Source<LapseTimesConformalFactorTag>>(sources)) +=
        pow<6>(get(conformal_factor)) / get(lapse_times_conformal_factor) *
        longitudinal_shift_square * 7. / 32.;
    // Compute shift source
    auto& shift_source = get<::Tags::Source<ShiftTag>>(sources);
    auto longitudinal_shift_dot_grad_psi =
        make_with_value<tnsr::I<DataVector, Dim, Frame::Inertial>>(
            longitudinal_shift, 0.);
    auto longitudinal_shift_dot_grad_alphapsi =
        make_with_value<tnsr::I<DataVector, Dim, Frame::Inertial>>(
            longitudinal_shift, 0.);
    for (size_t i = 0; i < Dim; i++) {
      for (size_t j = 0; j < Dim; j++) {
        longitudinal_shift_dot_grad_psi.get(i) +=
            longitudinal_shift.get(i, j) *
            get<ConformalFactorGradientTag>(vars).get(j);
        longitudinal_shift_dot_grad_alphapsi.get(i) +=
            longitudinal_shift.get(i, j) *
            get<LapseTimesConformalFactorGradientTag>(vars).get(j);
      }
    }
    for (size_t d = 0; d < Dim; d++) {
      shift_source.get(d) = (longitudinal_shift_dot_grad_alphapsi.get(d) /
                                 get(get<LapseTimesConformalFactorTag>(vars)) -
                             7. * longitudinal_shift_dot_grad_psi.get(d) /
                                 get(get<ConformalFactorTag>(vars))) +
                            16. * M_PI *
                                get(get<LapseTimesConformalFactorTag>(vars)) *
                                pow<3>(get(get<ConformalFactorTag>(vars))) *
                                momentum_density.get(d);
    }
    // Compute shift strain source
    get<::Tags::Source<ShiftStrainTag>>(sources) = get<ShiftStrainTag>(vars);
    return sources;
  }
};

// @{
/*!
 * \brief Compute the sources \f$S_A\f$ for the first-order formulation of the
 * **linearized** XCTS equations.
 *
 * These are the quantities that source the divergence of the fluxes computed in
 * `Xcts::first_order_fluxes` (which is already linear). Terms that are
 * independent of the system variables are not included here.
 *
 * \see `Xcts::FirstOrderSystem`
 */
template <size_t Dim>
void first_order_linearized_hamiltonian_sources(
    const gsl::not_null<Scalar<DataVector>*>
        source_for_conformal_factor_correction,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        source_for_conformal_factor_gradient_correction,
    const Scalar<DataVector>& conformal_factor_correction,
    const tnsr::I<DataVector, Dim, Frame::Inertial>&
        conformal_factor_gradient_correction,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& energy_density) noexcept;

template <size_t Dim, typename VarsTag, typename ConformalFactorCorrectionTag,
          typename ConformalFactorGradientCorrectionTag,
          typename ConformalFactorTag>
struct ComputeFirstOrderLinearizedHamiltonianSources
    : db::add_tag_prefix<::Tags::Source, VarsTag>,
      db::ComputeTag {
  using argument_tags = tmpl::list<VarsTag, ConformalFactorTag,
                                   gr::Tags::EnergyDensity<DataVector>>;
  static constexpr auto function(
      const db::item_type<VarsTag>& vars,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& energy_density) noexcept {
    auto sources = make_with_value<
        db::item_type<db::add_tag_prefix<::Tags::Source, VarsTag>>>(vars, 0.);
    first_order_linearized_hamiltonian_sources(
        make_not_null(
            &get<::Tags::Source<ConformalFactorCorrectionTag>>(sources)),
        make_not_null(
            &get<::Tags::Source<ConformalFactorGradientCorrectionTag>>(
                sources)),
        get<ConformalFactorCorrectionTag>(vars),
        get<ConformalFactorGradientCorrectionTag>(vars), conformal_factor,
        energy_density);
    return sources;
  }
};

template <size_t Dim>
void first_order_linearized_lapse_sources(
    const gsl::not_null<Scalar<DataVector>*>
        source_for_lapse_times_conformal_factor_correction,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        source_for_lapse_times_conformal_factor_gradient_correction,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>& lapse_times_conformal_factor_correction,
    const tnsr::I<DataVector, Dim, Frame::Inertial>&
        lapse_times_conformal_factor_gradient_correction,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const Scalar<DataVector>& energy_density,
    const Scalar<DataVector>& stress_trace) noexcept;

template <size_t Dim, typename VarsTag, typename ConformalFactorCorrectionTag,
          typename ConformalFactorGradientCorrectionTag,
          typename LapseTimesConformalFactorCorrectionTag,
          typename LapseTimesConformalFactorGradientCorrectionTag,
          typename ConformalFactorTag, typename LapseTimesConformalFactorTag>
struct ComputeFirstOrderLinearizedHamiltonianAndLapseSources
    : db::add_tag_prefix<::Tags::Source, VarsTag>,
      db::ComputeTag {
  using argument_tags =
      tmpl::list<VarsTag, ConformalFactorTag, LapseTimesConformalFactorTag,
                 gr::Tags::EnergyDensity<DataVector>,
                 gr::Tags::StressTrace<DataVector>>;
  static constexpr auto function(
      const db::item_type<VarsTag>& vars,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& lapse_times_conformal_factor,
      const Scalar<DataVector>& energy_density,
      const Scalar<DataVector>& stress_trace) noexcept {
    auto sources = make_with_value<
        db::item_type<db::add_tag_prefix<::Tags::Source, VarsTag>>>(vars, 0.);
    first_order_linearized_hamiltonian_sources(
        make_not_null(
            &get<::Tags::Source<ConformalFactorCorrectionTag>>(sources)),
        make_not_null(
            &get<::Tags::Source<ConformalFactorGradientCorrectionTag>>(
                sources)),
        get<ConformalFactorCorrectionTag>(vars),
        get<ConformalFactorGradientCorrectionTag>(vars), conformal_factor,
        energy_density);
    first_order_linearized_lapse_sources(
        make_not_null(
            &get<::Tags::Source<LapseTimesConformalFactorCorrectionTag>>(
                sources)),
        make_not_null(
            &get<
                ::Tags::Source<LapseTimesConformalFactorGradientCorrectionTag>>(
                sources)),
        get<ConformalFactorCorrectionTag>(vars),
        get<LapseTimesConformalFactorCorrectionTag>(vars),
        get<LapseTimesConformalFactorGradientCorrectionTag>(vars),
        conformal_factor, lapse_times_conformal_factor, energy_density,
        stress_trace);
    return sources;
  }
};
// @}

template <size_t Dim, typename VarsTag, typename ConformalFactorCorrectionTag,
          typename ConformalFactorGradientCorrectionTag,
          typename LapseTimesConformalFactorCorrectionTag,
          typename LapseTimesConformalFactorGradientCorrectionTag,
          typename ShiftCorrectionTag, typename ShiftStrainCorrectionTag,
          typename ConformalFactorTag, typename LapseTimesConformalFactorTag,
          typename ShiftTag>
struct ComputeFirstOrderLinearizedSources
    : db::add_tag_prefix<::Tags::Source, VarsTag>,
      db::ComputeTag {
  using argument_tags = tmpl::list<
      VarsTag,
      ::Tags::Flux<ShiftCorrectionTag, tmpl::size_t<Dim>, Frame::Inertial>,
      ConformalFactorTag, LapseTimesConformalFactorTag,
      ::Tags::Flux<ShiftTag, tmpl::size_t<Dim>, Frame::Inertial>,
      gr::Tags::EnergyDensity<DataVector>, gr::Tags::StressTrace<DataVector>,
      gr::Tags::MomentumDensity<Dim, Frame::Inertial, DataVector>>;
  static constexpr auto function(
      const db::item_type<VarsTag>& vars,
      const tnsr::IJ<DataVector, Dim, Frame::Inertial>&
          longitudinal_shift_correction,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& lapse_times_conformal_factor,
      const tnsr::IJ<DataVector, Dim, Frame::Inertial>& longitudinal_shift,
      const Scalar<DataVector>& energy_density,
      const Scalar<DataVector>& stress_trace,
      const tnsr::I<DataVector, Dim, Frame::Inertial>&
          momentum_density) noexcept {
    auto sources = make_with_value<
        db::item_type<db::add_tag_prefix<::Tags::Source, VarsTag>>>(vars, 0.);
    first_order_linearized_hamiltonian_sources(
        make_not_null(
            &get<::Tags::Source<ConformalFactorCorrectionTag>>(sources)),
        make_not_null(
            &get<::Tags::Source<ConformalFactorGradientCorrectionTag>>(
                sources)),
        get<ConformalFactorCorrectionTag>(vars),
        get<ConformalFactorGradientCorrectionTag>(vars), conformal_factor,
        energy_density);
    first_order_linearized_lapse_sources(
        make_not_null(
            &get<::Tags::Source<LapseTimesConformalFactorCorrectionTag>>(
                sources)),
        make_not_null(
            &get<
                ::Tags::Source<LapseTimesConformalFactorGradientCorrectionTag>>(
                sources)),
        get<ConformalFactorCorrectionTag>(vars),
        get<LapseTimesConformalFactorCorrectionTag>(vars),
        get<LapseTimesConformalFactorGradientCorrectionTag>(vars),
        conformal_factor, lapse_times_conformal_factor, energy_density,
        stress_trace);
    // Add shift terms
    DataVector longitudinal_shift_square{vars.number_of_grid_points(), 0.};
    for (size_t i = 0; i < Dim; i++) {
      for (size_t j = 0; j < Dim; j++) {
        longitudinal_shift_square += square(longitudinal_shift.get(i, j));
      }
    }
    DataVector longitudinal_shift_dot_correction{vars.number_of_grid_points(),
                                                 0.};
    for (size_t i = 0; i < Dim; i++) {
      for (size_t j = 0; j < Dim; j++) {
        longitudinal_shift_dot_correction +=
            longitudinal_shift.get(i, j) *
            longitudinal_shift_correction.get(i, j);
      }
    }
    get(get<::Tags::Source<ConformalFactorCorrectionTag>>(sources)) +=
        -7. / 32. * pow<6>(get(conformal_factor)) /
            square(get(lapse_times_conformal_factor)) *
            longitudinal_shift_square *
            get(get<ConformalFactorCorrectionTag>(vars)) +
        1. / 16. * pow<7>(get(conformal_factor)) /
            pow<3>(get(lapse_times_conformal_factor)) *
            longitudinal_shift_square *
            get(get<LapseTimesConformalFactorCorrectionTag>(vars)) -
        1. / 16. * pow<7>(get(conformal_factor)) /
            square(get(lapse_times_conformal_factor)) *
            longitudinal_shift_dot_correction);
    // TODO
    get(get<::Tags::Source<LapseTimesConformalFactorTag>>(sources)) +=
        pow<6>(get(conformal_factor)) / get(lapse_times_conformal_factor) *
        longitudinal_shift_square * 7. / 32.;
    // Compute shift source
    auto& shift_source = get<::Tags::Source<ShiftTag>>(sources);
    auto longitudinal_shift_dot_grad_psi =
        make_with_value<tnsr::I<DataVector, Dim, Frame::Inertial>>(
            longitudinal_shift, 0.);
    auto longitudinal_shift_dot_grad_alphapsi =
        make_with_value<tnsr::I<DataVector, Dim, Frame::Inertial>>(
            longitudinal_shift, 0.);
    for (size_t i = 0; i < Dim; i++) {
      for (size_t j = 0; j < Dim; j++) {
        longitudinal_shift_dot_grad_psi.get(i) +=
            longitudinal_shift.get(i, j) *
            get<ConformalFactorGradientTag>(vars).get(j);
        longitudinal_shift_dot_grad_alphapsi.get(i) +=
            longitudinal_shift.get(i, j) *
            get<LapseTimesConformalFactorGradientTag>(vars).get(j);
      }
    }
    for (size_t d = 0; d < Dim; d++) {
      shift_source.get(d) = (longitudinal_shift_dot_grad_alphapsi.get(d) /
                                 get(get<LapseTimesConformalFactorTag>(vars)) -
                             7. * longitudinal_shift_dot_grad_psi.get(d) /
                                 get(get<ConformalFactorTag>(vars))) +
                            16. * M_PI *
                                get(get<LapseTimesConformalFactorTag>(vars)) *
                                pow<3>(get(get<ConformalFactorTag>(vars))) *
                                momentum_density.get(d);
    }
    return sources;
  }
};

}  // namespace Xcts
