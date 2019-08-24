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
// @}

template <size_t Dim, typename VarsTag, typename ConformalFactorTag,
          typename LapseTimesConformalFactorTag>
struct ComputeHamiltonianAndLapseFluxes
    : db::add_tag_prefix<::Tags::Flux, VarsTag, tmpl::size_t<Dim>,
                         Frame::Inertial>,
      db::ComputeTag {
  using base = db::add_tag_prefix<::Tags::Flux, VarsTag, tmpl::size_t<Dim>,
                                  Frame::Inertial>;
  using argument_tags = tmpl::list<
      ::Tags::deriv<ConformalFactorTag, tmpl::size_t<Dim>, Frame::Inertial>,
      ::Tags::deriv<LapseTimesConformalFactorTag, tmpl::size_t<Dim>,
                    Frame::Inertial>>;
  static constexpr auto function(
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          conformal_factor_gradient,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          lapse_times_conformal_factor_gradient) noexcept {
    auto fluxes = make_with_value<db::item_type<db::add_tag_prefix<
        ::Tags::Flux, VarsTag, tmpl::size_t<Dim>, Frame::Inertial>>>(
        conformal_factor_gradient, 0.);
    Poisson::flux(
        make_not_null(&get<::Tags::Flux<ConformalFactorTag, tmpl::size_t<Dim>,
                                        Frame::Inertial>>(fluxes)),
        conformal_factor_gradient);
    Poisson::flux(
        make_not_null(
            &get<::Tags::Flux<LapseTimesConformalFactorTag, tmpl::size_t<Dim>,
                              Frame::Inertial>>(fluxes)),
        lapse_times_conformal_factor_gradient);
    return fluxes;
  }
};

template <size_t Dim, typename VarsTag, typename ConformalFactorTag,
          typename LapseTimesConformalFactorTag>
struct ComputeHamiltonianAndLapseNormalFluxes
    : db::add_tag_prefix<::Tags::NormalFlux, VarsTag, tmpl::size_t<Dim>,
                         Frame::Inertial>,
      db::ComputeTag {
  using base = db::add_tag_prefix<::Tags::NormalFlux, VarsTag,
                                  tmpl::size_t<Dim>, Frame::Inertial>;
  using argument_tags = tmpl::list<
      ConformalFactorTag, LapseTimesConformalFactorTag,
      ::Tags::Normalized<::Tags::UnnormalizedFaceNormal<Dim, Frame::Inertial>>>;
  static constexpr auto function(
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& lapse_times_conformal_factor,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& face_normal) noexcept {
    auto normal_times_conformal_factor = face_normal;
    for (size_t d = 0; d < Dim; d++) {
      normal_times_conformal_factor.get(d) *= get(conformal_factor);
    }
    auto fluxes = make_with_value<db::item_type<base>>(conformal_factor, 0.);
    Poisson::flux(
        make_not_null(
            &get<::Tags::NormalFlux<ConformalFactorTag, tmpl::size_t<Dim>,
                                    Frame::Inertial>>(fluxes)),
        std::move(normal_times_conformal_factor));
    auto normal_times_lapse_times_conformal_factor = face_normal;
    for (size_t d = 0; d < Dim; d++) {
      normal_times_lapse_times_conformal_factor.get(d) *= get(conformal_factor);
    }
    Poisson::flux(
        make_not_null(
            &get<::Tags::NormalFlux<LapseTimesConformalFactorTag,
                                    tmpl::size_t<Dim>, Frame::Inertial>>(
                fluxes)),
        std::move(normal_times_lapse_times_conformal_factor));
    return fluxes;
  }
};

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
void hamiltonian_sources(
    const gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& energy_density) noexcept;

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

void lapse_sources(const gsl::not_null<Scalar<DataVector>*>
                       source_for_lapse_times_conformal_factor,
                   const Scalar<DataVector>& lapse_times_conformal_factor,
                   const Scalar<DataVector>& conformal_factor,
                   const Scalar<DataVector>& energy_density,
                   const Scalar<DataVector>& stress_trace) noexcept;

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
          typename LapseTimesConformalFactorTag>
struct ComputeHamiltonianAndLapseSources
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
    hamiltonian_sources(
        make_not_null(&get<::Tags::Source<ConformalFactorTag>>(sources)),
        get<ConformalFactorTag>(vars), energy_density);
    lapse_sources(
        make_not_null(
            &get<::Tags::Source<LapseTimesConformalFactorTag>>(sources)),
        get<LapseTimesConformalFactorTag>(vars), get<ConformalFactorTag>(vars),
        energy_density, stress_trace);
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
void linearized_hamiltonian_sources(
    const gsl::not_null<Scalar<DataVector>*>
        source_for_conformal_factor_correction,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& energy_density) noexcept;

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

void linearized_lapse_sources(
    const gsl::not_null<Scalar<DataVector>*>
        source_for_lapse_times_conformal_factor_correction,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>& lapse_times_conformal_factor_correction,
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
          typename LapseTimesConformalFactorCorrectionTag,
          typename ConformalFactorTag, typename LapseTimesConformalFactorTag>
struct ComputeLinearizedHamiltonianAndLapseSources
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
    linearized_hamiltonian_sources(
        make_not_null(
            &get<::Tags::Source<ConformalFactorCorrectionTag>>(sources)),
        get<ConformalFactorCorrectionTag>(vars), conformal_factor,
        energy_density);
    linearized_lapse_sources(
        make_not_null(
            &get<::Tags::Source<LapseTimesConformalFactorCorrectionTag>>(
                sources)),
        get<ConformalFactorCorrectionTag>(vars),
        get<LapseTimesConformalFactorCorrectionTag>(vars), conformal_factor,
        lapse_times_conformal_factor, energy_density, stress_trace);
    return sources;
  }
};

}  // namespace Xcts
