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
namespace Xcts {
namespace Tags {
template <typename DataType>
struct ConformalFactor;
template <size_t Dim, typename Frame, typename DataType>
struct ConformalFactorGradient;
template <typename Tag>
struct Conformal;
}  // namespace Tags
}  // namespace Xcts
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
template <size_t Dim>
void first_order_fluxes(
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        flux_for_conformal_factor,
    const gsl::not_null<tnsr::IJ<DataVector, Dim, Frame::Inertial>*>
        flux_for_conformal_factor_gradient,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::I<DataVector, Dim, Frame::Inertial>&
        conformal_factor_gradient) noexcept;

template <size_t Dim, typename VarsTag, typename ConformalFactorTag,
          typename ConformalFactorGradientTag>
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
    first_order_fluxes(
        make_not_null(&get<::Tags::Flux<ConformalFactorTag, tmpl::size_t<Dim>,
                                        Frame::Inertial>>(fluxes)),
        make_not_null(
            &get<::Tags::Flux<ConformalFactorGradientTag, tmpl::size_t<Dim>,
                              Frame::Inertial>>(fluxes)),
        get<ConformalFactorTag>(vars), get<ConformalFactorGradientTag>(vars));
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
void first_order_sources(
    const gsl::not_null<Scalar<DataVector>*> source_for_conformal_factor,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        source_for_conformal_factor_gradient,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& conformal_factor_gradient,
    const Scalar<DataVector>& energy_density) noexcept;

template <size_t Dim, typename VarsTag, typename ConformalFactorTag,
          typename ConformalFactorGradientTag>
struct ComputeFirstOrderSources : db::add_tag_prefix<::Tags::Source, VarsTag>,
                                  db::ComputeTag {
  using argument_tags =
      tmpl::list<VarsTag, gr::Tags::EnergyDensity<DataVector>>;
  static constexpr auto function(
      const db::item_type<VarsTag>& vars,
      const Scalar<DataVector>& energy_density) noexcept {
    auto sources = make_with_value<
        db::item_type<db::add_tag_prefix<::Tags::Source, VarsTag>>>(vars, 0.);
    first_order_sources(
        make_not_null(&get<::Tags::Source<ConformalFactorTag>>(sources)),
        make_not_null(
            &get<::Tags::Source<ConformalFactorGradientTag>>(sources)),
        get<ConformalFactorTag>(vars), get<ConformalFactorGradientTag>(vars),
        energy_density);
    return sources;
  }
};
// @}

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
void first_order_linearized_sources(
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
struct ComputeFirstOrderLinearizedSources
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
    first_order_linearized_sources(
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
// @}

}  // namespace Xcts
