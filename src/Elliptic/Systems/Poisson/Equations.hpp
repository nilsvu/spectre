// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "Domain/FaceNormal.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"  // IWYU pragma: keep
#include "Options/Options.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tags::deriv
// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables

/// \cond
class DataVector;
template <size_t>
class Mesh;
namespace Tags {
template <typename>
struct NormalDotFlux;
template <typename>
struct Normalized;
}  // namespace Tags
namespace LinearSolver {
namespace Tags {
template <typename>
struct Operand;
}  // namespace Tags
}  // namespace LinearSolver
namespace Poisson {
struct Field;
template <size_t>
struct AuxiliaryField;
}  // namespace Poisson
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Poisson {

// @{
/*!
 * \brief Compute the fluxes \f$F^i_A\f$ for the first-order formulation of the
 * Poisson equation.
 *
 * The divergence of the fluxes computed here is taken to build the Poisson
 * operator.
 *
 * \note This compute item can be used both in the volume and on an interface
 * (using `Tags::InterfaceComputeItem`).
 *
 * \see `Poisson::FirstOrderSystem`
 */
template <size_t Dim>
void first_order_fluxes(
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        flux_for_field,
    const gsl::not_null<tnsr::IJ<DataVector, Dim, Frame::Inertial>*>
        flux_for_auxiliary_field,
    const Scalar<DataVector>& field,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& auxiliary_field) noexcept;

template <size_t Dim, typename VarsTag, typename FieldTag,
          typename AuxiliaryFieldTag>
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
        make_not_null(
            &get<::Tags::Flux<FieldTag, tmpl::size_t<Dim>, Frame::Inertial>>(
                fluxes)),
        make_not_null(&get<::Tags::Flux<AuxiliaryFieldTag, tmpl::size_t<Dim>,
                                        Frame::Inertial>>(fluxes)),
        get<FieldTag>(vars), get<AuxiliaryFieldTag>(vars));
    return fluxes;
  }
};
// @}

// @{
/*!
 * \brief Compute the sources \f$S_A\f$ for the first-order formulation of the
 * Poisson equation.
 *
 * These are the quantities that source the divergence of the fluxes computed in
 * `Poisson::first_order_fluxes`. Terms that are independent of the system
 * variables are not included here.
 *
 * \see `Poisson::FirstOrderSystem`
 */
template <size_t Dim>
void first_order_sources(
    const gsl::not_null<Scalar<DataVector>*> source_for_field,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        source_for_auxiliary_field,
    const Scalar<DataVector>& field,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& auxiliary_field) noexcept;

template <size_t Dim, typename VarsTag, typename FieldTag,
          typename AuxiliaryFieldTag>
struct ComputeFirstOrderSources : db::add_tag_prefix<::Tags::Source, VarsTag>,
                                  db::ComputeTag {
  using argument_tags = tmpl::list<VarsTag>;
  static constexpr auto function(const db::item_type<VarsTag>& vars) noexcept {
    auto sources = make_with_value<
        db::item_type<db::add_tag_prefix<::Tags::Source, VarsTag>>>(vars, 0.);
    first_order_sources(
        make_not_null(&get<::Tags::Source<FieldTag>>(sources)),
        make_not_null(&get<::Tags::Source<AuxiliaryFieldTag>>(sources)),
        get<FieldTag>(vars), get<AuxiliaryFieldTag>(vars));
    return sources;
  }
};
// @}

}  // namespace Poisson
