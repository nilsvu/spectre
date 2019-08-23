// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/FaceNormal.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Poisson {

template <size_t Dim>
void flux(
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        flux_for_field,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& field_gradient) noexcept;

template <size_t Dim>
struct Flux {
  void operator()(
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          flux_for_field,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& field_gradient) const
      noexcept {
    flux(flux_for_field, field_gradient);
  }

  void pup(PUP::er& /*p*/) noexcept {}
};

template <size_t Dim, typename VarsTag, typename FieldTag>
struct ComputeFluxes : db::add_tag_prefix<::Tags::Flux, VarsTag,
                                          tmpl::size_t<Dim>, Frame::Inertial>,
                       db::ComputeTag {
  using base = db::add_tag_prefix<::Tags::Flux, VarsTag, tmpl::size_t<Dim>,
                                  Frame::Inertial>;
  using argument_tags =
      tmpl::list<::Tags::deriv<FieldTag, tmpl::size_t<Dim>, Frame::Inertial>>;
  static constexpr auto function(
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          field_gradient) noexcept {
    auto fluxes = make_with_value<db::item_type<base>>(field_gradient, 0.);
    flux(make_not_null(
             &get<::Tags::Flux<FieldTag, tmpl::size_t<Dim>, Frame::Inertial>>(
                 fluxes)),
         field_gradient);
    return fluxes;
  }
};

template <size_t Dim, typename VarsTag, typename FieldTag>
struct ComputeNormalFluxes
    : db::add_tag_prefix<::Tags::NormalFlux, VarsTag, tmpl::size_t<Dim>,
                         Frame::Inertial>,
      db::ComputeTag {
  using base = db::add_tag_prefix<::Tags::NormalFlux, VarsTag,
                                  tmpl::size_t<Dim>, Frame::Inertial>;
  using argument_tags = tmpl::list<
      FieldTag,
      ::Tags::Normalized<::Tags::UnnormalizedFaceNormal<Dim, Frame::Inertial>>>;
  static constexpr auto function(
      const Scalar<DataVector>& field,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& face_normal) noexcept {
    auto normal_times_field = face_normal;
    for (size_t d = 0; d < Dim; d++) {
      normal_times_field.get(d) *= get(field);
    }
    auto fluxes = make_with_value<db::item_type<base>>(field, 0.);
    flux(make_not_null(&get<::Tags::NormalFlux<FieldTag, tmpl::size_t<Dim>,
                                               Frame::Inertial>>(fluxes)),
         std::move(normal_times_field));
    return fluxes;
  }
};

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
    const gsl::not_null<tnsr::Ij<DataVector, Dim, Frame::Inertial>*>
        flux_for_auxiliary_field,
    const Scalar<DataVector>& field,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& auxiliary_field) noexcept;

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
    auto fluxes = make_with_value<db::item_type<base>>(vars, 0.);
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

template <size_t Dim, typename VarsTag, typename FieldTag,
          typename AuxiliaryFieldTag>
struct ComputeFirstOrderNormalFluxes
    : db::add_tag_prefix<::Tags::NormalFlux, VarsTag, tmpl::size_t<Dim>,
                         Frame::Inertial>,
      db::ComputeTag {
  using base = db::add_tag_prefix<::Tags::NormalFlux, VarsTag,
                                  tmpl::size_t<Dim>, Frame::Inertial>;
  using argument_tags = tmpl::list<
      FieldTag,
      ::Tags::Normalized<::Tags::UnnormalizedFaceNormal<Dim, Frame::Inertial>>>;
  static constexpr auto function(
      const Scalar<DataVector>& field,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& face_normal) noexcept {
    auto normal_times_field = face_normal;
    for (size_t d = 0; d < Dim; d++) {
      normal_times_field.get(d) *= get(field);
    }
    auto fluxes = make_with_value<db::item_type<base>>(field, 0.);
    first_order_fluxes(
        make_not_null(&get<::Tags::NormalFlux<FieldTag, tmpl::size_t<Dim>,
                                              Frame::Inertial>>(fluxes)),
        make_not_null(
            &get<::Tags::NormalFlux<AuxiliaryFieldTag, tmpl::size_t<Dim>,
                                    Frame::Inertial>>(fluxes)),
        make_with_value<Scalar<DataVector>>(
            field, std::numeric_limits<double>::signaling_NaN()),
        std::move(normal_times_field));
    return fluxes;
  }
};

template <size_t Dim, typename VarsTag, typename FieldTag>
struct ComputeSecondOrderSources : db::add_tag_prefix<::Tags::Source, VarsTag>,
                                   db::ComputeTag {
  using argument_tags = tmpl::list<VarsTag>;
  static constexpr auto function(const db::item_type<VarsTag>& vars) noexcept {
    return make_with_value<
        db::item_type<db::add_tag_prefix<::Tags::Source, VarsTag>>>(vars, 0.);
  }
};

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
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        source_for_auxiliary_field,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& auxiliary_field) noexcept;

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
        get<AuxiliaryFieldTag>(vars));
    return sources;
  }
};
// @}

}  // namespace Poisson
