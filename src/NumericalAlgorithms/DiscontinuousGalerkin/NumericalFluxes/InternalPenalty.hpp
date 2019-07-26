// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Metafunctions.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "DataStructures/VariablesHelpers.hpp"
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
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace dg {
namespace NumericalFluxes {

/*!
 * \brief The internal penalty flux for the first-order formulation of the
 * Poisson equation.
 *
 * Computes the internal penalty numerical flux (see e.g.
 * \cite HesthavenWarburton, section 7.2) dotted with the interface unit normal
 *
 * \f{align}
 * n_i F^{*i}_u = \frac{1}{2} n_i \left(\partial^i u_\mathrm{int} + \partial^i
 * u_\mathrm{ext}\right) - \sigma \left(u_\mathrm{int} -
 * u_\mathrm{int}\right)
 * n_i F^{*i}_{v^j} = \frac{1}{2} n^j \left(u_\mathrm{int} +
 * u_\mathrm{ext}\right) \f}
 *
 * The penalty factor \f$\sigma\f$ is responsible for removing zero eigenmodes
 * and impacts the conditioning of the linear operator to be solved. It can be
 * chosen as \f$\sigma=C\frac{N_\mathrm{points}^2}{h}\f$ where
 * \f$N_\mathrm{points}\f$ is the number of collocation points (i.e. the
 * polynomial degree plus 1), \f$h\f$ is a measure of the element size in
 * inertial coordinates and \f$C\leq 1\f$ is a free parameter (see e.g. \cite
 * HesthavenWarburton, section 7.2).
 */
template <size_t Dim, typename FieldTagsList, typename AuxiliaryFieldTagsList>
struct FirstOrderInternalPenalty;

template <size_t Dim, typename... FieldTags, typename... AuxiliaryFieldTags>
struct FirstOrderInternalPenalty<Dim, tmpl::list<FieldTags...>,
                                 tmpl::list<AuxiliaryFieldTags...>> {
  // We should assert here that the FieldTags have one index less than their
  // respective AuxiliaryFieldTags

 public:
  struct PenaltyParameter {
    using type = double;
    // Currently this is used as the full prefactor to the penalty term. When it
    // becomes possible to compute a measure of the size $h$ of an element and
    // the number of collocation points $p$ on both sides of the mortar, this
    // should be changed to be just the parameter multiplying
    // $\frac{N_\mathrm{points}^2}{h}$.
    static constexpr OptionString help = {
        "The prefactor to the penalty term of the flux."};
  };
  using options = tmpl::list<PenaltyParameter>;
  static constexpr OptionString help = {
      "Computes the internal penalty flux for a Poisson system."};
  static std::string name() noexcept { return "InternalPenalty"; }

  FirstOrderInternalPenalty() = default;
  explicit FirstOrderInternalPenalty(double penalty_parameter)
      : penalty_parameter_(penalty_parameter) {}

  // clang-tidy: non-const reference
  void pup(PUP::er& p) noexcept { p | penalty_parameter_; }  // NOLINT

  struct FaceNormal : db::SimpleTag {
    using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
    static std::string name() noexcept { return "FaceNormal"; }
  };

  template <typename Tag>
  struct NormalDotDivFlux : db::PrefixTag, db::SimpleTag {
    using tag = Tag;
    using type = TensorMetafunctions::remove_first_index<typename Tag::type>;
    static std::string name() noexcept {
      return "NormalDotDivFlux(" + Tag::name() + ")";
    }
  };

  // These tags are sliced to the interface of the element and passed to
  // `package_data` to provide the data needed to compute the numerical fluxes.
  using argument_tags =
      tmpl::list<Tags::NormalDotFlux<AuxiliaryFieldTags>...,
                 Tags::div<Tags::Flux<AuxiliaryFieldTags, tmpl::size_t<Dim>,
                                      Frame::Inertial>...>,
                 Tags::Normalized<Tags::UnnormalizedFaceNormal<Dim>>>;

  // This is the data needed to compute the numerical flux.
  // `SendBoundaryFluxes` calls `package_data` to store these tags in a
  // Variables. Local and remote values of this data are then combined in the
  // `()` operator.
  using package_tags =
      tmpl::list<Tags::NormalDotFlux<AuxiliaryFieldTags>...,
                 NormalDotDivFlux<AuxiliaryFieldTags>..., FaceNormal>;

  // Following the packaged_data pointer, this function expects as arguments the
  // types in `argument_tags`.
  void package_data(
      const gsl::not_null<Variables<package_tags>*> packaged_data,
      const db::item_type<Tags::NormalDotFlux<
          AuxiliaryFieldTags>>&... normal_dot_auxiliary_field_fluxes,
      const db::item_type<Tags::div<
          Tags::Flux<AuxiliaryFieldTags, tmpl::size_t<Dim>,
                     Frame::Inertial>>>&... div_auxiliary_field_fluxes,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& interface_unit_normal)
      const noexcept {
    const auto helper = [&packaged_data, &interface_unit_normal ](
        const auto auxiliary_field_tag_v,
        const auto normal_dot_auxiliary_field_flux,
        const auto div_auxiliary_field_flux) noexcept {
      using auxiliary_field_tag = std::decay_t<decltype(auxiliary_field_tag_v)>;
      get<Tags::NormalDotFlux<auxiliary_field_tag>>(*packaged_data) =
          normal_dot_auxiliary_field_flux;
      get(get<NormalDotDivFlux<auxiliary_field_tag>>(*packaged_data)) =
          get<0>(interface_unit_normal) * get<0>(div_auxiliary_field_flux);
      for (size_t d = 1; d < Dim; d++) {
        get(get<NormalDotDivFlux<auxiliary_field_tag>>(*packaged_data)) +=
            interface_unit_normal.get(d) * div_auxiliary_field_flux.get(d);
      }
    };
    EXPAND_PACK_LEFT_TO_RIGHT(helper(AuxiliaryFieldTags{},
                                     normal_dot_auxiliary_field_fluxes,
                                     div_auxiliary_field_fluxes));
    get<FaceNormal>(*packaged_data) = interface_unit_normal;
  }

  // This function combines local and remote data to the numerical fluxes.
  // The numerical fluxes as not-null pointers are the first arguments. The
  // other arguments are the packaged types for the interior side followed by
  // the packaged types for the exterior side.
  void operator()(
      const gsl::not_null<db::item_type<Tags::NormalDotNumericalFlux<
          FieldTags>>*>... numerical_flux_for_fields,
      const gsl::not_null<db::item_type<Tags::NormalDotNumericalFlux<
          AuxiliaryFieldTags>>*>... numerical_flux_for_auxiliary_fields,
      const db::item_type<Tags::NormalDotFlux<
          AuxiliaryFieldTags>>&... normal_dot_auxiliary_flux_interiors,
      const db::item_type<NormalDotDivFlux<
          AuxiliaryFieldTags>>&... normal_dot_div_auxiliary_flux_interiors,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          interface_unit_normal_interior,
      const db::item_type<Tags::NormalDotFlux<
          AuxiliaryFieldTags>>&... minus_normal_dot_auxiliary_flux_exteriors,
      const db::item_type<NormalDotDivFlux<
          AuxiliaryFieldTags>>&... minus_normal_dot_div_aux_flux_exteriors,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          interface_unit_normal_exterior) const noexcept {
    // Need polynomial degress and element size to compute this dynamically
    const double penalty = penalty_parameter_;

    const auto helper =
        [
          &interface_unit_normal_interior, &interface_unit_normal_exterior,
          &penalty
        ](const auto numerical_flux_for_field,
          const auto numerical_flux_for_auxiliary_field,
          const auto normal_dot_auxiliary_flux_interior,
          const auto normal_dot_div_auxiliary_flux_interior,
          const auto minus_normal_dot_auxiliary_flux_exterior,
          const auto minus_normal_dot_div_aux_flux_exterior) noexcept {
      for (size_t d = 0; d < Dim; d++) {
        numerical_flux_for_auxiliary_field->get(d) =
            0.5 * (normal_dot_auxiliary_flux_interior.get(d) -
                   minus_normal_dot_auxiliary_flux_exterior.get(d));
      }
      DataVector jump_normal_dot_auxiliary_field_flux =
          get<0>(interface_unit_normal_interior) *
              get<0>(normal_dot_auxiliary_flux_interior) -
          get<0>(interface_unit_normal_exterior) *
              get<0>(minus_normal_dot_auxiliary_flux_exterior);
      for (size_t d = 1; d < Dim; d++) {
        jump_normal_dot_auxiliary_field_flux +=
            interface_unit_normal_interior.get(d) *
                normal_dot_auxiliary_flux_interior.get(d) -
            interface_unit_normal_exterior.get(d) *
                minus_normal_dot_auxiliary_flux_exterior.get(d);
      }
      get(*numerical_flux_for_field) =
          0.5 * (get(normal_dot_div_auxiliary_flux_interior) -
                 get(minus_normal_dot_div_aux_flux_exterior)) -
          penalty * jump_normal_dot_auxiliary_field_flux;
    };
    EXPAND_PACK_LEFT_TO_RIGHT(helper(numerical_flux_for_fields,
                                     numerical_flux_for_auxiliary_fields,
                                     normal_dot_auxiliary_flux_interiors,
                                     normal_dot_div_auxiliary_flux_interiors,
                                     minus_normal_dot_auxiliary_flux_exteriors,
                                     minus_normal_dot_div_aux_flux_exteriors));
  }

  // This function computes the boundary contributions from Dirichlet boundary
  // conditions. This data is what remains to be added to the boundaries when
  // homogeneous (i.e. zero) boundary conditions are assumed in the calculation
  // of the numerical fluxes, but we wish to impose inhomogeneous (i.e. nonzero)
  // boundary conditions. Since this contribution does not depend on the
  // numerical field values, but only on the Dirichlet boundary data, it may be
  // added as contribution to the source of the elliptic systems. Then, it
  // remains to solve the homogeneous problem with the modified source.
  // The first arguments to this function are the boundary contributions to
  // compute as not-null pointers, in the order they appear in the
  // `system::fields_tag`. They are followed by the field values of the tags in
  // `system::impose_boundary_conditions_on_fields`. The last argument is the
  // normalized unit covector to the element face.
  void compute_dirichlet_boundary(
      const gsl::not_null<db::item_type<Tags::NormalDotNumericalFlux<
          FieldTags>>*>... numerical_flux_for_fields,
      const gsl::not_null<db::item_type<Tags::NormalDotNumericalFlux<
          AuxiliaryFieldTags>>*>... numerical_flux_for_auxiliary_fields,
      const db::item_type<Tags::NormalDotFlux<
          FieldTags>>&... /*normal_dot_dirichlet_field_fluxes*/,
      const db::item_type<Tags::NormalDotFlux<
          AuxiliaryFieldTags>>&... normal_dot_dirichlet_auxiliary_field_fluxes,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& interface_unit_normal)
      const noexcept {
    // Need polynomial degress and element size to compute this dynamically
    const double penalty = penalty_parameter_;

    const auto helper = [&interface_unit_normal, &penalty ](
        const auto numerical_flux_for_field,
        const auto numerical_flux_for_auxiliary_field,
        const auto normal_dot_dirichlet_auxiliary_field_flux) noexcept {
      *numerical_flux_for_auxiliary_field =
          normal_dot_dirichlet_auxiliary_field_flux;
      get(*numerical_flux_for_field) =
          get<0>(interface_unit_normal) *
          get<0>(normal_dot_dirichlet_auxiliary_field_flux);
      for (size_t d = 1; d < Dim; d++) {
        get(*numerical_flux_for_field) +=
            interface_unit_normal.get(d) *
            normal_dot_dirichlet_auxiliary_field_flux.get(d);
      }
      get(*numerical_flux_for_field) *= 2 * penalty;
    };
    EXPAND_PACK_LEFT_TO_RIGHT(
        helper(numerical_flux_for_fields, numerical_flux_for_auxiliary_fields,
               normal_dot_dirichlet_auxiliary_field_fluxes));
  }

 private:
  double penalty_parameter_{};
};

}  // namespace NumericalFluxes
}  // namespace dg
