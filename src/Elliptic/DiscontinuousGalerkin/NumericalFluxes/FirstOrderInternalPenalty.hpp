// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Metafunctions.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "Options/Options.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

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

namespace elliptic {
namespace dg {
namespace NumericalFluxes {

/*!
 * \brief The internal penalty flux for first-order elliptic equations.
 *
 * \details Computes the internal penalty numerical flux (see e.g.
 * \cite HesthavenWarburton, section 7.2) dotted with the interface unit normal.
 *
 * We implement here a suggested generalization of the internal penalty flux
 * for any set of elliptic PDEs. It is designed for fluxes (i.e. principal parts
 * of the PDEs) that may depend on the dynamic fields. This is the case for the
 * velocity potential equation of binary neutron stars, for example.
 *
 * Formulating the elliptic PDEs in terms of a first-order _auxiliary_ variable
 * \f$v\f$ for each _primal_ variable \f$u\f$, they take the _flux-form_
 *
 * \f{align}
 * -\partial_i F^i_A + S_A = f_A
 * \f}
 *
 * where the fluxes and sources are indexed by the variables and we have
 * defined \f$f_A\f$ as those sources that are independent of the variables.
 * We now choose the internal penalty numerical fluxes \f$F^{*i}\f$ as follows
 * for each variable \f$u\f$ and their corresponding auxiliary variable \f$v\f$:
 *
 * \f{align}
 * n_i F^{*i}_u &= \frac{1}{2} n_i \left( F^i_u(\partial_j
 * F^j_v(u^\mathrm{int})) + F^i_u(\partial_j F^j_v(u^\mathrm{ext}))
 * \right) - \sigma n_i \left(F^i_u(n\otimes u^\mathrm{int}) - F^i_u(
 * n\otimes u^\mathrm{ext})\right) \\
 * n_i F^{*i}_v &= \frac{1}{2} n_i \left(F^i_v(u^\mathrm{int}) +
 * F^i_v(u^\mathrm{ext})\right)
 * \f}
 *
 * Note that we have assumed \f$n^\mathrm{ext}_i=-n_i\f$ here, i.e. face normals
 * don't depend on the dynamic variables (which may be discontinuous on element
 * faces). This is the case for the problems we are expecting to solve, because
 * those will be on fixed background metrics (e.g. a conformal metric for the
 * XCTS system).
 *
 * Also note that the numerical fluxes intentionally don't depend on the
 * auxiliary field values \f$v\f$. This property allows us to use the numerical
 * fluxes also for the second-order (or _primal_) DG formulation, where we
 * remove the need for an auxiliary variable. For the first-order system we
 * could replace the divergence in \f$F^{*i}_u\f$ with \f$v\f$, which would
 * result in a generalized _stabilized central flux_. It resembles the
 * Local-Lax-Friedrichs flux for evolution systems but remains consistent in the
 * stabilizer term even for principal parts that depend on dynamic variables.
 *
 * For a Poisson system \f$-\Delta u(x) = f(x)\f$ with auxiliary variable
 * \f$v_i=\partial_i u\f$ this numeric flux reduces to the standard internal
 * penalty flux (see e.g. \cite HesthavenWarburton, section 7.2, or
 * \cite Arnold2002)
 *
 * \f{align}
 * n_i F^{*i}_u = n_i v_i^* = \frac{1}{2} n_i \left(\partial_i u^\mathrm{int} +
 * \partial_i u^\mathrm{ext}\right) - \sigma \left(u^\mathrm{int} -
 * u^\mathrm{ext}\right)
 * n_i F^{*i}_{v_j} = n_j u^* = \frac{1}{2} n_j \left(u^\mathrm{int} +
 * u^\mathrm{ext}\right)
 * \f}
 *
 * where a sum over repeated indices is assumed, since the equation is
 * formulated on a Euclidean geometry.
 *
 * This generalization of the internal penalty flux is based on unpublished work
 * by Nils L. Fischer (nils.fischer@aei.mpg.de).
 *
 * The penalty factor \f$\sigma\f$ is responsible for removing zero eigenmodes
 * and impacts the conditioning of the linear operator to be solved. It can be
 * chosen as \f$\sigma=C\frac{N_\mathrm{points}^2}{h}\f$ where
 * \f$N_\mathrm{points}\f$ is the number of collocation points (i.e. the
 * polynomial degree plus 1), \f$h\f$ is a measure of the element size in
 * inertial coordinates and \f$C\leq 1\f$ is a free parameter (see e.g.
 * \cite HesthavenWarburton, section 7.2).
 */
template <size_t Dim, typename FluxesComputerTag, typename FieldTagsList,
          typename AuxiliaryFieldTagsList,
          typename FluxesComputer = db::item_type<FluxesComputerTag>,
          typename FluxesArgs = typename FluxesComputer::argument_tags>
struct FirstOrderInternalPenalty;

template <size_t Dim, typename FluxesComputerTag, typename... FieldTags,
          typename... AuxiliaryFieldTags, typename FluxesComputer,
          typename... FluxesArgs>
struct FirstOrderInternalPenalty<Dim, FluxesComputerTag,
                                 tmpl::list<FieldTags...>,
                                 tmpl::list<AuxiliaryFieldTags...>,
                                 FluxesComputer, tmpl::list<FluxesArgs...>> {
 private:
  using fluxes_computer_tag = FluxesComputerTag;

  template <typename Tag>
  struct NormalDotDivFlux : db::PrefixTag, db::SimpleTag {
    using tag = Tag;
    using type = TensorMetafunctions::remove_first_index<typename Tag::type>;
    static std::string name() noexcept {
      return "NormalDotDivFlux(" + Tag::name() + ")";
    }
  };

  template <typename Tag>
  struct NormalDotNormalDotFlux : db::PrefixTag, db::SimpleTag {
    using tag = Tag;
    using type = TensorMetafunctions::remove_first_index<typename Tag::type>;
    static std::string name() noexcept {
      return "NormalDotNormalDotFlux(" + Tag::name() + ")";
    }
  };

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
      "The internal penalty flux for elliptic systems."};
  static std::string name() noexcept { return "InternalPenalty"; }

  FirstOrderInternalPenalty() = default;
  explicit FirstOrderInternalPenalty(double penalty_parameter)
      : penalty_parameter_(penalty_parameter) {}

  // clang-tidy: non-const reference
  void pup(PUP::er& p) noexcept { p | penalty_parameter_; }  // NOLINT

  // These tags are sliced to the interface of the element and passed to
  // `package_data` to provide the data needed to compute the numerical fluxes.
  using argument_tags =
      tmpl::list<::Tags::NormalDotFlux<AuxiliaryFieldTags>...,
                 ::Tags::div<::Tags::Flux<AuxiliaryFieldTags, tmpl::size_t<Dim>,
                                          Frame::Inertial>>...,
                 fluxes_computer_tag, FluxesArgs...,
                 ::Tags::Normalized<::Tags::UnnormalizedFaceNormal<Dim>>>;
  using volume_tags = tmpl::list<fluxes_computer_tag>;

  // This is the data needed to compute the numerical flux.
  // `SendBoundaryFluxes` calls `package_data` to store these tags in a
  // Variables. Local and remote values of this data are then combined in the
  // `()` operator.
  using package_tags =
      tmpl::list<::Tags::NormalDotFlux<AuxiliaryFieldTags>...,
                 NormalDotDivFlux<AuxiliaryFieldTags>...,
                 NormalDotNormalDotFlux<AuxiliaryFieldTags>...>;

  // Following the packaged_data pointer, this function expects as arguments the
  // types in `argument_tags`.
  void package_data(
      const gsl::not_null<Variables<package_tags>*> packaged_data,
      const db::item_type<::Tags::NormalDotFlux<
          AuxiliaryFieldTags>>&... normal_dot_auxiliary_field_fluxes,
      const db::item_type<::Tags::div<
          ::Tags::Flux<AuxiliaryFieldTags, tmpl::size_t<Dim>,
                       Frame::Inertial>>>&... div_auxiliary_field_fluxes,
      const FluxesComputer& fluxes_computer,
      const db::item_type<FluxesArgs>&... fluxes_args,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& interface_unit_normal)
      const noexcept {
    auto principal_div_aux_field_fluxes = make_with_value<Variables<tmpl::list<
        ::Tags::Flux<FieldTags, tmpl::size_t<Dim>, Frame::Inertial>...>>>(
        interface_unit_normal, 0.);
    auto principal_ndot_aux_field_fluxes = make_with_value<Variables<tmpl::list<
        ::Tags::Flux<FieldTags, tmpl::size_t<Dim>, Frame::Inertial>...>>>(
        interface_unit_normal, 0.);
    fluxes_computer.apply(
        make_not_null(
            &get<::Tags::Flux<FieldTags, tmpl::size_t<Dim>, Frame::Inertial>>(
                principal_div_aux_field_fluxes))...,
        fluxes_args..., div_auxiliary_field_fluxes...);
    fluxes_computer.apply(
        make_not_null(
            &get<::Tags::Flux<FieldTags, tmpl::size_t<Dim>, Frame::Inertial>>(
                principal_ndot_aux_field_fluxes))...,
        fluxes_args..., normal_dot_auxiliary_field_fluxes...);
    const auto helper =
        [
          &packaged_data, &principal_div_aux_field_fluxes,
          &principal_ndot_aux_field_fluxes, &interface_unit_normal
        ](const auto field_tag_v, const auto auxiliary_field_tag_v,
          const auto normal_dot_auxiliary_field_flux) noexcept {
      using field_tag = std::decay_t<decltype(field_tag_v)>;
      using auxiliary_field_tag = std::decay_t<decltype(auxiliary_field_tag_v)>;
      // Compute n.F_v(u)
      get<::Tags::NormalDotFlux<auxiliary_field_tag>>(*packaged_data) =
          normal_dot_auxiliary_field_flux;
      // Compute n.F_u(div(F_v(u))) and n.F_u(n.F_v(u))
      normal_dot_flux(
          make_not_null(
              &get<NormalDotDivFlux<auxiliary_field_tag>>(*packaged_data)),
          interface_unit_normal,
          get<::Tags::Flux<field_tag, tmpl::size_t<Dim>, Frame::Inertial>>(
              principal_div_aux_field_fluxes));
      normal_dot_flux(
          make_not_null(&get<NormalDotNormalDotFlux<auxiliary_field_tag>>(
              *packaged_data)),
          interface_unit_normal,
          get<::Tags::Flux<field_tag, tmpl::size_t<Dim>, Frame::Inertial>>(
              principal_ndot_aux_field_fluxes));
    };
    EXPAND_PACK_LEFT_TO_RIGHT(helper(FieldTags{}, AuxiliaryFieldTags{},
                                     normal_dot_auxiliary_field_fluxes));
  }

  // This function combines local and remote data to the numerical fluxes.
  // The numerical fluxes as not-null pointers are the first arguments. The
  // other arguments are the packaged types for the interior side followed by
  // the packaged types for the exterior side.
  void operator()(
      const gsl::not_null<db::item_type<::Tags::NormalDotNumericalFlux<
          FieldTags>>*>... numerical_flux_for_fields,
      const gsl::not_null<db::item_type<::Tags::NormalDotNumericalFlux<
          AuxiliaryFieldTags>>*>... numerical_flux_for_auxiliary_fields,
      const db::item_type<::Tags::NormalDotFlux<
          AuxiliaryFieldTags>>&... normal_dot_auxiliary_flux_interiors,
      const db::item_type<NormalDotDivFlux<
          AuxiliaryFieldTags>>&... normal_dot_div_auxiliary_flux_interiors,
      const db::item_type<NormalDotNormalDotFlux<
          AuxiliaryFieldTags>>&... ndot_ndot_aux_flux_interiors,
      const db::item_type<::Tags::NormalDotFlux<
          AuxiliaryFieldTags>>&... minus_normal_dot_auxiliary_flux_exteriors,
      const db::item_type<NormalDotDivFlux<
          AuxiliaryFieldTags>>&... minus_normal_dot_div_aux_flux_exteriors,
      const db::item_type<NormalDotDivFlux<
          AuxiliaryFieldTags>>&... ndot_ndot_aux_flux_exteriors) const
      noexcept {
    // Need polynomial degress and element size to compute this dynamically
    const double penalty = penalty_parameter_;

    const auto helper = [&penalty](
        const auto numerical_flux_for_field,
        const auto numerical_flux_for_auxiliary_field,
        const auto normal_dot_auxiliary_flux_interior,
        const auto normal_dot_div_auxiliary_flux_interior,
        const auto ndot_ndot_aux_flux_interior,
        const auto minus_normal_dot_auxiliary_flux_exterior,
        const auto minus_normal_dot_div_aux_flux_exterior,
        const auto ndot_ndot_aux_flux_exterior) noexcept {
      for (auto it = numerical_flux_for_auxiliary_field->begin();
           it != numerical_flux_for_auxiliary_field->end(); it++) {
        const auto index =
            numerical_flux_for_auxiliary_field->get_tensor_index(it);
        *it = 0.5 * (normal_dot_auxiliary_flux_interior.get(index) -
                     minus_normal_dot_auxiliary_flux_exterior.get(index));
      }
      for (auto it = numerical_flux_for_field->begin();
           it != numerical_flux_for_field->end(); it++) {
        const auto index = numerical_flux_for_field->get_tensor_index(it);
        *it = 0.5 * (normal_dot_div_auxiliary_flux_interior.get(index) -
                     minus_normal_dot_div_aux_flux_exterior.get(index)) -
              penalty * (ndot_ndot_aux_flux_interior.get(index) -
                         ndot_ndot_aux_flux_exterior.get(index));
      }
    };
    EXPAND_PACK_LEFT_TO_RIGHT(helper(
        numerical_flux_for_fields, numerical_flux_for_auxiliary_fields,
        normal_dot_auxiliary_flux_interiors,
        normal_dot_div_auxiliary_flux_interiors, ndot_ndot_aux_flux_interiors,
        minus_normal_dot_auxiliary_flux_exteriors,
        minus_normal_dot_div_aux_flux_exteriors, ndot_ndot_aux_flux_exteriors));
  }

  // This function computes the boundary contributions from Dirichlet boundary
  // conditions. This data is what remains to be added to the boundaries when
  // homogeneous (i.e. zero) boundary conditions are assumed in the calculation
  // of the numerical fluxes, but we wish to impose inhomogeneous (i.e. nonzero)
  // boundary conditions. Since this contribution does not depend on the
  // numerical field values, but only on the Dirichlet boundary data, it may be
  // added as contribution to the source of the elliptic systems. Then, it
  // remains to solve the homogeneous problem with the modified source.
  void compute_dirichlet_boundary(
      const gsl::not_null<db::item_type<::Tags::NormalDotNumericalFlux<
          FieldTags>>*>... numerical_flux_for_fields,
      const gsl::not_null<db::item_type<::Tags::NormalDotNumericalFlux<
          AuxiliaryFieldTags>>*>... numerical_flux_for_auxiliary_fields,
      const db::item_type<FieldTags>&... dirichlet_fields,
      const FluxesComputer& fluxes_computer,
      const db::item_type<FluxesArgs>&... fluxes_args,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& interface_unit_normal)
      const noexcept {
    // Need polynomial degress and element size to compute this dynamically
    const double penalty = penalty_parameter_;

    // Compute n.F_v(u)
    auto dirichlet_auxiliary_field_fluxes =
        make_with_value<Variables<tmpl::list<::Tags::Flux<
            AuxiliaryFieldTags, tmpl::size_t<Dim>, Frame::Inertial>...>>>(
            interface_unit_normal, 0.);
    fluxes_computer.apply(
        make_not_null(&get<::Tags::Flux<AuxiliaryFieldTags, tmpl::size_t<Dim>,
                                        Frame::Inertial>>(
            dirichlet_auxiliary_field_fluxes))...,
        fluxes_args..., dirichlet_fields...);
    const auto helper =
        [&interface_unit_normal, &dirichlet_auxiliary_field_fluxes ](
            const auto auxiliary_field_tag_v,
            const auto numerical_flux_for_auxiliary_field) noexcept {
      using auxiliary_field_tag = std::decay_t<decltype(auxiliary_field_tag_v)>;
      normal_dot_flux(
          numerical_flux_for_auxiliary_field, interface_unit_normal,
          get<::Tags::Flux<auxiliary_field_tag, tmpl::size_t<Dim>,
                           Frame::Inertial>>(dirichlet_auxiliary_field_fluxes));
    };
    EXPAND_PACK_LEFT_TO_RIGHT(
        helper(AuxiliaryFieldTags{}, numerical_flux_for_auxiliary_fields));

    // Compute 2 * sigma * n.F_u(n.F_v(u))
    auto principal_dirichlet_auxiliary_field_fluxes =
        make_with_value<Variables<tmpl::list<
            ::Tags::Flux<FieldTags, tmpl::size_t<Dim>, Frame::Inertial>...>>>(
            interface_unit_normal, 0.);
    fluxes_computer.apply(
        make_not_null(
            &get<::Tags::Flux<FieldTags, tmpl::size_t<Dim>, Frame::Inertial>>(
                principal_dirichlet_auxiliary_field_fluxes))...,
        fluxes_args..., *numerical_flux_for_auxiliary_fields...);
    const auto helper2 = [
      &interface_unit_normal, &penalty,
      &principal_dirichlet_auxiliary_field_fluxes
    ](const auto field_tag_v, const auto numerical_flux_for_field) noexcept {
      using field_tag = std::decay_t<decltype(field_tag_v)>;
      normal_dot_flux(
          numerical_flux_for_field, interface_unit_normal,
          get<::Tags::Flux<field_tag, tmpl::size_t<Dim>, Frame::Inertial>>(
              principal_dirichlet_auxiliary_field_fluxes));
      for (auto it = numerical_flux_for_field->begin();
           it != numerical_flux_for_field->end(); it++) {
        *it *= 2 * penalty;
      }
    };
    EXPAND_PACK_LEFT_TO_RIGHT(helper2(FieldTags{}, numerical_flux_for_fields));
  }

 private:
  double penalty_parameter_{};
};

}  // namespace NumericalFluxes
}  // namespace dg
}  // namespace elliptic
