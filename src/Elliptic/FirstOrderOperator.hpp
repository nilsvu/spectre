// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic {

namespace first_order_operator_detail {

template <size_t Dim, typename PrimalFields, typename AuxiliaryFields>
struct FirstOrderFluxesImpl;

template <size_t Dim, typename... PrimalFields, typename... AuxiliaryFields>
struct FirstOrderFluxesImpl<Dim, tmpl::list<PrimalFields...>,
                            tmpl::list<AuxiliaryFields...>> {
  template <typename VarsTags, typename FluxesComputer, typename... FluxesArgs>
  static constexpr void apply(
      const gsl::not_null<Variables<db::wrap_tags_in<
          ::Tags::Flux, VarsTags, tmpl::size_t<Dim>, Frame::Inertial>>*>
          fluxes,
      const Variables<VarsTags>& vars, const FluxesComputer& fluxes_computer,
      const FluxesArgs&... fluxes_args) noexcept {
    // Compute fluxes for primal fields
    fluxes_computer.apply(
        make_not_null(&get<::Tags::Flux<PrimalFields, tmpl::size_t<Dim>,
                                        Frame::Inertial>>(*fluxes))...,
        fluxes_args..., get<AuxiliaryFields>(vars)...);
    // Compute fluxes for auxiliary fields
    fluxes_computer.apply(
        make_not_null(&get<::Tags::Flux<AuxiliaryFields, tmpl::size_t<Dim>,
                                        Frame::Inertial>>(*fluxes))...,
        fluxes_args..., get<PrimalFields>(vars)...);
  }
};

template <size_t Dim, typename PrimalFields, typename AuxiliaryFields,
          typename SourcesComputer>
struct FirstOrderSourcesImpl;

template <size_t Dim, typename... PrimalFields, typename... AuxiliaryFields,
          typename SourcesComputer>
struct FirstOrderSourcesImpl<Dim, tmpl::list<PrimalFields...>,
                             tmpl::list<AuxiliaryFields...>, SourcesComputer> {
  template <typename VarsTags, typename... SourcesArgs>
  static constexpr void apply(
      const gsl::not_null<
          Variables<db::wrap_tags_in<::Tags::Source, VarsTags>>*>
          sources,
      const Variables<VarsTags>& vars,
      const Variables<db::wrap_tags_in<
          ::Tags::Flux, VarsTags, tmpl::size_t<Dim>, Frame::Inertial>>& fluxes,
      const SourcesArgs&... sources_args) noexcept {
    // Set auxiliary field sources to the auxiliary field values to begin with.
    // This is the standard choice, since the auxiliary equations define the
    // auxiliary variables. Source computers can adjust or add to these sources.
    tmpl::for_each<tmpl::list<AuxiliaryFields...>>(
        [&sources, &vars](const auto auxiliary_field_tag_v) noexcept {
          using auxiliary_field_tag =
              tmpl::type_from<decltype(auxiliary_field_tag_v)>;
          get<::Tags::Source<auxiliary_field_tag>>(*sources) =
              get<auxiliary_field_tag>(vars);
        });
    // Call into the sources computer to set primal field sources and possibly
    // adjust auxiliary field sources. The sources depend on the primal and the
    // auxiliary variables. However, we pass the volume fluxes instead of the
    // auxiliary variables to the source computer as an optimization so they
    // don't have to be re-computed.
    SourcesComputer::apply(
        make_not_null(&get<::Tags::Source<PrimalFields>>(*sources))...,
        make_not_null(&get<::Tags::Source<AuxiliaryFields>>(*sources))...,
        sources_args..., get<PrimalFields>(vars)...,
        get<::Tags::Flux<PrimalFields, tmpl::size_t<Dim>, Frame::Inertial>>(
            fluxes)...);
  }
};

}  // namespace first_order_operator_detail

// @{
/*!
 * \brief Compute the fluxes \f$F^i(u)\f$ for the first-order formulation of
 * elliptic systems.
 *
 * \see `elliptic::first_order_operator`
 */
template <size_t Dim, typename PrimalFields, typename AuxiliaryFields,
          typename VarsTags, typename FluxesComputer, typename... FluxesArgs>
void first_order_fluxes(
    gsl::not_null<Variables<db::wrap_tags_in<
        ::Tags::Flux, VarsTags, tmpl::size_t<Dim>, Frame::Inertial>>*>
        fluxes,
    const Variables<VarsTags>& vars, const FluxesComputer& fluxes_computer,
    const FluxesArgs&... fluxes_args) noexcept {
  // Resize result variables
  if (UNLIKELY(fluxes->number_of_grid_points() !=
               vars.number_of_grid_points())) {
    fluxes->initialize(vars.number_of_grid_points());
  }
  first_order_operator_detail::FirstOrderFluxesImpl<
      Dim, PrimalFields, AuxiliaryFields>::apply(std::move(fluxes), vars,
                                                 fluxes_computer,
                                                 fluxes_args...);
}

template <size_t Dim, typename PrimalFields, typename AuxiliaryFields,
          typename VarsTags, typename FluxesComputer, typename... FluxesArgs>
auto first_order_fluxes(const Variables<VarsTags>& vars,
                        const FluxesComputer& fluxes_computer,
                        const FluxesArgs&... fluxes_args) noexcept {
  Variables<db::wrap_tags_in<::Tags::Flux, VarsTags, tmpl::size_t<Dim>,
                             Frame::Inertial>>
      fluxes{vars.number_of_grid_points()};
  first_order_operator_detail::FirstOrderFluxesImpl<
      Dim, PrimalFields, AuxiliaryFields>::apply(make_not_null(&fluxes), vars,
                                                 fluxes_computer,
                                                 fluxes_args...);
  return fluxes;
}
// @}

// @{
/*!
 * \brief Compute the sources \f$S(u)\f$ for the first-order formulation of
 * elliptic systems.
 *
 * This function takes the `fluxes` as an argument in addition to the variables
 * as an optimization. The fluxes will generally be computed before the sources
 * anyway, so we pass them to the source computers to avoid having to re-compute
 * them for source-terms that have the same form as the fluxes.
 *
 * \see `elliptic::first_order_operator`
 */
template <size_t Dim, typename PrimalFields, typename AuxiliaryFields,
          typename SourcesComputer, typename VarsTags, typename... SourcesArgs>
void first_order_sources(
    gsl::not_null<Variables<db::wrap_tags_in<::Tags::Source, VarsTags>>*>
        sources,
    const Variables<VarsTags>& vars,
    const Variables<db::wrap_tags_in<::Tags::Flux, VarsTags, tmpl::size_t<Dim>,
                                     Frame::Inertial>>& fluxes,
    const SourcesArgs&... sources_args) noexcept {
  if (UNLIKELY(sources->number_of_grid_points() !=
               vars.number_of_grid_points())) {
    sources->initialize(vars.number_of_grid_points());
  }
  first_order_operator_detail::FirstOrderSourcesImpl<
      Dim, PrimalFields, AuxiliaryFields,
      SourcesComputer>::apply(std::move(sources), vars, fluxes,
                              sources_args...);
}

template <size_t Dim, typename PrimalFields, typename AuxiliaryFields,
          typename SourcesComputer, typename VarsTags, typename... SourcesArgs>
auto first_order_sources(
    const Variables<VarsTags>& vars,
    const Variables<db::wrap_tags_in<::Tags::Flux, VarsTags, tmpl::size_t<Dim>,
                                     Frame::Inertial>>& fluxes,
    const SourcesArgs&... sources_args) noexcept {
  Variables<db::wrap_tags_in<::Tags::Source, VarsTags>> sources{
      vars.number_of_grid_points()};
  first_order_operator_detail::FirstOrderSourcesImpl<
      Dim, PrimalFields, AuxiliaryFields,
      SourcesComputer>::apply(make_not_null(&sources), vars, fluxes,
                              sources_args...);
  return sources;
}
// @}

//@{
/*!
 * \brief Compute the bulk contribution to the operator represented by the
 * `OperatorTags`.
 *
 * This function computes \f$A(u)=-\partial_i F^i(u) + S(u)\f$, where \f$F^i\f$
 * and \f$S\f$ are the fluxes and sources of the system of first-order PDEs,
 * respectively. They are defined such that \f$A(u(x))=f(x)\f$ is the full
 * system of equations, with \f$f(x)\f$ representing sources that are
 * independent of the variables \f$u\f$. In a DG setting, boundary contributions
 * can be added to \f$A(u)\f$ to build the full DG operator.
 */
template <typename OperatorTags, typename DivFluxesTags, typename SourcesTags>
void first_order_operator(
    const gsl::not_null<Variables<OperatorTags>*> operator_applied_to_vars,
    const Variables<DivFluxesTags>& div_fluxes,
    const Variables<SourcesTags>& sources) noexcept {
  *operator_applied_to_vars = sources - div_fluxes;
}

// template <typename OperatorTags, typename DivFluxesTags, typename
// SourcesTags,
//           size_t Dim>
// void first_order_operator_massive(
//     const gsl::not_null<Variables<OperatorTags>*> operator_applied_to_vars,
//     const Variables<DivFluxesTags>& div_fluxes,
//     const Variables<SourcesTags>& sources, const Mesh<Dim>& mesh,
//     const Scalar<DataVector>& det_jacobian) noexcept {
//   std::array<Matrix, Dim> mass_matrices{};
//   for (size_t d = 0; d < Dim; d++) {
//     gsl::at(mass_matrices, d) = Spectral::mass_matrix(mesh.slice_through(d));
//   }
//   if (UNLIKELY(operator_applied_to_vars->number_of_grid_points() !=
//                mesh.number_of_grid_points())) {
//     operator_applied_to_vars->initialize(mesh.number_of_grid_points());
//   }
//   apply_matrices(
//       operator_applied_to_vars, mass_matrices,
//       Variables<OperatorTags>(get(det_jacobian) *
//                               Variables<OperatorTags>(sources - div_fluxes)),
//       mesh.extents());
// }

template <typename PrimalFields, typename AuxiliaryFields,
          typename SourcesComputer, typename OperatorTags, typename FluxesTags,
          typename VarsTags, typename FluxesComputer, size_t Dim,
          typename... FluxesArgs, typename... SourcesArgs>
void first_order_operator(
    const gsl::not_null<Variables<OperatorTags>*> operator_applied_to_vars,
    const gsl::not_null<Variables<FluxesTags>*> fluxes,
    const gsl::not_null<Variables<db::wrap_tags_in<::Tags::div, FluxesTags>>*>
        div_fluxes,
    const Variables<VarsTags>& vars, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>&
        inv_jacobian,
    const FluxesComputer& fluxes_computer,
    const std::tuple<FluxesArgs...>& fluxes_args,
    const std::tuple<SourcesArgs...>& sources_args) noexcept {
  using all_fields = tmpl::append<PrimalFields, AuxiliaryFields>;
  // Compute (volume) fluxes
  std::apply(
      [&](const auto&... expanded_fluxes_args) {
        first_order_fluxes<Dim, PrimalFields, AuxiliaryFields>(
            fluxes, Variables<all_fields>(vars), fluxes_computer,
            expanded_fluxes_args...);
      },
      fluxes_args);
  // Compute divergence of volume fluxes
  divergence(div_fluxes, *fluxes, mesh, inv_jacobian);
  // Compute sources
  auto sources = std::apply(
      [&](const auto&... expanded_sources_args) {
        return first_order_sources<Dim, PrimalFields, AuxiliaryFields,
                                   SourcesComputer>(
            Variables<all_fields>(vars), *fluxes, expanded_sources_args...);
      },
      sources_args);
  // Forward to overload above that subtracts div(fluxes) and sources
  first_order_operator(operator_applied_to_vars, *div_fluxes,
                       std::move(sources));
}

// template <typename PrimalFields, typename AuxiliaryFields,
//           typename SourcesComputer, typename OperatorTags, typename
//           FluxesTags, typename VarsTags, typename FluxesComputer, size_t Dim,
//           typename... FluxesArgs, typename... SourcesArgs>
// void first_order_operator_massive(
//     const gsl::not_null<Variables<OperatorTags>*> operator_applied_to_vars,
//     const gsl::not_null<Variables<FluxesTags>*> fluxes,
//     const gsl::not_null<Variables<db::wrap_tags_in<::Tags::div,
//     FluxesTags>>*>
//         div_fluxes,
//     const Variables<VarsTags>& vars, const Mesh<Dim>& mesh,
//     const InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>&
//         inv_jacobian,
//     const FluxesComputer& fluxes_computer,
//     const std::tuple<FluxesArgs...>& fluxes_args,
//     const std::tuple<SourcesArgs...>& sources_args) noexcept {
//   using all_fields = tmpl::append<PrimalFields, AuxiliaryFields>;
//   // Compute (volume) fluxes
//   std::apply(
//       [&](const auto&... expanded_fluxes_args) {
//         first_order_fluxes<Dim, PrimalFields, AuxiliaryFields>(
//             fluxes, Variables<all_fields>(vars), fluxes_computer,
//             expanded_fluxes_args...);
//       },
//       fluxes_args);
//   // Compute divergence of volume fluxes
//   divergence(div_fluxes, *fluxes, mesh, inv_jacobian);
//   // Compute sources
//   auto sources = std::apply(
//       [&](const auto&... expanded_sources_args) {
//         return first_order_sources<Dim, PrimalFields, AuxiliaryFields,
//                                    SourcesComputer>(
//             Variables<all_fields>(vars), *fluxes, expanded_sources_args...);
//       },
//       sources_args);
//   // Forward to overload above that subtracts div(fluxes) and sources
//   first_order_operator(operator_applied_to_vars, *div_fluxes,
//                        std::move(sources));
// }
//@}

/*!
 * \brief Mutating DataBox invokable to compute the bulk contribution to the
 * operator represented by the `OperatorTag` applied to the `VarsTag`
 *
 * \see `elliptic::first_order_operator`
 *
 * We generally build the operator \f$A(u)\f$ to perform elliptic solver
 * iterations. This invokable can be used to build operators for both the linear
 * solver and the nonlinear solver iterations by providing the appropriate
 * `OperatorTag` and `VarsTag`. For example, the `OperatorTag` for the linear
 * solver would typically be `LinearSolver::Tags::OperatorAppliedTo`. It is the
 * elliptic equivalent to the time derivative in evolution schemes. Instead of
 * using it to evolve the variables, the iterative linear solver uses it to
 * compute a better approximation to the equation \f$A(u)=f\f$ (as detailed
 * above).
 *
 * With:
 * - `operator_tag` = `db::add_tag_prefix<OperatorTag, VarsTag>`
 * - `fluxes_tag` = `db::add_tag_prefix<::Tags::Flux, VarsTag,
 * tmpl::size_t<Dim>, Frame::Inertial>`
 * - `div_fluxes_tag` = `db::add_tag_prefix<Tags::div, fluxes_tag>`
 * - `sources_tag` = `db::add_tag_prefix<::Tags::Source, VarsTag>`
 *
 * Uses:
 * - DataBox:
 *   - `div_fluxes_tag`
 *   - `sources_tag`
 *
 * DataBox changes:
 * - Modifies:
 *   - `operator_tag`
 */
template <size_t Dim, template <typename> class OperatorTag, typename VarsTag,
          bool MassiveOperator>
struct FirstOrderOperator {
 private:
  using operator_tag = db::add_tag_prefix<OperatorTag, VarsTag>;
  using fluxes_tag = db::add_tag_prefix<::Tags::Flux, VarsTag,
                                        tmpl::size_t<Dim>, Frame::Inertial>;
  using div_fluxes_tag = db::add_tag_prefix<::Tags::div, fluxes_tag>;
  using sources_tag = db::add_tag_prefix<::Tags::Source, VarsTag>;

 public:
  using return_tags = tmpl::list<operator_tag>;
  using argument_tags =
      tmpl::list<div_fluxes_tag, sources_tag, domain::Tags::Mesh<Dim>,
                 domain::Tags::DetJacobian<Frame::Logical, Frame::Inertial>>;
  static void apply(
      const gsl::not_null<typename operator_tag::type*> operator_data,
      const typename div_fluxes_tag::type& div_fluxes,
      const typename sources_tag::type& sources, const Mesh<Dim>& mesh,
      const Scalar<DataVector>& det_jac) noexcept {
    first_order_operator(operator_data, div_fluxes, sources);
    if constexpr (MassiveOperator) {
      apply_mass(operator_data, mesh, det_jac);
    }
  }
};

template <size_t Dim, typename... PrimalFields, typename... AuxiliaryFields,
          typename BoundaryConditions, typename... BoundaryConditionsArgs,
          typename FluxesComputer, typename... FluxesArgs>
void impose_first_order_boundary_conditions(
    const gsl::not_null<
        Variables<tmpl::list<::Tags::NormalDotFlux<PrimalFields>...,
                             ::Tags::NormalDotFlux<AuxiliaryFields>...>>*>
        exterior_n_dot_fluxes,
    const gsl::not_null<Variables<tmpl::list<PrimalFields...>>*>
        primal_vars_buffer,
    const gsl::not_null<Variables<tmpl::list<
        ::Tags::Flux<AuxiliaryFields, tmpl::size_t<Dim>, Frame::Inertial>...>>*>
        auxiliary_fluxes_buffer,
    const Variables<tmpl::list<PrimalFields..., AuxiliaryFields...>>&
        interior_vars,
    const Variables<tmpl::list<::Tags::NormalDotFlux<PrimalFields>...,
                               ::Tags::NormalDotFlux<AuxiliaryFields>...>>&
        interior_n_dot_fluxes,
    const tnsr::i<DataVector, Dim>& exterior_face_normal,
    const BoundaryConditions& boundary_conditions,
    const std::tuple<BoundaryConditionsArgs...>& boundary_conditions_args,
    const FluxesComputer& fluxes_computer,
    const std::tuple<FluxesArgs...>& fluxes_args) noexcept {
  // Copy interior n.F over to the exterior. We invert the sign to account for
  // the inverted normal on exterior faces. Neumann-type boundary conditions
  // will directly modify the "primal" n.F in this return variable.
  *exterior_n_dot_fluxes = -interior_n_dot_fluxes;
  // Also copy interior vars over to the exterior. Dirichlet-type boundary
  // conditions will modify these so we can compute the "auxiliary" n.F in the
  // `exterior_n_dot_fluxes` return variable. The `primal_vars_buffer` and the
  // other memory buffers are not considered return variables.
  *primal_vars_buffer =
      interior_vars.extract_subset<tmpl::list<PrimalFields...>>();
  // Apply the boundary conditions
  std::apply(boundary_conditions,
             std::tuple_cat(
                 boundary_conditions_args,
                 std::make_tuple(
                     make_not_null(&get<PrimalFields>(*primal_vars_buffer))...,
                     make_not_null(&get<::Tags::NormalDotFlux<PrimalFields>>(
                         *exterior_n_dot_fluxes))...)));
  // Compute fluxes from the Dirichlet fields
  std::apply(
      [&fluxes_computer, &auxiliary_fluxes_buffer,
       &primal_vars_buffer](const auto&... expanded_fluxes_args) noexcept {
        fluxes_computer.apply(
            make_not_null(&get<::Tags::Flux<AuxiliaryFields, tmpl::size_t<Dim>,
                                            Frame::Inertial>>(
                *auxiliary_fluxes_buffer))...,
            expanded_fluxes_args..., get<PrimalFields>(*primal_vars_buffer)...);
      },
      fluxes_args);
  // Compute n.F from the Dirichlet fields
  tmpl::for_each<tmpl::list<AuxiliaryFields...>>(
      [&exterior_n_dot_fluxes, &auxiliary_fluxes_buffer,
       &exterior_face_normal](auto tag_v) noexcept {
        using aux_tag = tmpl::type_from<decltype(tag_v)>;
        normal_dot_flux(
            make_not_null(
                &get<::Tags::NormalDotFlux<aux_tag>>(*exterior_n_dot_fluxes)),
            exterior_face_normal,
            get<::Tags::Flux<aux_tag, tmpl::size_t<Dim>, Frame::Inertial>>(
                *auxiliary_fluxes_buffer));
      });
  // Impose boundary conditions through the fluxes, i.e. on the boundary flux
  // average:
  // n_i {{F^i}} = n_i (F^i_int + F^i_ext) / 2 = n_i F^i_boundary
  // => n_i F^i_ext = 2 * F^i_boundary - n_i F^i_int.
  // Just setting F^i_ext = F^i_boundary also works, but converges way slower.
  *exterior_n_dot_fluxes *= 2.;
  // This sign is inverted to account for the opposite signs of the normals
  *exterior_n_dot_fluxes += interior_n_dot_fluxes;
}

}  // namespace elliptic
