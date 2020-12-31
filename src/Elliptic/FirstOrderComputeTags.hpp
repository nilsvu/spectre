// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/InterfaceHelpers.hpp"
#include "Elliptic/DiscontinuousGalerkin/Tags.hpp"
#include "Elliptic/FirstOrderOperator.hpp"
#include "Elliptic/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic {
namespace Tags {

template <size_t Dim, typename FluxesComputer, typename VariablesTag,
          typename PrimalVariables, typename AuxiliaryVariables>
struct FirstOrderFluxesCompute
    : db::add_tag_prefix<::Tags::Flux, VariablesTag, tmpl::size_t<Dim>,
                         Frame::Inertial>,
      db::ComputeTag {
 private:
  using fluxes_computer_tag = elliptic::Tags::FluxesComputer<FluxesComputer>;

 public:
  using base = db::add_tag_prefix<::Tags::Flux, VariablesTag, tmpl::size_t<Dim>,
                                  Frame::Inertial>;
  using argument_tags = tmpl::push_front<typename FluxesComputer::argument_tags,
                                         VariablesTag, fluxes_computer_tag>;
  using volume_tags =
      tmpl::push_front<get_volume_tags<FluxesComputer>, fluxes_computer_tag>;
  using return_type = typename base::type;
  template <typename... FluxesArgs>
  static void function(const gsl::not_null<return_type*> fluxes,
                       const typename VariablesTag::type& vars,
                       const FluxesComputer& fluxes_computer,
                       const FluxesArgs&... fluxes_args) noexcept {
    *fluxes = return_type{vars.number_of_grid_points()};
    elliptic::first_order_fluxes<Dim, PrimalVariables, AuxiliaryVariables>(
        fluxes, vars, fluxes_computer, fluxes_args...);
  }
};

template <size_t Dim, typename SourcesComputer, typename VariablesTag,
          typename PrimalVariables, typename AuxiliaryVariables>
struct FirstOrderSourcesCompute
    : db::add_tag_prefix<::Tags::Source, VariablesTag>,
      db::ComputeTag {
  using base = db::add_tag_prefix<::Tags::Source, VariablesTag>;
  using argument_tags =
      tmpl::push_front<typename SourcesComputer::argument_tags, VariablesTag,
                       db::add_tag_prefix<::Tags::Flux, VariablesTag,
                                          tmpl::size_t<Dim>, Frame::Inertial>>;
  using volume_tags = get_volume_tags<SourcesComputer>;
  using return_type = typename base::type;
  template <typename Vars, typename Fluxes, typename... SourcesArgs>
  static void function(const gsl::not_null<return_type*> sources,
                       const Vars& vars, const Fluxes& fluxes,
                       const SourcesArgs&... sources_args) noexcept {
    *sources = return_type{vars.number_of_grid_points()};
    elliptic::first_order_sources<Dim, PrimalVariables, AuxiliaryVariables,
                                  SourcesComputer>(sources, vars, fluxes,
                                                   sources_args...);
  }
};

// Sets n.F_u(div(F_v)) <- n.F_u to impose neumann boundary conditions through
// the internal penalty flux.
template <size_t Dim, typename PrimalVariables>
struct ImposeAuxiliaryConstraint;

template <size_t Dim, typename... PrimalVariables>
struct ImposeAuxiliaryConstraint<Dim, tmpl::list<PrimalVariables...>>
    : ::Tags::Variables<db::wrap_tags_in<
          elliptic::dg::Tags::NormalDotDivAuxFlux,
          tmpl::list<PrimalVariables...>, tmpl::size_t<Dim>, Frame::Inertial>>,
      db::ComputeTag {
  using primal_variables = tmpl::list<PrimalVariables...>;
  using base = ::Tags::Variables<db::wrap_tags_in<
      elliptic::dg::Tags::NormalDotDivAuxFlux, tmpl::list<PrimalVariables...>,
      tmpl::size_t<Dim>, Frame::Inertial>>;
  using argument_tags = tmpl::push_front<
      db::wrap_tags_in<::Tags::NormalDotFlux, primal_variables>,
      domain::Tags::Mesh<Dim - 1>>;
  using return_type = db::item_type<base>;

  template <typename... NDotPrimalFluxes>
  static void function(
      const gsl::not_null<return_type*> n_dot_div_aux_fluxes,
      const Mesh<Dim - 1>& mesh,
      const NDotPrimalFluxes&... n_dot_primal_fluxes) noexcept {
    *n_dot_div_aux_fluxes = return_type{mesh.number_of_grid_points()};
    const auto helper = [](const auto n_dot_div_aux_flux,
                           const auto& n_dot_primal_flux) noexcept {
      *n_dot_div_aux_flux = n_dot_primal_flux;
    };
    EXPAND_PACK_LEFT_TO_RIGHT(helper(
        make_not_null(&get<elliptic::dg::Tags::NormalDotDivAuxFlux<
                          PrimalVariables, tmpl::size_t<Dim>, Frame::Inertial>>(
            *n_dot_div_aux_fluxes)),
        n_dot_primal_fluxes));
  }
};

}  // namespace Tags
}  // namespace elliptic
