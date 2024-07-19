// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementLogicalCoordinates.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/FaceNormal.hpp"
#include "Domain/Tags/Faces.hpp"
#include "Elliptic/DiscontinuousGalerkin/Tags.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/Equations.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/ApplyMassMatrix.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "ParallelAlgorithms/Amr/Protocols/Projector.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Tags.hpp"
#include "PointwiseFunctions/AnalyticData/SelfForce/Scalar/CircularOrbit.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
struct GlobalCache;
}  // namespace Parallel
/// \endcond

namespace ScalarSelfForce::Actions {

/*!
 * \brief Initialize the "fixed sources" of the elliptic equations, i.e. their
 * variable-independent source term \f$f(x)\f$
 *
 * This action initializes \f$f(x)\f$ in an elliptic system of PDEs \f$-div(F) +
 * S = f(x)\f$.
 *
 * Uses:
 * - System:
 *   - `primal_fields`
 * - DataBox:
 *   - `BackgroundTag`
 *   - `Tags::Coordinates<Dim, Frame::Inertial>`
 *
 * DataBox:
 * - Adds:
 *   - `db::wrap_tags_in<::Tags::FixedSource, primal_fields>`
 */
template <typename System, typename BackgroundTag, typename SchwarzOptionsGroup,
          size_t Dim = System::volume_dim>
struct InitializeEffectiveSource : tt::ConformsTo<::amr::protocols::Projector> {
 private:
  using fixed_sources_tag = ::Tags::Variables<
      db::wrap_tags_in<::Tags::FixedSource, typename System::primal_fields>>;
  using singular_vars_tag = ::Tags::Variables<tmpl::list<
      Tags::SingularField,
      ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>>>;
  using singular_vars_on_mortars_tag =
      ::Tags::Variables<tmpl::list<Tags::SingularField,
                                   ::Tags::NormalDotFlux<Tags::SingularField>>>;
  using analytic_tags_list = tmpl::push_back<
      typename fixed_sources_tag::tags_list, Tags::SingularField,
      ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>,
      Tags::BoyerLindquistRadius>;
  template <typename Tag>
  using overlaps_tag =
      LinearSolver::Schwarz::Tags::Overlaps<Tag, Dim, SchwarzOptionsGroup>;

 public:  // Iterable action
  using const_global_cache_tags =
      tmpl::list<elliptic::dg::Tags::Massive, BackgroundTag>;
  using simple_tags =
      tmpl::list<fixed_sources_tag, singular_vars_tag,
                 overlaps_tag<singular_vars_on_mortars_tag>,
                 Tags::BoyerLindquistRadius, Tags::FieldIsRegularized,
                 overlaps_tag<Tags::FieldIsRegularized>>;
  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    db::mutate_apply<InitializeEffectiveSource>(make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }

 public:  // DataBox mutator, amr::protocols::Projector
  using return_tags = simple_tags;
  using argument_tags = tmpl::list<
      domain::Tags::Coordinates<Dim, Frame::Inertial>,
      domain::Tags::Domain<Dim>, domain::Tags::Element<Dim>,
      ::Tags::Mortars<domain::Tags::Coordinates<Dim, Frame::Inertial>, Dim>,
      BackgroundTag, elliptic::dg::Tags::Massive, domain::Tags::Mesh<Dim>,
      domain::Tags::DetInvJacobian<Frame::ElementLogical, Frame::Inertial>,
      Parallel::Tags::Metavariables>;

  template <typename Background, typename Metavariables, typename... AmrData>
  static void apply(
      const gsl::not_null<typename fixed_sources_tag::type*> fixed_sources,
      const gsl::not_null<typename singular_vars_tag::type*> singular_vars,
      const gsl::not_null<
          DirectionalIdMap<Dim, typename singular_vars_on_mortars_tag::type>*>
          singular_vars_on_mortars,
      const gsl::not_null<Scalar<DataVector>*> bl_radius,
      const gsl::not_null<bool*> field_is_regularized,
      const gsl::not_null<DirectionalIdMap<Dim, bool>*>
          neighbors_field_is_regularized,
      const tnsr::I<DataVector, Dim>& inertial_coords,
      const Domain<Dim>& domain, const Element<Dim>& element,
      const DirectionalIdMap<Dim, tnsr::I<DataVector, Dim>>&
          all_mortar_inertial_coords,
      const Background& background, const bool massive, const Mesh<Dim>& mesh,
      const Scalar<DataVector>& det_inv_jacobian, const Metavariables& /*meta*/,
      const AmrData&... /*amr_data*/) {
    const auto& circular_orbit =
        dynamic_cast<const ScalarSelfForce::AnalyticData::CircularOrbit&>(
            background);

    // Check if this element and its neighbors solve for the regular field or
    // the full field
    const tnsr::I<double, 2> puncture_pos = circular_orbit.puncture_position();
    const auto puncture_in_element =
        [&puncture_pos, &domain](const ElementId<Dim>& element_id) -> bool {
      const auto& block = domain.blocks()[element_id.block_id()];
      //   const ElementMap<Dim, Frame::Inertial> element_map{element_id,
      //   block}; const tnsr::I<DataVector, Dim, Frame::ElementLogical>
      //   xi_corners{
      //       {{{-1., 1.}, {-1., 1.}}}};
      //   const auto x_corners = element_map(xi_corners);
      //   const auto& r_star_bounds = get<0>(x_corners);
      //   const auto& theta_bounds = get<1>(x_corners);
      //   const double worldtube_radius = 10;
      //   const double worldtube_angular_size = 0.25 * M_PI_4;
      //   return (r_star_bounds[1] >= get<0>(puncture_pos) - worldtube_radius
      //   and
      //           r_star_bounds[0] <= get<0>(puncture_pos) + worldtube_radius
      //           and theta_bounds[1] >=
      //               get<1>(puncture_pos) - worldtube_angular_size and
      //           theta_bounds[0] <= get<1>(puncture_pos) +
      //           worldtube_angular_size);
      const auto block_logical_coords =
          block_logical_coordinates_single_point(puncture_pos, block);
      return block_logical_coords.has_value();
      //   return element_logical_coordinates(*block_logical_coords, element_id)
      //       .has_value();
      // };

      // TODO: why doesn't this work? Neighbors in r_star are also regularized??
      //   return element_id.block_id() == 4;
    };
    *field_is_regularized = puncture_in_element(element.id());
    neighbors_field_is_regularized->clear();
    for (const auto& [direction, neighbors] : element.neighbors()) {
      for (const auto& neighbor_id : neighbors) {
        const LinearSolver::Schwarz::OverlapId<Dim> overlap_id{direction,
                                                               neighbor_id};
        neighbors_field_is_regularized->emplace(
            overlap_id, puncture_in_element(neighbor_id));
      }
    }

    // Only set the effective source if solving for the regular field
    if (*field_is_regularized) {
      const auto vars =
          circular_orbit.variables(inertial_coords, analytic_tags_list{});
      fixed_sources->initialize(mesh.number_of_grid_points());
      singular_vars->initialize(mesh.number_of_grid_points());
      get<::Tags::FixedSource<Tags::MMode>>(*fixed_sources) =
          get<::Tags::FixedSource<Tags::MMode>>(vars);
      get<Tags::SingularField>(*singular_vars) = get<Tags::SingularField>(vars);
      get<::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>>(
          *singular_vars) =
          get<::Tags::deriv<Tags::SingularField, tmpl::size_t<2>,
                            Frame::Inertial>>(vars);
      *bl_radius = get<Tags::BoyerLindquistRadius>(vars);
      // Apply DG mass matrix to the fixed sources if the DG operator is massive
      if (massive) {
        *fixed_sources /= get(det_inv_jacobian);
        ::dg::apply_mass_matrix(fixed_sources, mesh);
      }

      // Set the singular field and its derivative on the faces
      singular_vars_on_mortars->clear();
      for (const auto& [mortar_id, mortar_inertial_coords] :
           all_mortar_inertial_coords) {
        const auto vars_on_mortar = circular_orbit.variables(
            mortar_inertial_coords, analytic_tags_list{});
        const auto background_on_mortar = circular_orbit.variables(
            mortar_inertial_coords,
            tmpl::list<Tags::Alpha, Tags::Beta, Tags::Gamma>{});
        auto& singular_vars_on_mortar = (*singular_vars_on_mortars)[mortar_id];
        singular_vars_on_mortar.initialize(
            mortar_inertial_coords.begin()->size());
        get<Tags::SingularField>(singular_vars_on_mortar) =
            get<Tags::SingularField>(vars_on_mortar);
        const auto& deriv_singular_field_on_mortar =
            get<::Tags::deriv<Tags::SingularField, tmpl::size_t<2>,
                              Frame::Inertial>>(vars_on_mortar);
        const auto& alpha_on_mortar = get<Tags::Alpha>(background_on_mortar);
        tnsr::I<ComplexDataVector, Dim> singular_field_flux_on_mortar{};
        ScalarSelfForce::fluxes(make_not_null(&singular_field_flux_on_mortar),
                                alpha_on_mortar,
                                deriv_singular_field_on_mortar);
        // Assuming mortar normal is just (1, 0, 0)
        tnsr::i<DataVector, Dim> mortar_normal{
            mortar_inertial_coords.begin()->size(), 0.};
        mortar_normal.get(mortar_id.direction().dimension()) =
            mortar_id.direction().sign();
        normal_dot_flux(
            make_not_null(&get<::Tags::NormalDotFlux<Tags::SingularField>>(
                singular_vars_on_mortar)),
            mortar_normal, singular_field_flux_on_mortar);
      }
    } else {
      *fixed_sources = Variables<typename fixed_sources_tag::tags_list>{
          mesh.number_of_grid_points(), 0.};
      *singular_vars = Variables<typename singular_vars_tag::tags_list>{
          mesh.number_of_grid_points(), 0.};
      *bl_radius = Scalar<DataVector>{mesh.number_of_grid_points(), 0.};
      singular_vars_on_mortars->clear();
      for (const auto& [mortar_id, mortar_inertial_coords] :
           all_mortar_inertial_coords) {
        (void)mortar_inertial_coords;
        (*singular_vars_on_mortars)[mortar_id];
      }
    }
  }
};

}  // namespace ScalarSelfForce::Actions
