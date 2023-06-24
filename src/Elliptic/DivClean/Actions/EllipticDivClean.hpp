// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/InitializeSubdomain.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/SubdomainOperator.hpp"
#include "Elliptic/SubdomainPreconditioners/MinusLaplacian.hpp"
#include "Elliptic/Systems/Poisson/BoundaryConditions/Robin.hpp"
#include "Elliptic/Systems/Poisson/FirstOrderSystem.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Schwarz.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace elliptic::divclean::Actions {

template <typename OptionsGroup>
struct Enabled : db::SimpleTag {
  using type = bool;
  using group = OptionsGroup;
  static constexpr Options::String help = "Enable elliptic div cleaning";
  static constexpr bool pass_metavariables = false;
  using option_tags = tmpl::list<Enabled>;
  static type create_from_options(const type& value) { return value; }
};

template <typename OptionsGroup>
struct EllipticDivClean {
 private:
  static constexpr size_t Dim = 3;
  using poisson_system =
      Poisson::FirstOrderSystem<Dim, Poisson::Geometry::FlatCartesian>;
  //   using elliptic_div_clean_vars_tag =
  //       ::Tags::Variables<tmpl::list<grmhd::ValenciaDivClean::Tags::TildePhi>>;
  //   using elliptic_div_clean_source_tag = ::Tags::Variables<
  //       tmpl::list<::Tags::div<grmhd::ValenciaDivClean::Tags::TildeB<>>>>;
  //   using schwarz_smoother = LinearSolver::Schwarz::Schwarz<
  //       elliptic_div_clean_vars_tag, OptionTags::SchwarzSmootherGroup,
  //       subdomain_operator, tmpl::list<>, elliptic_div_clean_source_tag>;

  using div_b_tag = ::Tags::div<hydro::Tags::MagneticField<DataVector, Dim>>;

 public:
  using SubdomainData = LinearSolver::Schwarz::ElementCenteredSubdomainData<
      Dim, tmpl::list<div_b_tag>>;
  struct subdomain_data_buffer_tag : db::SimpleTag {
    using type = SubdomainData;
  };
  using SubdomainOperator = elliptic::dg::subdomain_operator::SubdomainOperator<
      poisson_system, OptionsGroup, tmpl::list<>,
      tmpl::list<Poisson::BoundaryConditions::Robin<Dim>>>;
  using SubdomainSolver = LinearSolver::Serial::LinearSolver<
      tmpl::list<LinearSolver::Serial::Registrars::Gmres<SubdomainData>,
                 elliptic::subdomain_preconditioners::Registrars::
                     MinusLaplacian<Dim, OptionsGroup>,
                 LinearSolver::Serial::Registrars::ExplicitInverse>>;
  using subdomain_solver_tag = LinearSolver::Schwarz::Tags::SubdomainSolver<
      std::unique_ptr<SubdomainSolver>, OptionsGroup>;

 public:
  using const_global_cache_tags =
      tmpl::list<LinearSolver::Schwarz::Tags::MaxOverlap<OptionsGroup>,
                 Enabled<OptionsGroup>>;
  using simple_tags_from_options = tmpl::list<subdomain_solver_tag>;

  using simple_tags = tmpl::list<subdomain_data_buffer_tag>;
  using compute_tags = tmpl::list<>;
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    if (not db::get<Enabled<OptionsGroup>>(box)) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }

    // TODO:
    // - Decide where to store div clean field and source
    // - Initialize subdomain
    // - Handle curved background in div(B) source
    // - Handle curved background in Poisson operator
    // const size_t max_overlap =
    //     db::get<LinearSolver::Schwarz::Tags::MaxOverlap<OptionsGroup>>(box);
    // const auto& element = db::get<domain::Tags::Element<Dim>>(box);
    const auto& mesh = db::get<domain::Tags::Mesh<Dim>>(box);
    const size_t num_points = mesh.number_of_grid_points();
    const auto& inv_jacobian =
        db::get<domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                              Frame::Inertial>>(box);
    const auto& active_grid =
        db::get<evolution::dg::subcell::Tags::ActiveGrid>(box);
    const auto& subcell_mesh =
        db::get<evolution::dg::subcell::Tags::Mesh<Dim>>(box);

    // Wait for communicated overlap data
    // const bool has_overlaps =
    //     max_overlap > 0 and element.number_of_neighbors() > 0;
    // if (LIKELY(has_overlaps) and
    //     not dg::has_received_from_all_mortars<overlap_residuals_inbox_tag>(
    //         iteration_id, element, inboxes)) {
    //   return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    // }

    // Assemble subdomain. Project div(B) to DG grid if on subcell.
    const auto& div_b = db::get<div_b_tag>(box);
    db::mutate<subdomain_data_buffer_tag>(
        [&div_b, &num_points, &active_grid, &mesh,
         &subcell_mesh](const gsl::not_null<SubdomainData*> subdomain_data) {
          if (subdomain_data->element_data.number_of_grid_points() !=
              num_points) {
            subdomain_data->element_data.initialize(num_points);
          }
          if (active_grid == evolution::dg::subcell::ActiveGrid::Dg) {
            get<div_b_tag>(subdomain_data->element_data) = div_b;
          } else {
            // TODO: multiply with jac det?
            evolution::dg::subcell::fd::reconstruct(
                make_not_null(
                    &get(get<div_b_tag>(subdomain_data->element_data))),
                get(div_b), mesh, subcell_mesh.extents(),
                evolution::dg::subcell::fd::ReconstructionMethod::DimByDim);
          }
          // Nothing was communicated if the overlaps are empty
          //   if (LIKELY(has_overlaps)) {
          //     subdomain_data->overlap_data =
          //         std::move(tuples::get<overlap_residuals_inbox_tag>(inboxes)
          //                       .extract(iteration_id)
          //                       .mapped());
          //   }
        },
        make_not_null(&box));
    const auto& subdomain_source = db::get<subdomain_data_buffer_tag>(box);

    // Allocate workspace memory for repeatedly applying the subdomain operator
    const SubdomainOperator subdomain_operator{};

    // Solve the subdomain problem
    // const auto& subdomain_solver = get<subdomain_solver_tag>(box);
    auto subdomain_solve_initial_guess_in_solution_out =
        make_with_value<SubdomainData>(subdomain_source, 0.);
    // const auto subdomain_solve_has_converged = subdomain_solver.solve(
    //     make_not_null(&subdomain_solve_initial_guess_in_solution_out),
    //     subdomain_operator, subdomain_source, std::forward_as_tuple(box),
    // );
    // Re-naming the solution buffer for the code below
    const auto& subdomain_solution =
        subdomain_solve_initial_guess_in_solution_out;

    // Use solution to correct B-field in this element: B += grad(phi)
    // Also kill div-cleaning field in this element because we solved the div
    // constraint.
    auto b_correction = partial_derivative(
        get<div_b_tag>(subdomain_solution.element_data), mesh, inv_jacobian);
    if (active_grid != evolution::dg::subcell::ActiveGrid::Dg) {
      for (size_t d = 0; d < Dim; ++d) {
        b_correction.get(d) = evolution::dg::subcell::fd::project(
            b_correction.get(d), mesh, subcell_mesh.extents());
      }
    }
    db::mutate<hydro::Tags::MagneticField<DataVector, Dim>,
               hydro::Tags::DivergenceCleaningField<DataVector>>(
        [&b_correction](
            const gsl::not_null<tnsr::I<DataVector, Dim>*> magnetic_field,
            const gsl::not_null<Scalar<DataVector>*> div_cleaning_field) {
          for (size_t d = 0; d < Dim; ++d) {
            magnetic_field->get(d) += b_correction.get(d);
          }
          get(*div_cleaning_field) = 0.;
        },
        make_not_null(&box));

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

}  // namespace elliptic::divclean::Actions
