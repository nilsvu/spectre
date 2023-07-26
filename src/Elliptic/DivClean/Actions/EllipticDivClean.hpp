// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/DiscontinuousGalerkin/Actions/ApplyOperator.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/InitializeSubdomain.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/SubdomainOperator.hpp"
#include "Elliptic/FiniteDifference/FdOperator.hpp"
#include "Elliptic/SubdomainPreconditioners/MinusLaplacian.hpp"
#include "Elliptic/Systems/Poisson/BoundaryConditions/Robin.hpp"
#include "Elliptic/Systems/Poisson/FirstOrderSystem.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Jacobians.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Actions/CommunicateOverlapFields.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Schwarz.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace elliptic::divclean::Actions {

static constexpr size_t Dim = 3;
using poisson_system =
    Poisson::FirstOrderSystem<Dim, Poisson::Geometry::FlatCartesian>;
using div_clean_potential_tag = Poisson::Tags::Field;
using div_b_tag = ::Tags::div<hydro::Tags::MagneticField<DataVector, Dim>>;
using communicated_tags =
    tmpl::list<div_b_tag, evolution::dg::subcell::Tags::ActiveGrid,
               domain::Tags::Mesh<Dim>,
               evolution::dg::subcell::Tags::Mesh<Dim>>;

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
struct RunOnSubcells : db::SimpleTag {
  using type = bool;
  using group = OptionsGroup;
  static constexpr Options::String help =
      "Allow running the elliptic solver on FD subcells, instead of projecting "
      "to the DG grid. This currently works only for zero overlaps.";
  static constexpr bool pass_metavariables = false;
  using option_tags =
      tmpl::list<RunOnSubcells,
                 LinearSolver::Schwarz::OptionTags::MaxOverlap<OptionsGroup>>;
  static type create_from_options(const bool value, const bool max_overlap) {
    if (max_overlap > 0 and value) {
      ERROR(
          "Running elliptic div cleaning on subcells is currently only "
          "supported for zero overlaps. Either set 'MaxOverlap: 0', or set "
          "'RunOnSubcells: False'.");
    }
    return value;
  }
};

template <typename OptionsGroup>
struct InitializeElement {
  using simple_tags = tmpl::list<domain::Tags::ElementMap<Dim>,
                                 elliptic::dg::Tags::PenaltyParameter,
                                 elliptic::dg::Tags::Massive>;
  using compute_tags = tmpl::list<>;
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    // TODO:
    // - [ ] Overlap communication
    // - [ ] FD derivatives
    // - [ ] Moving mesh
    // - [ ] Curved background geometry
    // - [ ] Dynamic background geometry
    // - [ ] AMR
    // - [ ] FD
    // - [ ] Local time stepping
    const auto& domain = db::get<domain::Tags::Domain<Dim>>(box);
    const auto& block = domain.blocks()[element_id.block_id()];
    ASSERT(not block.is_time_dependent(), "Moving mesh not yet implemented");
    ElementMap<Dim, Frame::Inertial> inertial_element_map{
        element_id, block.stationary_map().get_clone()};

    Initialization::mutate_assign<tmpl::list<
        domain::Tags::ElementMap<Dim>, elliptic::dg::Tags::PenaltyParameter,
        elliptic::dg::Tags::Massive>>(
        make_not_null(&box), std::move(inertial_element_map), 1., false);

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

template <typename OptionsGroup>
using initialize_element =
    tmpl::list<InitializeElement<OptionsGroup>,
               elliptic::dg::Actions::initialize_operator<poisson_system>,
               elliptic::dg::subdomain_operator::Actions::InitializeSubdomain<
                   poisson_system, void, OptionsGroup>>;

namespace Tags {
template <typename SolverType, typename OptionsGroup,
          evolution::dg::subcell::ActiveGrid Grid>
struct SubdomainSolver : db::SimpleTag {
  using type = SolverType;
  static constexpr bool pass_metavariables = false;
  using option_tags =
      tmpl::list<LinearSolver::Schwarz::OptionTags::SubdomainSolver<
          SolverType, OptionsGroup>>;
  static type create_from_options(const type& value) {
    return deserialize<type>(serialize<type>(value).data());
  }
};
}  // namespace Tags

template <typename OptionsGroup>
struct EllipticDivClean {
  using SubdomainData = LinearSolver::Schwarz::ElementCenteredSubdomainData<
      Dim, tmpl::list<div_clean_potential_tag>>;
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
  using dg_subdomain_solver_tag =
      Tags::SubdomainSolver<std::unique_ptr<SubdomainSolver>, OptionsGroup,
                            evolution::dg::subcell::ActiveGrid::Dg>;
  using subcell_subdomain_solver_tag =
      Tags::SubdomainSolver<std::unique_ptr<SubdomainSolver>, OptionsGroup,
                            evolution::dg::subcell::ActiveGrid::Subcell>;
  using overlap_data_inbox_tag =
      LinearSolver::Schwarz::Actions::detail::OverlapFieldsTag<
          Dim, communicated_tags, OptionsGroup, TimeStepId>;

 public:
  using const_global_cache_tags =
      tmpl::list<LinearSolver::Schwarz::Tags::MaxOverlap<OptionsGroup>,
                 Enabled<OptionsGroup>, RunOnSubcells<OptionsGroup>,
                 logging::Tags::Verbosity<OptionsGroup>,
                 grmhd::ValenciaDivClean::Tags::EnableDivCleaning>;
  using simple_tags_from_options =
      tmpl::list<dg_subdomain_solver_tag, subcell_subdomain_solver_tag>;
  using inbox_tags = tmpl::list<overlap_data_inbox_tag>;

  using simple_tags = tmpl::list<subdomain_data_buffer_tag>;
  using compute_tags = tmpl::list<>;
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    if (not db::get<grmhd::ValenciaDivClean::Tags::EnableDivCleaning>(box)) {
      db::mutate<hydro::Tags::DivergenceCleaningField<DataVector>>(
          [](const gsl::not_null<Scalar<DataVector>*> div_cleaning_field) {
            get(*div_cleaning_field) = 0.;
          },
          make_not_null(&box));
    }
    if (not db::get<Enabled<OptionsGroup>>(box)) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }

    // TODO:
    // - Handle curved background in div(B) source
    // - Handle curved background in Poisson operator
    const size_t max_overlap =
        db::get<LinearSolver::Schwarz::Tags::MaxOverlap<OptionsGroup>>(box);
    const bool run_on_subcells = db::get<RunOnSubcells<OptionsGroup>>(box);
    const auto& element = db::get<domain::Tags::Element<Dim>>(box);
    const auto& active_grid =
        db::get<evolution::dg::subcell::Tags::ActiveGrid>(box);
    const auto& dg_mesh = db::get<domain::Tags::Mesh<Dim>>(box);
    const auto& subcell_mesh =
        db::get<evolution::dg::subcell::Tags::Mesh<Dim>>(box);
    const evolution::dg::subcell::ActiveGrid solver_grid =
        run_on_subcells ? active_grid : evolution::dg::subcell::ActiveGrid::Dg;
    const auto& mesh = solver_grid == evolution::dg::subcell::ActiveGrid::Dg
                           ? dg_mesh
                           : subcell_mesh;
    const auto& inv_jacobian =
        solver_grid == evolution::dg::subcell::ActiveGrid::Dg
            ? db::get<domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                                    Frame::Inertial>>(box)
            : db::get<evolution::dg::subcell::fd::Tags::
                          InverseJacobianLogicalToInertial<Dim>>(box);
    const auto& temporal_id = db::get<::Tags::TimeStepId>(box);
    const auto& extruding_extents =
        db::get<LinearSolver::Schwarz::Tags::Overlaps<
            elliptic::dg::subdomain_operator::Tags::ExtrudingExtent, Dim,
            OptionsGroup>>(box);

    // Wait for communicated overlap data
    const bool has_overlaps =
        max_overlap > 0 and element.number_of_neighbors() > 0;
    if (LIKELY(has_overlaps) and
        not ::dg::has_received_from_all_mortars<overlap_data_inbox_tag>(
            temporal_id, element, inboxes)) {
      return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    }

    // Assemble subdomain. Project div(B) to DG grid if on subcell.
    // TODO: Apply mass matrix?
    const auto& div_b = db::get<div_b_tag>(box);
    db::mutate<subdomain_data_buffer_tag>(
        [&div_b, &active_grid, &solver_grid, &mesh, &dg_mesh, &subcell_mesh,
         &has_overlaps, &temporal_id, &inboxes, &extruding_extents, &element,
         &run_on_subcells](const gsl::not_null<SubdomainData*> subdomain_data) {
          set_number_of_grid_points(
              make_not_null(&subdomain_data->element_data),
              mesh.number_of_grid_points());
          auto& div_clean_source =
              get<div_clean_potential_tag>(subdomain_data->element_data);
          if (active_grid == solver_grid) {
            div_clean_source = div_b;
          } else {
            // TODO: multiply with jac det?
            evolution::dg::subcell::fd::reconstruct(
                make_not_null(&get(div_clean_source)), get(div_b), dg_mesh,
                subcell_mesh.extents(),
                evolution::dg::subcell::fd::ReconstructionMethod::DimByDim);
          }
          // Nothing was communicated if the overlaps are empty
          if (not has_overlaps) {
            return;
          }
          ASSERT(not run_on_subcells,
                 "Nonzero overlaps are currently not supported when running on "
                 "subcells.");
          const auto inbox_overlap_data =
              std::move(tuples::get<overlap_data_inbox_tag>(inboxes)
                            .extract(temporal_id)
                            .mapped());
          auto& local_overlap_data = subdomain_data->overlap_data;
          local_overlap_data.clear();
          for (const auto& [overlap_id, overlap_data] : inbox_overlap_data) {
            const auto& overlap_div_b = get<div_b_tag>(overlap_data);
            // Project to DG grid if on subcell
            const auto& overlap_active_grid =
                get<evolution::dg::subcell::Tags::ActiveGrid>(overlap_data);
            const auto& overlap_dg_mesh =
                get<domain::Tags::Mesh<Dim>>(overlap_data);
            Variables<tmpl::list<div_clean_potential_tag>> overlap_vars{
                overlap_dg_mesh.number_of_grid_points()};
            if (overlap_active_grid == evolution::dg::subcell::ActiveGrid::Dg) {
              get<div_clean_potential_tag>(overlap_vars) = overlap_div_b;
            } else {
              const auto& overlap_subcell_mesh =
                  get<evolution::dg::subcell::Tags::Mesh<Dim>>(overlap_data);
              // TODO: multiply with jac det?
              evolution::dg::subcell::fd::reconstruct(
                  make_not_null(
                      &get(get<div_clean_potential_tag>(overlap_vars))),
                  get(overlap_div_b), overlap_dg_mesh,
                  overlap_subcell_mesh.extents(),
                  evolution::dg::subcell::fd::ReconstructionMethod::DimByDim);
            }
            const auto& direction = overlap_id.first;
            const auto& orientation =
                element.neighbors().at(direction).orientation();
            const auto direction_from_neighbor =
                orientation(direction.opposite());
            local_overlap_data[overlap_id] =
                LinearSolver::Schwarz::data_on_overlap(
                    overlap_vars, overlap_dg_mesh.extents(),
                    extruding_extents.at(overlap_id), direction_from_neighbor);
          }
        },
        make_not_null(&box));
    const auto& subdomain_source = db::get<subdomain_data_buffer_tag>(box);

    // Allocate workspace memory for repeatedly applying the subdomain operator
    const SubdomainOperator subdomain_operator{};

    // Create boundary conditions for the subdomain problem
    using BoundaryId = std::pair<size_t, Direction<Dim>>;
    using BoundaryConditionsBase =
        typename poisson_system::boundary_conditions_base;
    const Poisson::BoundaryConditions::Robin<Dim> subdomain_boundary_condition{
        1., 0., 0.};
    std::unordered_map<BoundaryId, const BoundaryConditionsBase&,
                       boost::hash<BoundaryId>>
        subdomain_boundary_conditions{};
    const auto& all_boundary_conditions =
        db::get<domain::Tags::ExternalBoundaryConditions<Dim>>(box);
    for (size_t block_id = 0; block_id < all_boundary_conditions.size();
         ++block_id) {
      const auto& block_boundary_conditions = all_boundary_conditions[block_id];
      for (const auto& [direction, boundary_condition] :
           block_boundary_conditions) {
        const auto boundary_id = std::make_pair(block_id, direction);
        subdomain_boundary_conditions.emplace(
            boundary_id, std::cref(subdomain_boundary_condition));
      }
    }

    // Solve the subdomain problem
    // TODO: reset the subdomain solver if the operator changes
    auto subdomain_solve_initial_guess_in_solution_out =
        make_with_value<SubdomainData>(subdomain_source, 0.);
    Convergence::HasConverged subdomain_solve_has_converged{};
    if (solver_grid == evolution::dg::subcell::ActiveGrid::Subcell) {
      const auto& subdomain_solver = get<subcell_subdomain_solver_tag>(box);
      subdomain_solve_has_converged = subdomain_solver.solve(
          make_not_null(&subdomain_solve_initial_guess_in_solution_out),
          [&mesh, &inv_jacobian](const gsl::not_null<SubdomainData*> result,
                                 const SubdomainData& operand,
                                 const auto&... /*args*/) {
            elliptic::fd::apply_operator(
                make_not_null(
                    &get(get<div_clean_potential_tag>(result->element_data))),
                get(get<div_clean_potential_tag>(operand.element_data)), mesh,
                inv_jacobian);
          },
          subdomain_source,
          std::forward_as_tuple(box, subdomain_boundary_conditions));
    } else {
      const auto& subdomain_solver = get<dg_subdomain_solver_tag>(box);
      subdomain_solve_has_converged = subdomain_solver.solve(
          make_not_null(&subdomain_solve_initial_guess_in_solution_out),
          subdomain_operator, subdomain_source,
          std::forward_as_tuple(box, subdomain_boundary_conditions));
    }
    // Re-naming the solution buffer for the code below
    const auto& subdomain_solution =
        subdomain_solve_initial_guess_in_solution_out;

    if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                 ::Verbosity::Verbose)) {
      const double time = db::get<::Tags::Time>(box);
      if (not subdomain_solve_has_converged or
          subdomain_solve_has_converged.reason() ==
              Convergence::Reason::MaxIterations) {
        Parallel::printf(
            "%s %s(%f): WARNING: Subdomain solver did not converge in %zu "
            "iterations: %e -> %e\n",
            element_id, pretty_type::name<OptionsGroup>(), time,
            subdomain_solve_has_converged.num_iterations(),
            subdomain_solve_has_converged.initial_residual_magnitude(),
            subdomain_solve_has_converged.residual_magnitude());
      } else if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                          ::Verbosity::Debug)) {
        Parallel::printf(
            "%s %s(%f): Subdomain solver converged in %zu iterations (%s): %e"
            "-> %e\n",
            element_id, pretty_type::name<OptionsGroup>(), time,
            subdomain_solve_has_converged.num_iterations(),
            subdomain_solve_has_converged.reason(),
            subdomain_solve_has_converged.initial_residual_magnitude(),
            subdomain_solve_has_converged.residual_magnitude());
      }
    }

    // Use solution to correct B-field in this element: B += grad(phi)
    // Also kill div-cleaning field in this element because we solved the div
    // constraint.
    const auto& div_clean_potential =
        get<div_clean_potential_tag>(subdomain_solution.element_data);
    auto b_correction =
        partial_derivative(div_clean_potential, mesh, inv_jacobian);
    if (active_grid != solver_grid) {
      for (size_t d = 0; d < Dim; ++d) {
        b_correction.get(d) = evolution::dg::subcell::fd::project(
            b_correction.get(d), dg_mesh, subcell_mesh.extents());
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

template <typename OptionsGroup>
using clean_actions =
    tmpl::list<LinearSolver::Schwarz::Actions::SendOverlapFields<
                   communicated_tags, OptionsGroup, false, ::Tags::TimeStepId>,
               EllipticDivClean<OptionsGroup>>;

}  // namespace elliptic::divclean::Actions
