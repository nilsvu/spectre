// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Actions/ComputeOperatorAction.hpp"
#include "Elliptic/Actions/InitializeAnalyticSolution.hpp"
#include "Elliptic/Actions/InitializeIterationIds.hpp"
#include "Elliptic/Actions/InitializeSystem.hpp"
#include "Elliptic/DiscontinuousGalerkin/Actions/InitializeFluxes.hpp"
#include "Elliptic/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Elliptic/DiscontinuousGalerkin/ImposeInhomogeneousBoundaryConditionsOnSource.hpp"
#include "Elliptic/Systems/Poisson/Actions/Observe.hpp"
#include "Elliptic/Systems/Poisson/FirstOrderSystem.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "IO/Observer/Actions.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/FluxCommunication2.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/InitializeDomain.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/InitializeFluxes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/InitializeInterfaces.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/InitializeMortars.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/StrongFirstOrder.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NumericalFluxes/InternalPenalty.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/PopulateBoundaryMortars.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearSolver/Actions/TerminateIfConverged.hpp"
#include "NumericalAlgorithms/LinearSolver/Gmres/Gmres.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/ProductOfSinusoids.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim>
struct Metavariables {
  static constexpr OptionString help{
      "Find the solution to a Poisson problem in Dim spatial dimensions.\n"
      "Analytic solution: ProductOfSinusoids\n"
      "Linear solver: GMRES\n"
      "Numerical flux: FirstOrderInternalPenalty"};

  // The system provides all equations specific to the problem.
  using system = Poisson::FirstOrderSystem<Dim>;
  using fields_tag = typename system::fields_tag;

  // Specify the analytic solution and corresponding source to define the
  // Poisson problem
  using analytic_solution_tag =
      OptionTags::AnalyticSolution<Poisson::Solutions::ProductOfSinusoids<Dim>>;

  // Specify the linear solver algorithm. We must use GMRES since the operator
  // is not positive-definite for the first-order system.
  using linear_solver = LinearSolver::Gmres<Metavariables, fields_tag>;
  using linear_operand_tag = typename linear_solver::operand_tag;

  // Specify the DG boundary scheme. We use the strong first-order scheme here
  // that only requires us to compute normals dotted into the first-order
  // fluxes.
  using numerical_flux = dg::NumericalFluxes::FirstOrderInternalPenalty<
      Dim, tmpl::list<LinearSolver::Tags::Operand<Poisson::Field>>,
      tmpl::list<LinearSolver::Tags::Operand<Poisson::AuxiliaryField<Dim>>>>;
  using boundary_scheme = dg::BoundarySchemes::StrongFirstOrder<
      Dim, linear_operand_tag,
      Tags::NormalDotNumericalFluxComputer<numerical_flux>,
      LinearSolver::Tags::IterationId>;

  // Set up the domain creator from the input file.
  using domain_creator_tag = OptionTags::DomainCreator<Dim, Frame::Inertial>;

  // Collect all items to store in the cache.
  using const_global_cache_tag_list =
      tmpl::list<analytic_solution_tag,
                 OptionTags::NumericalFlux<numerical_flux>>;

  // Collect all reduction tags for observers
  using observed_reduction_data_tags = observers::collect_reduction_data_tags<
      tmpl::list<Poisson::Actions::Observe, linear_solver>>;

  // Specify all global synchronization points.
  enum class Phase { Initialization, RegisterWithObserver, Solve, Exit };

  // Construct the DgElementArray parallel component
  using initialization_actions = tmpl::list<
      dg::Actions::InitializeDomain<Dim>,
      elliptic::Actions::InitializeAnalyticSolution,
      elliptic::Actions::InitializeSystem,
      dg::Actions::InitializeInterfaces<
          system, dg::Initialization::slice_tags_to_face<>,
          dg::Initialization::slice_tags_to_exterior<>>,
      elliptic::dg::Actions::InitializeFluxes<boundary_scheme, system>,
      elliptic::Actions::InitializeIterationIds<
          LinearSolver::Tags::IterationId>,
      dg::Actions::InitializeMortars<boundary_scheme, true>,
      elliptic::dg::Actions::ImposeInhomogeneousBoundaryConditionsOnSource<
          typename system::fields_tag, boundary_scheme>,
      typename linear_solver::initialize_element,
      Initialization::Actions::RemoveOptionsAndTerminatePhase>;

  using element_array_component = elliptic::DgElementArray<
      Metavariables,
      Parallel::ForwardAllOptionsToDataBox<
          Initialization::option_tags<initialization_actions>>,
      tmpl::list<
          Parallel::PhaseActions<Phase, Phase::Initialization,
                                 initialization_actions>,

          Parallel::PhaseActions<
              Phase, Phase::RegisterWithObserver,
              tmpl::list<observers::Actions::RegisterWithObservers<
                             Poisson::Actions::Observe>,
                         Parallel::Actions::TerminatePhase>>,

          Parallel::PhaseActions<
              Phase, Phase::Solve,
              tmpl::list<Poisson::Actions::Observe,
                         LinearSolver::Actions::TerminateIfConverged,
                         dg::Actions::SendDataForFluxes<boundary_scheme>,
                         elliptic::Actions::ComputeOperatorAction<
                             Dim, LinearSolver::Tags::OperatorAppliedTo,
                             linear_operand_tag>,
                         Actions::MutateApply<
                             ::dg::PopulateBoundaryMortars<boundary_scheme>>,
                         dg::Actions::ReceiveDataForFluxes<boundary_scheme>,
                         Actions::MutateApply<boundary_scheme>,
                         typename linear_solver::perform_step>>>>;

  // Specify all parallel components that will execute actions at some point.
  using component_list =
      tmpl::flatten<tmpl::list<element_array_component,
                               typename linear_solver::component_list,
                               observers::Observer<Metavariables>,
                               observers::ObserverWriter<Metavariables>>>;

  // Specify the transitions between phases.
  static Phase determine_next_phase(
      const Phase& current_phase,
      const Parallel::CProxy_ConstGlobalCache<
          Metavariables>& /*cache_proxy*/) noexcept {
    switch (current_phase) {
      case Phase::Initialization:
        return Phase::RegisterWithObserver;
      case Phase::RegisterWithObserver:
        return Phase::Solve;
      case Phase::Solve:
        return Phase::Exit;
      case Phase::Exit:
        ERROR(
            "Should never call determine_next_phase with the current phase "
            "being 'Exit'");
      default:
        ERROR(
            "Unknown type of phase. Did you static_cast<Phase> an integral "
            "value?");
    }
  }
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &domain::creators::register_derived_with_charm};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
/// \endcond
