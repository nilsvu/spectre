// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Actions/InitializeAnalyticSolution.hpp"
#include "Elliptic/Actions/InitializeSystem.hpp"
#include "Elliptic/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Elliptic/DiscontinuousGalerkin/ImposeBoundaryConditions.hpp"
#include "Elliptic/DiscontinuousGalerkin/ImposeInhomogeneousBoundaryConditionsOnSource.hpp"
#include "Elliptic/DiscontinuousGalerkin/InitializeFluxes.hpp"
#include "Elliptic/DiscontinuousGalerkin/NumericalFluxes/FirstOrderInternalPenalty.hpp"
#include "Elliptic/FirstOrderOperator.hpp"
#include "Elliptic/Systems/Poisson/FirstOrderSystem.hpp"
#include "Elliptic/Tags.hpp"
#include "Elliptic/Triggers/EveryNIterations.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "IO/Observer/Actions.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/RegisterObservers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/CollectDataForFluxes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/FluxCommunication.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/StrongFirstOrder/StrongFirstOrder.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "Options/Options.hpp"
#include "Parallel/Actions/Goto.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeDomain.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeInterfaces.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeMortars.hpp"
#include "ParallelAlgorithms/Events/ObserveErrorNorms.hpp"
#include "ParallelAlgorithms/Events/ObserveFields.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "ParallelAlgorithms/LinearSolver/Actions/TerminateIfConverged.hpp"
#include "ParallelAlgorithms/LinearSolver/Gmres/Gmres.hpp"
#include "ParallelAlgorithms/LinearSolver/Richardson/Richardson.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Schwarz.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/SubdomainData.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/Lorentzian.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/Moustache.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/ProductOfSinusoids.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/TMPL.hpp"

struct LinearSolverGroup {
  static std::string name() noexcept { return "LinearSolver"; }
  static constexpr OptionString help = "The iterative linear solver";
};

struct LinearSolverOptions {
  using group = LinearSolverGroup;
  static std::string name() noexcept { return "Schwarz"; }
  static constexpr OptionString help =
      "Options for the iterative linear solver";
};

struct PreconditionerGroup {
  static std::string name() noexcept { return "Preconditioner"; }
  static constexpr OptionString help =
      "The preconditioner for the linear solves";
};

struct PreconditionerOptions {
  using group = PreconditionerGroup;
  static std::string name() noexcept { return "Richardson"; }
  static constexpr OptionString help =
      "Options for the Richardson preconditioner";
};

template <size_t Dim, typename FieldsTag, typename PrimalFields,
          typename AuxiliaryFields, typename FluxesComputerTag,
          typename SourcesComputer, typename NumericalFluxesComputerTag>
struct DgSubdomainOperator;

template <size_t Dim, typename FieldsTag, typename... PrimalFields,
          typename... AuxiliaryFields, typename FluxesComputerTag,
          typename SourcesComputer, typename NumericalFluxesComputerTag>
struct DgSubdomainOperator<Dim, FieldsTag, tmpl::list<PrimalFields...>,
                           tmpl::list<AuxiliaryFields...>, FluxesComputerTag,
                           SourcesComputer, NumericalFluxesComputerTag> {
 public:
  static constexpr size_t volume_dim = Dim;

  using all_fields_tags = db::get_variables_tags_list<FieldsTag>;
  using fluxes_tags =
      db::wrap_tags_in<::Tags::Flux, all_fields_tags, tmpl::size_t<volume_dim>,
                       Frame::Inertial>;
  using div_fluxes_tags = db::wrap_tags_in<::Tags::div, fluxes_tags>;
  using n_dot_fluxes_tags =
      db::wrap_tags_in<::Tags::NormalDotFlux, all_fields_tags>;

  using FluxesComputerType = db::const_item_type<FluxesComputerTag>;
  using NumericalFluxesComputerType =
      db::const_item_type<NumericalFluxesComputerTag>;
  using SubdomainDataType =
      LinearSolver::schwarz_detail::SubdomainData<volume_dim, all_fields_tags>;
  using OverlapDataType = typename SubdomainDataType::OverlapDataType;
  using BoundaryData = dg::SimpleBoundaryData<
      tmpl::remove_duplicates<tmpl::append<
          n_dot_fluxes_tags,
          typename NumericalFluxesComputerType::package_field_tags>>,
      typename NumericalFluxesComputerType::package_extra_tags>;

  template <size_t VolumeDim>
  using MortarId = std::pair<::Direction<VolumeDim>, ElementId<VolumeDim>>;
  template <size_t MortarDim>
  using MortarSizes = std::array<Spectral::MortarSize, MortarDim>;
  template <size_t VolumeDim, typename ValueType>
  using MortarMap = std::unordered_map<MortarId<VolumeDim>, ValueType,
                                       boost::hash<MortarId<VolumeDim>>>;

 private:
  // These functions are specific to the strong first-order internal penalty
  // scheme
  static BoundaryData package_boundary_data(
      const NumericalFluxesComputerType& numerical_fluxes_computer,
      const FluxesComputerType& fluxes_computer,
      const tnsr::i<DataVector, volume_dim>& face_normal,
      const Variables<n_dot_fluxes_tags>& n_dot_fluxes,
      const Variables<div_fluxes_tags>& div_fluxes) noexcept {
    BoundaryData boundary_data{n_dot_fluxes.number_of_grid_points()};
    boundary_data.field_data.assign_subset(n_dot_fluxes);
    dg::NumericalFluxes::package_data(
        make_not_null(&boundary_data), numerical_fluxes_computer,
        get<::Tags::NormalDotFlux<AuxiliaryFields>>(n_dot_fluxes)...,
        get<::Tags::div<::Tags::Flux<AuxiliaryFields, tmpl::size_t<volume_dim>,
                                     Frame::Inertial>>>(div_fluxes)...,
        fluxes_computer, face_normal);
    return boundary_data;
  }
  static void apply_boundary_contribution(
      const gsl::not_null<Variables<all_fields_tags>*> result,
      const NumericalFluxesComputerType& numerical_fluxes_computer,
      const BoundaryData& local_boundary_data,
      const BoundaryData& remote_boundary_data,
      const Scalar<DataVector>& magnitude_of_face_normal,
      const Mesh<volume_dim>& mesh, const Direction<volume_dim>& direction,
      const Mesh<volume_dim - 1>& mortar_mesh,
      const MortarSizes<volume_dim - 1>& mortar_size) noexcept {
    const size_t dimension = direction.dimension();
    auto boundary_contribution =
        dg::BoundarySchemes::strong_first_order_boundary_flux<all_fields_tags>(
            local_boundary_data, remote_boundary_data,
            numerical_fluxes_computer, magnitude_of_face_normal,
            mesh.extents(dimension), mesh.slice_away(dimension), mortar_mesh,
            mortar_size);
    add_slice_to_data(result, std::move(boundary_contribution), mesh.extents(),
                      dimension, index_to_slice_at(mesh.extents(), direction));
  }

 public:
  using inv_jacobian_tag =
      ::Tags::InverseJacobian<::Tags::ElementMap<volume_dim>,
                              ::Tags::Coordinates<volume_dim, Frame::Logical>>;
  using argument_tags = tmpl::list<
      ::Tags::Element<volume_dim>, ::Tags::Mesh<volume_dim>, inv_jacobian_tag,
      FluxesComputerTag, NumericalFluxesComputerTag,
      ::Tags::Interface<
          ::Tags::InternalDirections<volume_dim>,
          ::Tags::Normalized<::Tags::UnnormalizedFaceNormal<volume_dim>>>,
      ::Tags::Interface<
          ::Tags::BoundaryDirectionsInterior<volume_dim>,
          ::Tags::Normalized<::Tags::UnnormalizedFaceNormal<volume_dim>>>,
      ::Tags::Interface<
          ::Tags::InternalDirections<volume_dim>,
          ::Tags::Magnitude<::Tags::UnnormalizedFaceNormal<volume_dim>>>,
      ::Tags::Interface<
          ::Tags::BoundaryDirectionsInterior<volume_dim>,
          ::Tags::Magnitude<::Tags::UnnormalizedFaceNormal<volume_dim>>>,
      ::Tags::Mortars<::Tags::Mesh<volume_dim - 1>, volume_dim>,
      ::Tags::Mortars<::Tags::MortarSize<volume_dim - 1>, volume_dim>>;
  static SubdomainDataType apply(
      const SubdomainDataType& arg, const Element<volume_dim>& element,
      const Mesh<volume_dim>& mesh,
      const db::const_item_type<inv_jacobian_tag>& inv_jacobian,
      const FluxesComputerType& fluxes_computer,
      const NumericalFluxesComputerType& numerical_fluxes_computer,
      const db::const_item_type<::Tags::Interface<
          ::Tags::InternalDirections<volume_dim>,
          ::Tags::Normalized<::Tags::UnnormalizedFaceNormal<volume_dim>>>>&
          internal_face_normals,
      const db::const_item_type<::Tags::Interface<
          ::Tags::BoundaryDirectionsInterior<volume_dim>,
          ::Tags::Normalized<::Tags::UnnormalizedFaceNormal<volume_dim>>>>&
          boundary_face_normals,
      const db::const_item_type<::Tags::Interface<
          ::Tags::InternalDirections<volume_dim>,
          ::Tags::Magnitude<::Tags::UnnormalizedFaceNormal<volume_dim>>>>&
          internal_face_normal_magnitudes,
      const db::const_item_type<::Tags::Interface<
          ::Tags::BoundaryDirectionsInterior<volume_dim>,
          ::Tags::Magnitude<::Tags::UnnormalizedFaceNormal<volume_dim>>>>&
          boundary_face_normal_magnitudes,
      const db::const_item_type<::Tags::Mortars<::Tags::Mesh<volume_dim - 1>,
                                                volume_dim>>& mortar_meshes,
      const db::const_item_type<
          ::Tags::Mortars<::Tags::MortarSize<volume_dim - 1>, volume_dim>>&
          mortar_sizes) noexcept {
    SubdomainDataType result{arg.element_data.number_of_grid_points()};
    // Since the subdomain operator is called repeatedly for the subdomain solve
    // it could help performance to avoid re-allocating memory by storing the
    // tensor quantities in a buffer.
    // Parallel::printf("\n\nComputing subdomain operator of:\n%s\n",
    //                  arg.element_data);
    // Compute bulk contribution in central element
    const auto central_fluxes =
        elliptic::first_order_fluxes<volume_dim, tmpl::list<PrimalFields...>,
                                     tmpl::list<AuxiliaryFields...>>(
            arg.element_data, fluxes_computer);
    const auto central_div_fluxes =
        divergence(central_fluxes, mesh, inv_jacobian);
    elliptic::first_order_operator(
        make_not_null(&result.element_data), central_div_fluxes,
        elliptic::first_order_sources<tmpl::list<PrimalFields...>,
                                      tmpl::list<AuxiliaryFields...>,
                                      SourcesComputer>(arg.element_data));
    // Add boundary contributions
    for (const auto& mortar_id_and_mesh : mortar_meshes) {
      const auto& mortar_id = mortar_id_and_mesh.first;
      const auto& mortar_mesh = mortar_id_and_mesh.second;
      const auto& mortar_size = mortar_sizes.at(mortar_id);
      const auto& direction = mortar_id.first;
      const auto& neighbor_id = mortar_id.second;

      const size_t dimension = direction.dimension();
      const auto face_mesh = mesh.slice_away(dimension);
      const size_t face_num_points = face_mesh.number_of_grid_points();
      const size_t slice_index = index_to_slice_at(mesh.extents(), direction);

      const bool is_boundary =
          neighbor_id == ElementId<volume_dim>::external_boundary_id();

      const tnsr::i<DataVector, volume_dim>& face_normal =
          is_boundary ? boundary_face_normals.at(direction)
                      : internal_face_normals.at(direction);
      const Scalar<DataVector>& magnitude_of_face_normal =
          is_boundary ? boundary_face_normal_magnitudes.at(direction)
                      : internal_face_normal_magnitudes.at(direction);

      // Compute normal dot fluxes
      const auto central_fluxes_on_face =
          data_on_slice(central_fluxes, mesh.extents(), dimension, slice_index);
      const auto normal_dot_central_fluxes =
          normal_dot_flux<all_fields_tags>(face_normal, central_fluxes_on_face);

      // Slice flux divergences to face
      const auto central_div_fluxes_on_face = data_on_slice(
          central_div_fluxes, mesh.extents(), dimension, slice_index);

      // Assemble local boundary data
      const auto local_boundary_data = package_boundary_data(
          numerical_fluxes_computer, fluxes_computer, face_normal,
          normal_dot_central_fluxes, central_div_fluxes_on_face);

      // Assemble remote boundary data
      auto remote_face_normal = face_normal;
      for (size_t d = 0; d < volume_dim; d++) {
        remote_face_normal.get(d) *= -1.;
      }
      BoundaryData remote_boundary_data;
      if (is_boundary) {
        // On exterior ("ghost") faces, manufacture boundary data that represent
        // homogeneous Dirichlet boundary conditions
        const auto central_vars_on_face = data_on_slice(
            arg.element_data, mesh.extents(), dimension, slice_index);
        typename SubdomainDataType::Vars ghost_vars{face_num_points};
        ::elliptic::dg::homogeneous_dirichlet_boundary_conditions<
            tmpl::list<PrimalFields...>>(make_not_null(&ghost_vars),
                                         central_vars_on_face);
        const auto ghost_fluxes =
            ::elliptic::first_order_fluxes<volume_dim,
                                           tmpl::list<PrimalFields...>,
                                           tmpl::list<AuxiliaryFields...>>(
                ghost_vars, fluxes_computer);
        const auto ghost_normal_dot_fluxes =
            normal_dot_flux<all_fields_tags>(remote_face_normal, ghost_fluxes);
        remote_boundary_data =
            package_boundary_data(numerical_fluxes_computer, fluxes_computer,
                                  remote_face_normal, ghost_normal_dot_fluxes,
                                  // TODO: Is this correct?
                                  central_div_fluxes_on_face);
      } else {
        // On internal boundaries, get neighbor data from workspace
        // Note that all overlap data is oriented from the perspective of the
        // neighbor. We could re-orient it before sending, but then we'd also
        // have to re-orient the meshes.
        const auto& neighbor_orientation =
            element.neighbors().at(direction).orientation();
        const auto direction_from_neighbor =
            neighbor_orientation(direction.opposite());
        const size_t dimension_in_neighbor =
            direction_from_neighbor.dimension();
        // Parallel::printf("> Internal mortar %s:\n", mortar_id);
        const auto& overlap_data = arg.boundary_data.at(mortar_id);
        // Parallel::printf("Overlap data: %s\n", overlap_data.field_data);
        const auto& neighbor_mesh = overlap_data.volume_mesh;
        const size_t overlap =
            overlap_data.overlap_extents[dimension_in_neighbor];

        // Extend the overlap data to the full neighbor mesh by filling it
        // with zeros and adding the overlapping slices
        typename SubdomainDataType::Vars neighbor_data{
            neighbor_mesh.number_of_grid_points(), 0.};
        for (size_t i = 0; i < overlap; i++) {
          add_slice_to_data(
              make_not_null(&neighbor_data),
              data_on_slice(overlap_data.field_data,
                            overlap_data.overlap_extents, dimension_in_neighbor,
                            index_to_slice_at(overlap_data.overlap_extents,
                                              direction_from_neighbor, i)),
              neighbor_mesh.extents(), dimension_in_neighbor,
              index_to_slice_at(neighbor_mesh.extents(),
                                direction_from_neighbor, i));
        }
        // Parallel::printf("Extended overlap data: %s\n", neighbor_data);

        // Compute the volume contribution in the neighbor from the extended
        // overlap data
        // TODO: Make sure fluxes args are used from neighbor
        const auto neighbor_fluxes =
            ::elliptic::first_order_fluxes<volume_dim,
                                           tmpl::list<PrimalFields...>,
                                           tmpl::list<AuxiliaryFields...>>(
                neighbor_data, fluxes_computer);
        const auto neighbor_div_fluxes = divergence(
            neighbor_fluxes, neighbor_mesh, overlap_data.inv_jacobian);
        typename SubdomainDataType::Vars neighbor_result_extended{
            neighbor_mesh.number_of_grid_points()};
        elliptic::first_order_operator(
            make_not_null(&neighbor_result_extended), neighbor_div_fluxes,
            elliptic::first_order_sources<tmpl::list<PrimalFields...>,
                                          tmpl::list<AuxiliaryFields...>,
                                          SourcesComputer>(neighbor_data));
        // Parallel::printf("Extended result on overlap: %s\n",
        //                  neighbor_result_extended);

        const auto neighbor_fluxes_on_face = data_on_slice(
            neighbor_fluxes, neighbor_mesh.extents(), dimension_in_neighbor,
            index_to_slice_at(neighbor_mesh.extents(),
                              direction_from_neighbor));
        const auto neighbor_div_fluxes_on_face = data_on_slice(
            neighbor_div_fluxes, neighbor_mesh.extents(), dimension_in_neighbor,
            index_to_slice_at(neighbor_mesh.extents(),
                              direction_from_neighbor));
        auto remote_normal_dot_fluxes = normal_dot_flux<all_fields_tags>(
            remote_face_normal, neighbor_fluxes_on_face);
        remote_boundary_data = package_boundary_data(
            numerical_fluxes_computer, fluxes_computer, remote_face_normal,
            remote_normal_dot_fluxes, neighbor_div_fluxes_on_face);
        // TODO: orient and project

        // Apply the boundary contribution to the neighbor overlap
        apply_boundary_contribution(
            make_not_null(&neighbor_result_extended), numerical_fluxes_computer,
            remote_boundary_data, local_boundary_data,
            overlap_data.magnitude_of_face_normal, neighbor_mesh,
            direction_from_neighbor, overlap_data.mortar_mesh,
            overlap_data.mortar_size);
        // Parallel::printf("Extended result on overlap incl. boundary contribs:
        // %s\n",
        //                  neighbor_result_extended);
        typename SubdomainDataType::Vars neighbor_result{
            overlap_data.overlap_extents.product(), 0.};
        for (size_t i = 0; i < overlap; i++) {
          add_slice_to_data(
              make_not_null(&neighbor_result),
              data_on_slice(neighbor_result_extended, neighbor_mesh.extents(),
                            dimension_in_neighbor,
                            index_to_slice_at(neighbor_mesh.extents(),
                                              direction_from_neighbor, i)),
              overlap_data.overlap_extents, dimension_in_neighbor,
              index_to_slice_at(overlap_data.overlap_extents,
                                direction_from_neighbor, i));
        }
        // TODO: Fake boundary contributions from the other mortars of the
        // neighbor by filling their data with zeros
        // Parallel::printf("Final result on overlap: %s\n", neighbor_result);
        // We make things easy by copying the argument data and changing the
        // field data. This should be improved.
        OverlapDataType overlap_result = overlap_data;
        overlap_result.field_data = std::move(neighbor_result);
        result.boundary_data.emplace(mortar_id, std::move(overlap_result));
      }

      // Apply the boundary contribution to the central element
      apply_boundary_contribution(
          make_not_null(&result.element_data), numerical_fluxes_computer,
          local_boundary_data, remote_boundary_data, magnitude_of_face_normal,
          mesh, direction, mortar_mesh, mortar_size);
    }

    // Parallel::printf("Result:\n%s\n",
    //                  result.element_data);
    return result;
  }
};

/// \cond
template <typename System, typename InitialGuess, typename BoundaryConditions>
struct Metavariables {
  using system = System;
  static constexpr size_t volume_dim = system::volume_dim;
  using initial_guess = InitialGuess;
  using boundary_conditions = BoundaryConditions;

  static constexpr OptionString help{
      "Find the solution to a linear elliptic problem.\n"
      "Linear solver: GMRES\n"
      "Numerical flux: FirstOrderInternalPenaltyFlux"};

  using fluxes_computer_tag =
      elliptic::Tags::FluxesComputer<typename system::fluxes>;

  // Parse numerical flux parameters from the input file to store in the cache.
  using normal_dot_numerical_flux = Tags::NumericalFlux<
      elliptic::dg::NumericalFluxes::FirstOrderInternalPenalty<
          volume_dim, fluxes_computer_tag, typename system::primal_variables,
          typename system::auxiliary_variables>>;

  using temporal_id = LinearSolver::Tags::IterationId<LinearSolverOptions>;

  // Only Dirichlet boundary conditions are currently supported, and they are
  // are all imposed by analytic solutions right now.
  // This will be generalized ASAP. We will also support numeric initial guesses
  // and analytic initial guesses that aren't solutions ("analytic data").
  using analytic_solution_tag = Tags::AnalyticSolution<boundary_conditions>;

  // The linear solver algorithm. We must use GMRES since the operator is
  // not positive-definite for the first-order system.
  using subdomain_operator = DgSubdomainOperator<
      volume_dim, typename system::fields_tag, typename system::primal_fields,
      typename system::auxiliary_fields, fluxes_computer_tag,
      typename system::sources, normal_dot_numerical_flux>;
  using linear_solver =
      LinearSolver::Schwarz<Metavariables, typename system::fields_tag,
                            LinearSolverOptions, subdomain_operator>;
  //   using linear_solver =
  //       LinearSolver::Gmres<Metavariables, typename system::fields_tag,
  //                           LinearSolverOptions>;
  //   using preconditioner = LinearSolver::Richardson<
  //       typename linear_solver::operand_tag, PreconditionerOptions,
  //       typename linear_solver::preconditioner_source_tag>;

  // Specify the DG boundary scheme. We use the strong first-order scheme here
  // that only requires us to compute normals dotted into the first-order
  // fluxes.
  using boundary_scheme = dg::BoundarySchemes::StrongFirstOrder<
      volume_dim, typename system::variables_tag, normal_dot_numerical_flux,
      temporal_id>;

  // Collect events and triggers
  // (public for use by the Charm++ registration code)
  using observe_fields =
      db::get_variables_tags_list<typename system::fields_tag>;
  using analytic_solution_fields = observe_fields;
  using events = tmpl::list<
      dg::Events::Registrars::ObserveFields<
          volume_dim, temporal_id, observe_fields, analytic_solution_fields>,
      dg::Events::Registrars::ObserveErrorNorms<temporal_id,
                                                analytic_solution_fields>>;
  using triggers =
      tmpl::list<elliptic::Triggers::Registrars::EveryNIterations<temporal_id>>;

  // Collect all items to store in the cache.
  using const_global_cache_tags =
      tmpl::list<analytic_solution_tag, fluxes_computer_tag,
                 normal_dot_numerical_flux,
                 Tags::EventsAndTriggers<events, triggers>>;

  // Collect all reduction tags for observers
  struct element_observation_type {};
  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<tmpl::flatten<tmpl::list<
          typename Event<events>::creatable_classes, linear_solver>>>;

  // Specify all global synchronization points.
  enum class Phase { Initialization, RegisterWithObserver, Solve, Exit };

  using initialization_actions = tmpl::list<
      dg::Actions::InitializeDomain<volume_dim>,
      elliptic::Actions::InitializeSystem,
      elliptic::Actions::InitializeAnalyticSolution<analytic_solution_tag,
                                                    analytic_solution_fields>,
      dg::Actions::InitializeInterfaces<
          system,
          dg::Initialization::slice_tags_to_face<
              typename system::variables_tag>,
          dg::Initialization::slice_tags_to_exterior<>>,
      elliptic::dg::Actions::ImposeInhomogeneousBoundaryConditionsOnSource<
          Metavariables>,
      typename linear_solver::initialize_element,
      //   typename preconditioner::initialize_element,
      dg::Actions::InitializeMortars<boundary_scheme>,
      elliptic::dg::Actions::InitializeFluxes<Metavariables>,
      Initialization::Actions::RemoveOptionsAndTerminatePhase>;

  using build_linear_operator_actions = tmpl::list<
      dg::Actions::CollectDataForFluxes<boundary_scheme,
                                        ::Tags::InternalDirections<volume_dim>>,
      dg::Actions::SendDataForFluxes<boundary_scheme>,
      Actions::MutateApply<elliptic::FirstOrderOperator<
          volume_dim, LinearSolver::Tags::OperatorAppliedTo,
          typename system::variables_tag>>,
      elliptic::dg::Actions::ImposeHomogeneousDirichletBoundaryConditions<
          Metavariables>,
      dg::Actions::CollectDataForFluxes<
          boundary_scheme, ::Tags::BoundaryDirectionsInterior<volume_dim>>,
      dg::Actions::ReceiveDataForFluxes<boundary_scheme>,
      Actions::MutateApply<boundary_scheme>>;

  // Specify all parallel components that will execute actions at some point.
  using component_list = tmpl::append<
      tmpl::list<elliptic::DgElementArray<
          Metavariables,
          tmpl::list<
              Parallel::PhaseActions<Phase, Phase::Initialization,
                                     initialization_actions>,

              Parallel::PhaseActions<
                  Phase, Phase::RegisterWithObserver,
                  tmpl::list<observers::Actions::RegisterWithObservers<
                                 observers::RegisterObservers<
                                     temporal_id, element_observation_type>>,
                             // We prepare the linear solve here to avoid
                             // adding an extra phase. We can't do it
                             // before registration because it
                             // contributes to observers.
                             typename linear_solver::prepare_solve,
                             Parallel::Actions::TerminatePhase>>,

              Parallel::PhaseActions<
                  Phase, Phase::Solve,
                  tmpl::flatten<tmpl::list<
                      typename linear_solver::prepare_step,
                      Actions::RunEventsAndTriggers,
                      LinearSolver::Actions::TerminateIfConverged<
                          typename linear_solver::options_group>,
                      //   typename preconditioner::prepare_solve,
                      //   ::Actions::RepeatUntil<
                      //       LinearSolver::Tags::HasConverged<
                      //           typename preconditioner::options_group>,
                      //       tmpl::list<typename preconditioner::prepare_step,
                      //                  build_linear_operator_actions,
                      //                  typename
                      //                  preconditioner::perform_step>>,
                      build_linear_operator_actions,
                      typename linear_solver::perform_step>>>>>>,
      typename linear_solver::component_list,
      tmpl::list<observers::Observer<Metavariables>,
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
    &setup_error_handling, &domain::creators::register_derived_with_charm,
    &Parallel::register_derived_classes_with_charm<
        Event<metavariables::events>>,
    &Parallel::register_derived_classes_with_charm<
        Trigger<metavariables::triggers>>};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
/// \endcond
