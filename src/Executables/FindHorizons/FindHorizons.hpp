// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>
#include <string>

#include "ApparentHorizons/HorizonAliases.hpp"
#include "ApparentHorizons/ObjectLabel.hpp"
#include "ApparentHorizons/ObserveCenters.hpp"
#include "ApparentHorizons/StrahlkorperGr.hpp"
#include "ApparentHorizons/Tags.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "Domain/Creators/Factory3D.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Protocols/Metavariables.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Evolution/Initialization/DgDomain.hpp"
#include "IO/Importers/Actions/ReadVolumeData.hpp"
#include "IO/Importers/Actions/ReceiveVolumeData.hpp"
#include "IO/Importers/Actions/RegisterWithElementDataReader.hpp"
#include "IO/Importers/ElementDataReader.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Algorithms/AlgorithmArray.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/CleanUpInterpolator.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InitializeInterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolationTargetReceiveVars.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolatorReceivePoints.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolatorReceiveVolumeData.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolatorRegisterElement.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/TryToInterpolate.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ErrorOnFailedApparentHorizon.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/FindApparentHorizon.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ObserveSurfaceData.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ObserveTimeSeriesOnSurface.hpp"
#include "ParallelAlgorithms/Interpolation/Events/Interpolate.hpp"
#include "ParallelAlgorithms/Interpolation/Interpolate.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTarget.hpp"
#include "ParallelAlgorithms/Interpolation/Interpolator.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/ComputeVarsToInterpolate.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "ParallelAlgorithms/Interpolation/Targets/ApparentHorizon.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Transform.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/Overloader.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace FindHorizons {

namespace OptionTags {
struct VolumeDataGroup {
  static std::string name() { return "VolumeData"; }
  static constexpr Options::String help =
      "Volume data to load and find horizons in";
  using group = importers::OptionTags::Group;
};
}  // namespace OptionTags

namespace Actions {

template <size_t Dim, typename FieldsTagsList>
struct InitializeFields {
  using simple_tags =
      tmpl::list<::Tags::Time, ::Tags::Variables<FieldsTagsList>>;
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    // For now, set the time to the observation value at which we want to read
    // data, or to zero if it's determined from the available observations
    // automatically. The time is currently only used to index interpolations to
    // the AH finder. We can generalize this to step through the available
    // observations in the file, handle time-dependent domains, etc. To do that,
    // the importer will have to send information about the observation value it
    // read from the file, as well as the available observation IDs/values in
    // the file.
    const double observation_time = std::visit(
        Overloader{
            [](const double local_obs_value) { return local_obs_value; },
            [](const importers::ObservationSelector /*local_obs_selector*/) {
              return 0.;
            }},
        db::get<importers::Tags::ObservationValue<OptionTags::VolumeDataGroup>>(
            box));
    ::Initialization::mutate_assign<tmpl::list<::Tags::Time>>(
        make_not_null(&box), observation_time);
    // Nothing to do to initialize the fields. They will be read from the
    // volume data file.
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

// Send volume data to the interpolator, which will trigger an apparent horizon
// find
template <typename InterpolationTargetTag>
struct DispatchApparentHorizonFinder {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<Dim>& element_id, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    intrp::interpolate<InterpolationTargetTag>(
        db::get<::Tags::Time>(box), db::get<domain::Tags::Mesh<Dim>>(box),
        cache, element_id,
        db::get<gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataVector>>(box),
        db::get<gr::Tags::ExtrinsicCurvature<Dim, Frame::Inertial, DataVector>>(
            box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

}  // namespace Actions

template <size_t Dim>
struct ComputeHorizonVolumeQuantities
    : tt::ConformsTo<intrp::protocols::ComputeVarsToInterpolate> {
  using allowed_src_tags = tmpl::list<
      gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataVector>,
      gr::Tags::ExtrinsicCurvature<Dim, Frame::Inertial, DataVector>>;
  using required_src_tags = allowed_src_tags;
  template <typename TargetFrame>
  using allowed_dest_tags =
      tmpl::list<gr::Tags::SpatialMetric<Dim, TargetFrame>,
                 gr::Tags::InverseSpatialMetric<Dim, TargetFrame>,
                 gr::Tags::ExtrinsicCurvature<Dim, TargetFrame>,
                 gr::Tags::SpatialChristoffelSecondKind<Dim, TargetFrame>,
                 gr::Tags::SpatialRicci<Dim, TargetFrame>>;
  template <typename TargetFrame>
  using required_dest_tags = allowed_dest_tags<TargetFrame>;

  template <typename SrcTagList, typename DestTagList>
  static void apply(
      const gsl::not_null<Variables<DestTagList>*> target_vars,
      const Variables<SrcTagList>& src_vars, const Mesh<Dim>& mesh,
      const Jacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>& jacobian,
      const InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Grid>&
          inv_jacobian) {
    // Inertial-frame (src) quantities
    const auto& inertial_spatial_metric =
        get<gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataVector>>(
            src_vars);
    const auto& inertial_ext_curvature =
        get<gr::Tags::ExtrinsicCurvature<Dim, Frame::Inertial, DataVector>>(
            src_vars);
    // Grid-frame (target) quantities
    auto& grid_spatial_metric =
        get<gr::Tags::SpatialMetric<Dim, Frame::Grid, DataVector>>(
            *target_vars);
    auto& grid_ext_curvature =
        get<gr::Tags::ExtrinsicCurvature<Dim, Frame::Grid, DataVector>>(
            *target_vars);
    auto& grid_inv_spatial_metric =
        get<gr::Tags::InverseSpatialMetric<Dim, Frame::Grid, DataVector>>(
            *target_vars);
    auto& spatial_christoffel_second_kind = get<
        gr::Tags::SpatialChristoffelSecondKind<Dim, Frame::Grid, DataVector>>(
        *target_vars);
    auto& spatial_ricci =
        get<gr::Tags::SpatialRicci<Dim, Frame::Grid>>(*target_vars);
    // Transform quantities from inertial to grid frame
    transform::to_different_frame(make_not_null(&grid_spatial_metric),
                                  inertial_spatial_metric, jacobian);
    transform::to_different_frame(make_not_null(&grid_ext_curvature),
                                  inertial_ext_curvature, jacobian);
    // Compute derived quantities
    Scalar<DataVector> unused_det{mesh.number_of_grid_points()};
    determinant_and_inverse(make_not_null(&unused_det),
                            make_not_null(&grid_inv_spatial_metric),
                            grid_spatial_metric);
    const auto deriv_spatial_metric =
        ::partial_derivative(grid_spatial_metric, mesh, inv_jacobian);
    gr::christoffel_second_kind(make_not_null(&spatial_christoffel_second_kind),
                                deriv_spatial_metric, grid_inv_spatial_metric);
    const auto deriv_spatial_christoffel_second_kind = ::partial_derivative(
        spatial_christoffel_second_kind, mesh, inv_jacobian);
    gr::ricci_tensor(make_not_null(&spatial_ricci),
                     spatial_christoffel_second_kind,
                     deriv_spatial_christoffel_second_kind);
  }
};

template <size_t Dim, ah::ObjectLabel Label>
struct ApparentHorizon
    : tt::ConformsTo<intrp::protocols::InterpolationTargetTag> {
  static std::string name() { return "Ah" + ah::name(Label); }
  using temporal_id = ::Tags::Time;
  using compute_target_points =
      intrp::TargetPoints::ApparentHorizon<ApparentHorizon, Frame::Grid>;
  using post_interpolation_callback =
      intrp::callbacks::FindApparentHorizon<ApparentHorizon, Frame::Grid>;
  using horizon_find_failure_callback =
      intrp::callbacks::ErrorOnFailedApparentHorizon;
  using post_horizon_find_callbacks = tmpl::list<
      intrp::callbacks::ObserveTimeSeriesOnSurface<
          ::ah::tags_for_observing<Frame::Grid>, ApparentHorizon>,
      intrp::callbacks::ObserveSurfaceData<::ah::surface_tags_for_observing,
                                           ApparentHorizon, ::Frame::Grid>,
      ah::callbacks::ObserveCenters<ApparentHorizon>>;

  using vars_to_interpolate_to_target =
      ::ah::vars_to_interpolate_to_target<Dim, Frame::Grid>;
  using compute_vars_to_interpolate = ComputeHorizonVolumeQuantities<Dim>;
  using compute_items_on_target =
      ::ah::compute_items_on_target<Dim, Frame::Grid>;
};

template <size_t Dim, bool TwoHorizons>
struct Metavariables {
  static constexpr size_t volume_dim = Dim;

  static constexpr Options::String help{
      "Find apparent horizons in volume data."};

  // A placeholder system for the domain creators
  struct system {};

  using AhA = ApparentHorizon<Dim, ah::ObjectLabel::A>;
  using AhB = ApparentHorizon<Dim, ah::ObjectLabel::B>;
  static constexpr bool two_horizons = TwoHorizons;

  struct domain : tt::ConformsTo<::domain::protocols::Metavariables> {
    static constexpr bool enable_time_dependent_maps = false;
  };

  using const_global_cache_tags = tmpl::list<>;

  using adm_vars = tmpl::list<
      gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<Dim, Frame::Inertial, DataVector>,
      gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataVector>,
      gr::Tags::ExtrinsicCurvature<Dim, Frame::Inertial, DataVector>>;

  using interpolator_source_vars =
      typename ComputeHorizonVolumeQuantities<Dim>::required_src_tags;
  using interpolation_target_tags = tmpl::flatten<
      tmpl::list<AhA, tmpl::conditional_t<two_horizons, AhB, tmpl::list<>>>>;
  using temporal_id = ::Tags::Time;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<DomainCreator<Dim>, domain_creators<Dim>>>;
  };

  using element_array = ::DgElementArray<
      Metavariables,
      tmpl::list<
          Parallel::PhaseActions<
              Parallel::Phase::Initialization,
              tmpl::list<
                  evolution::dg::Initialization::Domain<Dim>,
                  Actions::InitializeFields<Dim, adm_vars>,
                  ::Initialization::Actions::RemoveOptionsAndTerminatePhase>>,
          Parallel::PhaseActions<
              Parallel::Phase::Register,
              tmpl::list<importers::Actions::RegisterWithElementDataReader,
                         intrp::Actions::RegisterElementWithInterpolator,
                         Parallel::Actions::TerminatePhase>>,
          Parallel::PhaseActions<
              Parallel::Phase::Execute,
              tmpl::list<
                  importers::Actions::ReadVolumeData<
                      OptionTags::VolumeDataGroup, adm_vars>,
                  importers::Actions::ReceiveVolumeData<
                      OptionTags::VolumeDataGroup, adm_vars>,
                  Actions::DispatchApparentHorizonFinder<AhA>,
                  tmpl::conditional_t<
                      two_horizons, Actions::DispatchApparentHorizonFinder<AhB>,
                      tmpl::list<>>,
                  Parallel::Actions::TerminatePhase>>>>;

  using component_list = tmpl::flatten<tmpl::list<
      element_array, importers::ElementDataReader<Metavariables>,
      intrp::Interpolator<Metavariables>,
      intrp::InterpolationTarget<Metavariables, AhA>,
      tmpl::conditional_t<two_horizons,
                          intrp::InterpolationTarget<Metavariables, AhB>,
                          tmpl::list<>>,
      observers::Observer<Metavariables>,
      observers::ObserverWriter<Metavariables>>>;

  using observed_reduction_data_tags = tmpl::list<>;

  static constexpr std::array<Parallel::Phase, 5> default_phase_order{
      {Parallel::Phase::Initialization, Parallel::Phase::Register,
       Parallel::Phase::Execute, Parallel::Phase::Exit}};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};

}  // namespace FindHorizons

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &setup_memory_allocation_failure_reporting,
    &disable_openblas_multithreading,
    &domain::creators::register_derived_with_charm,
    &Parallel::register_factory_classes_with_charm<metavariables>};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
/// \endcond
