// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <map>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Matrix.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "ErrorHandling/Assert.hpp"
#include "IO/Observer/Tags.hpp"
#include "Informer/LogActions.hpp"
#include "Informer/Tags.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
#include "Parallel/Tags.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Actions/RestrictFields.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/InterMeshOperators.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/MeshHierarchy.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Tags.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace logging {
template <typename Metavariables>
struct Logger;
}  // namespace logging
/// \endcond

namespace LinearSolver::multigrid::detail {

template <typename FieldsTag, typename OptionsGroup, typename SourceTag>
struct SendCorrectionToFinerGrid;

template <size_t Dim, typename FieldsTag, typename OptionsGroup,
          typename SourceTag>
struct InitializeElement {
 private:
  using fields_tag = FieldsTag;
  using operator_applied_to_fields_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, fields_tag>;
  using source_tag = SourceTag;

 public:
  using initialization_tags =
      tmpl::list<Tags::BaseRefinementLevels<Dim>,
                 Tags::ParentRefinementLevels<Dim>,
                 domain::Tags::InitialExtents<Dim>, Tags::MultigridLevel>;
  using initialization_tags_to_keep = tmpl::list<Tags::MultigridLevel>;
  using const_global_cache_tags =
      tmpl::list<Convergence::Tags::Iterations<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ElementId<Dim>& element_id, const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const bool is_coarsest_grid =
        get<domain::Tags::InitialRefinementLevels<Dim>>(box) ==
            get<Tags::ParentRefinementLevels<Dim>>(box) and
        get<domain::Tags::InitialExtents<Dim>>(box) ==
            get<Tags::ParentExtents<Dim>>(box);
    std::optional<ElementId<Dim>> parent_id =
        is_coarsest_grid ? std::nullopt
                         : std::make_optional(
                               LinearSolver::multigrid::parent_id(element_id));
    auto child_ids = LinearSolver::multigrid::child_ids(
        element_id,
        get<Tags::BaseRefinementLevels<Dim>>(box)[element_id.block_id()],
        get<domain::Tags::InitialRefinementLevels<Dim>>(
            box)[element_id.block_id()]);

    // Parallel::printf(
    //     "Multigrid level %zu element %s has parent %s and children %s.\n",
    //     get<Tags::MultigridLevel>(box), element_id,
    //     (parent_id ? get_output(*parent_id) : "none"), child_ids);

    auto observation_key_suffix =
        std::make_optional(get<Tags::MultigridLevel>(box) == 0
                               ? std::string{""}
                               : (std::string{"Level"} +
                                  get_output(get<Tags::MultigridLevel>(box))));

    const auto& mesh = db::get<domain::Tags::Mesh<Dim>>(box);
    // Assuming all multigrid levels have the same extents
    auto parent_mesh =
        parent_id ? domain::Initialization::create_initial_mesh(
                        db::get<Tags::ParentExtents<Dim>>(box), *parent_id)
                  : Mesh<Dim>{};
    // Parallel::printf("Mesh: %s with parent mesh %s\n", mesh, parent_mesh);

    auto restriction_operator =
        parent_id ? LinearSolver::multigrid::restriction_operator<
                        Metavariables::massive_operator>(
                        mesh, parent_mesh, element_id.segment_ids(),
                        parent_id->segment_ids())
                  : std::array<Matrix, Dim>{};

    auto prolongation_operator =
        parent_id ? LinearSolver::multigrid::prolongation_operator(
                        parent_mesh, mesh, element_id.segment_ids(),
                        parent_id->segment_ids())
                  : std::array<Matrix, Dim>{};

    const size_t num_points = mesh.number_of_grid_points();
    return std::make_tuple(
        ::Initialization::merge_into_databox<
            InitializeElement,
            db::AddSimpleTags<
                Tags::ParentElementId<Dim>, Tags::ChildElementIds<Dim>,
                Tags::ParentMesh<Dim>,
                observers::Tags::ObservationKeySuffix<Tags::MultigridLevel>,
                observers::Tags::ObservationKeySuffix<Tags::IsFinestLevel>,
                Tags::RestrictionOperator<Dim, OptionsGroup>,
                Tags::ProlongationOperator<Dim, OptionsGroup>,
                db::add_tag_prefix<Tags::PreSmoothingInitial, fields_tag>,
                db::add_tag_prefix<Tags::PreSmoothingSource, fields_tag>,
                db::add_tag_prefix<Tags::PreSmoothingResult, fields_tag>,
                db::add_tag_prefix<Tags::PreSmoothingResidual, fields_tag>,
                db::add_tag_prefix<Tags::PostSmoothingInitial, fields_tag>,
                db::add_tag_prefix<Tags::PostSmoothingSource, fields_tag>,
                db::add_tag_prefix<Tags::PostSmoothingResult, fields_tag>,
                db::add_tag_prefix<Tags::PostSmoothingResidual, fields_tag>>,
            db::AddComputeTags<
                domain::Tags::JacobianComputeFromMap<
                    domain::Tags::ElementMap<Dim>,
                    domain::Tags::Coordinates<Dim, Frame::Logical>>,
                domain::Tags::DetJacobianCompute<Dim, Frame::Logical,
                                                 Frame::Inertial>>>(
            std::move(box), std::move(parent_id), std::move(child_ids),
            std::move(parent_mesh), std::move(observation_key_suffix),
            get<Tags::MultigridLevel>(box) == 0
                ? observation_key_suffix
                : std::optional<std::string>{std::nullopt},
            std::move(restriction_operator), std::move(prolongation_operator),
            // The `PrepareSolve` action populates these
            // tags with initial values.
            db::item_type<
                db::add_tag_prefix<Tags::PreSmoothingInitial, fields_tag>>{
                num_points, 0.},
            db::item_type<
                db::add_tag_prefix<Tags::PreSmoothingSource, fields_tag>>{
                num_points, 0.},
            db::item_type<
                db::add_tag_prefix<Tags::PreSmoothingResult, fields_tag>>{
                num_points, 0.},
            db::item_type<
                db::add_tag_prefix<Tags::PreSmoothingResidual, fields_tag>>{
                num_points, 0.},
            db::item_type<
                db::add_tag_prefix<Tags::PostSmoothingInitial, fields_tag>>{
                num_points, 0.},
            db::item_type<
                db::add_tag_prefix<Tags::PostSmoothingSource, fields_tag>>{
                num_points, 0.},
            db::item_type<
                db::add_tag_prefix<Tags::PostSmoothingResult, fields_tag>>{
                num_points, 0.},
            db::item_type<
                db::add_tag_prefix<Tags::PostSmoothingResidual, fields_tag>>{
                num_points, 0.}));
  }
};

template <typename FieldsTag, typename OptionsGroup, typename SourceTag>
struct PrepareSolve {
 private:
  using fields_tag = FieldsTag;

 public:
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& inboxes,
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                 ::Verbosity::Debug)) {
      Parallel::printf(
          "%s " + Options::name<OptionsGroup>() + ": Prepare solve\n",
          element_id);
    }
    // Terminate the algorithm on coarser multigrid levels. This allows the
    // finest level to control when to terminate the algorithm. The
    // `Actions::SendFieldsToCoarserGrid` action will re-start the algorithm
    // on the coarser levels whenever they receive data. We make sure to not
    // terminate the algorithm if data has already been received from children,
    // because that means the algorithm should certainly continue.
    const bool terminate =
        (not db::get<LinearSolver::multigrid::Tags::IsFinestLevel>(box)) and
        (not(tuples::get<DataFromChildrenInboxTag<Dim, SourceTag>>(inboxes)
                 .count(0) > 0));
    return {std::move(box), terminate};
  }
};

template <typename FieldsTag, typename OptionsGroup, typename SourceTag>
using SendResidualToCoarserGrid = Actions::SendFieldsToCoarserGrid<
    db::add_tag_prefix<LinearSolver::Tags::Residual, FieldsTag>, OptionsGroup,
    SourceTag>;

template <size_t Dim, typename FieldsTag, typename OptionsGroup,
          typename SourceTag>
using ReceiveResidualFromFinerGrid = Actions::ReceiveFieldsFromFinerGrid<
    Dim, db::add_tag_prefix<LinearSolver::Tags::Residual, FieldsTag>,
    OptionsGroup, SourceTag>;

template <typename OptionsGroup>
struct PreSmoothingLogFormatter {
  static std::string apply(const size_t iteration_id, const size_t mg_level,
                           const bool is_coarsest_level) noexcept {
    return "'" + Options::name<OptionsGroup>() + "' iteration " +
           get_output(iteration_id + 1) +
           (is_coarsest_level ? " bottom-smoothing " : " pre-smoothing ") +
           "done on multigrid-level " + get_output(mg_level) + ".";
  }
};

template <typename FieldsTag, typename OptionsGroup, typename SourceTag>
struct PreparePreSmoothing {
 private:
  using fields_tag = FieldsTag;
  using operator_applied_to_fields_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, fields_tag>;
  using source_tag = SourceTag;

 public:
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                 ::Verbosity::Debug)) {
      Parallel::printf(
          "%s " + Options::name<OptionsGroup>() + ": Prepare pre-smoothing\n",
          element_id);
    }

    // On coarser grids the smoother solves for a correction to the finer-grid
    // fields, so we set its initial guess to zero. On the finest grid we smooth
    // the fields directly, so there's nothing to prepare.
    if (not get<Tags::IsFinestLevel>(box)) {
      db::mutate<fields_tag, operator_applied_to_fields_tag>(
          make_not_null(&box),
          [](const auto fields, const auto operator_applied_to_fields,
             const db::const_item_type<source_tag>& source) noexcept {
            *fields = make_with_value<db::item_type<fields_tag>>(source, 0.);
            // We can set the linear operator applied to the initial fields to
            // zero as well, since it's linear. This may save the smoother an
            // operator application on coarser grids if it's optimized for this.
            *operator_applied_to_fields =
                make_with_value<db::item_type<operator_applied_to_fields_tag>>(
                    source, 0.);
          },
          db::get<source_tag>(box));
    }

    // Record pre-smoothing fields for debugging
    db::mutate<db::add_tag_prefix<Tags::PreSmoothingInitial, fields_tag>>(
        make_not_null(&box),
        [](const auto x, const auto& y) noexcept { *x = y; },
        db::get<fields_tag>(box));
    db::mutate<db::add_tag_prefix<Tags::PreSmoothingSource, fields_tag>>(
        make_not_null(&box),
        [](const auto x, const auto& y) noexcept { *x = y; },
        db::get<source_tag>(box));

    return {std::move(box)};
  }
};

template <typename FieldsTag, typename OptionsGroup, typename SourceTag>
struct SkipPostsmoothingAtBottom {
 private:
  using fields_tag = FieldsTag;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>;

 public:
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool, size_t> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const bool is_coarsest_level =
        not get<Tags::ParentElementId<Dim>>(box).has_value();

    // Do some logging
    if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(cache) >=
                 ::Verbosity::Verbose)) {
      Parallel::contribute_to_reduction<
          logging::Actions::Log<PreSmoothingLogFormatter<OptionsGroup>>,
          ParallelComponent, Tags::MultigridLevel>(
          Parallel::ReductionData<
              Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
              Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
              Parallel::ReductionDatum<bool, funcl::AssertEqual<>>>{
              get<Convergence::Tags::IterationId<OptionsGroup>>(box),
              get<Tags::MultigridLevel>(box), is_coarsest_level},
          Parallel::get_parallel_component<ParallelComponent>(
              cache)[element_id],
          Parallel::get_parallel_component<logging::Logger<Metavariables>>(
              cache),
          *get<Parallel::Tags::SectionBase<Tags::MultigridLevel>>(box),
          get<Tags::MultigridLevel>(box));
    }

    // Record pre-smoothing result for debugging
    db::mutate<db::add_tag_prefix<Tags::PreSmoothingResult, fields_tag>>(
        make_not_null(&box),
        [](const auto x, const auto& y) noexcept { *x = y; },
        db::get<fields_tag>(box));
    db::mutate<db::add_tag_prefix<Tags::PreSmoothingResidual, fields_tag>>(
        make_not_null(&box),
        [](const auto x, const auto& y) noexcept { *x = y; },
        db::get<residual_tag>(box));

    // On the coarsest grid skip the second smoothing step
    const size_t first_action_after_post_smoothing_index = tmpl::index_of<
        ActionList,
        SendCorrectionToFinerGrid<FieldsTag, OptionsGroup, SourceTag>>::value;
    const size_t this_action_index =
        tmpl::index_of<ActionList, SkipPostsmoothingAtBottom>::value;
    return {std::move(box), false,
            is_coarsest_level ? first_action_after_post_smoothing_index
                              : (this_action_index + 1)};
  }
};

template <typename FieldsTag>
struct CorrectionInboxTag
    : public Parallel::InboxInserters::Value<CorrectionInboxTag<FieldsTag>> {
  using temporal_id = size_t;
  using type = std::map<temporal_id, db::item_type<FieldsTag>>;
};

template <typename OptionsGroup>
struct PostSmoothingLogFormatter {
  static std::string apply(const size_t iteration_id,
                           const size_t mg_level) noexcept {
    return "'" + Options::name<OptionsGroup>() + "' iteration " +
           get_output(iteration_id + 1) +
           " post-smoothing done on multigrid-level " + get_output(mg_level) +
           ".";
  }
};

template <typename FieldsTag, typename OptionsGroup, typename SourceTag>
struct SendCorrectionToFinerGrid {
 private:
  using fields_tag = FieldsTag;

 public:
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const bool is_coarsest_level =
        not get<Tags::ParentElementId<Dim>>(box).has_value();

    if (not is_coarsest_level and
        UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(cache) >=
                 ::Verbosity::Verbose)) {
      Parallel::contribute_to_reduction<
          logging::Actions::Log<PostSmoothingLogFormatter<OptionsGroup>>,
          ParallelComponent, Tags::MultigridLevel>(
          Parallel::ReductionData<
              Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
              Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>>{
              get<Convergence::Tags::IterationId<OptionsGroup>>(box),
              get<Tags::MultigridLevel>(box)},
          Parallel::get_parallel_component<ParallelComponent>(
              cache)[element_id],
          Parallel::get_parallel_component<logging::Logger<Metavariables>>(
              cache),
          *get<Parallel::Tags::SectionBase<Tags::MultigridLevel>>(box),
          get<Tags::MultigridLevel>(box));
    }

    const auto& child_ids = get<Tags::ChildElementIds<Dim>>(box);
    db::mutate<db::add_tag_prefix<Tags::PostSmoothingResult, fields_tag>>(
        make_not_null(&box),
        [](const auto x, const auto& y) noexcept { *x = y; },
        db::get<fields_tag>(box));
    db::mutate<db::add_tag_prefix<Tags::PostSmoothingResidual, fields_tag>>(
        make_not_null(&box),
        [](const auto x, const auto& y) noexcept { *x = y; },
        db::get<db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>>(
            box));
    if (child_ids.empty()) {
      return {std::move(box)};
    }
    auto& receiver_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);
    const auto& temporal_id =
        db::get<Convergence::Tags::IterationId<OptionsGroup>>(box);

    if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                 ::Verbosity::Debug)) {
      Parallel::printf("%s " + Options::name<OptionsGroup>() +
                           "(%zu): Send correction to children\n",
                       element_id, temporal_id);
    }

    for (const auto& child_id : child_ids) {
      Parallel::receive_data<CorrectionInboxTag<FieldsTag>>(
          receiver_proxy[child_id], temporal_id, get<fields_tag>(box));
    }
    return {std::move(box)};
  }
};

template <size_t Dim, typename FieldsTag, typename OptionsGroup,
          typename SourceTag>
struct ReceiveCorrectionFromCoarserGrid {
 private:
  using fields_tag = FieldsTag;

 public:
  using inbox_tags = tmpl::list<CorrectionInboxTag<FieldsTag>>;

  template <typename DbTags, typename... InboxTags, typename Metavariables>
  static bool is_ready(const db::DataBox<DbTags>& box,
                       const tuples::TaggedTuple<InboxTags...>& inboxes,
                       const Parallel::GlobalCache<Metavariables>& /*cache*/,
                       const ElementId<Dim>& /*element_id*/) noexcept {
    const auto& parent_id = get<Tags::ParentElementId<Dim>>(box);
    if (not parent_id) {
      return true;
    }
    const auto& inbox = tuples::get<CorrectionInboxTag<FieldsTag>>(inboxes);
    const auto& temporal_id =
        db::get<Convergence::Tags::IterationId<OptionsGroup>>(box);
    const auto temporal_received = inbox.find(temporal_id);
    if (temporal_received == inbox.end()) {
      return false;
    }
    return true;
  }

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& parent_id = get<Tags::ParentElementId<Dim>>(box);
    // We should always have a `parent_id` at this point because we skip this
    // part of the algorithm on the coarsest grid with the
    // `SkipPostsmoothingAtBottom` action
    ASSERT(parent_id,
           "Trying to receive data from parent but no parent is set on element "
               << element_id << ".");

    auto& inbox = tuples::get<CorrectionInboxTag<FieldsTag>>(inboxes);
    const auto& temporal_id =
        db::get<Convergence::Tags::IterationId<OptionsGroup>>(box);
    const auto temporal_received = inbox.find(temporal_id);
    const auto& parent_correction = temporal_received->second;

    if (UNLIKELY(get<logging::Tags::Verbosity<OptionsGroup>>(box) >=
                 ::Verbosity::Debug)) {
      Parallel::printf("%s " + Options::name<OptionsGroup>() +
                           "(%zu): Prolongate correction from parent\n",
                       element_id, temporal_id);
    }

    // Apply prolongation operator
    // TODO: Do nothing when parent is the same element
    db::mutate<fields_tag>(
        make_not_null(&box),
        [&](const gsl::not_null<db::item_type<fields_tag>*> fields,
            const std::array<Matrix, Dim>& prolongation_operator,
            const Mesh<Dim>& parent_mesh) noexcept {
          *fields += apply_matrices(prolongation_operator, parent_correction,
                                    parent_mesh.extents());
        },
        db::get<Tags::ProlongationOperator<Dim, OptionsGroup>>(box),
        db::get<Tags::ParentMesh<Dim>>(box));

    db::mutate<db::add_tag_prefix<Tags::PostSmoothingInitial, fields_tag>>(
        make_not_null(&box),
        [](const auto x, const auto& y) noexcept { *x = y; },
        db::get<fields_tag>(box));
    db::mutate<db::add_tag_prefix<Tags::PostSmoothingSource, fields_tag>>(
        make_not_null(&box),
        [](const auto x, const auto& y) noexcept { *x = y; },
        db::get<SourceTag>(box));

    inbox.erase(temporal_received);
    return {std::move(box)};
  }
};

}  // namespace LinearSolver::multigrid::detail
