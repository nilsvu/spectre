// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <optional>
#include <pup.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/IndexToSliceAt.hpp"
#include "Domain/Tags.hpp"
#include "IO/Observer/GetSectionObservationKey.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/Algorithms/AlgorithmArray.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/Initialization/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/OptionalHelpers.hpp"
#include "Utilities/System/ParallelInfo.hpp"
#include "Utilities/TMPL.hpp"

#include "Parallel/Printf.hpp"

namespace Events {

namespace ObserveExcision_detail {
namespace Tags {

struct ExcisionName : db::SimpleTag {
  using type = std::string;
  static constexpr bool pass_metavariables = false;
  using option_tags = tmpl::list<>;
  static type create_from_options() { return ""; };
};

template <size_t Dim, typename TemporalIdTag, typename BoundaryFields>
struct BoundaryData : db::SimpleTag {
  using type =
      std::map<typename TemporalIdTag::type,
               std::unordered_map<ElementId<Dim>, Variables<BoundaryFields>>>;
};
}  // namespace Tags

template <size_t Dim, typename TemporalIdTag, typename BoundaryFields>
struct InitializeExcision {
  using simple_tags =
      tmpl::list<StrahlkorperTags::Strahlkorper<Frame::Inertial>,
                 Tags::BoundaryData<Dim, TemporalIdTag, BoundaryFields>>;
  using compute_tags = StrahlkorperTags::compute_items_tags<Frame::Inertial>;

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename Metavariables, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {
    const size_t l_max = 3;
    const size_t m_max = l_max;
    const std::array<double, 3> center{{0., 0., 0.}};
    const double radius = 2.;
    Strahlkorper<Frame::Inertial> strahlkorper{l_max, m_max, radius, center};
    Initialization::mutate_assign<tmpl::list<
        StrahlkorperTags::Strahlkorper<Frame::Inertial>>>(
        make_not_null(&box), std::move(strahlkorper));
    return std::make_tuple(std::move(box), true);
  }
};

template <typename Metavariables, typename TemporalIdTag,
          typename BoundaryFields, typename StrahlkorperComputeTags>
struct ExcisionObserver {
  static constexpr size_t Dim = Metavariables::volume_dim;

  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<
          ::Actions::SetupDataBox,
          InitializeExcision<Dim, TemporalIdTag, BoundaryFields>,
          ::Initialization::Actions::AddComputeTags<StrahlkorperComputeTags>>>>;
  using array_index = int;
  using const_global_cache_tags = tmpl::list<domain::Tags::Domain<Dim>>;
  using array_allocation_tags = tmpl::list<Tags::ExcisionName>;
  using initialization_tags = array_allocation_tags;

  static void allocate_array(
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache,
      const tuples::tagged_tuple_from_typelist<initialization_tags>&
          default_initialization_items) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    auto& excision_observer =
        Parallel::get_parallel_component<ExcisionObserver>(local_cache);
    const auto& domain = Parallel::get<domain::Tags::Domain<Dim>>(local_cache);
    const auto& excisions = domain.excision_spheres();
    int excision_id = 0;
    for (const auto& [excision_name, excision] : excisions) {
      auto initialization_items = default_initialization_items;
      get<Tags::ExcisionName>(initialization_items) = excision_name;
      const size_t target_proc = 0;
      excision_observer(excision_id)
          .insert(global_cache, std::move(initialization_items), target_proc);
      ++excision_id;
    }
    excision_observer.doneInserting();
  }

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::get_parallel_component<ExcisionObserver>(local_cache)
        .start_phase(next_phase);
  }
};

template <size_t Dim, typename TemporalIdTag, typename BoundaryFields>
struct ObserveExcision {
  template <
      typename ParallelComponent, typename DbTagsList, typename Metavariables,
      typename ArrayIndex, typename DataBox = db::DataBox<DbTagsList>,
      Requires<db::tag_is_retrievable_v<
          StrahlkorperTags::Strahlkorper<Frame::Inertial>, DataBox>> = nullptr>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const typename TemporalIdTag::type& temporal_id,
                    const double observation_value,
                    const ElementId<Dim>& element_id,
                    Variables<BoundaryFields>&& boundary_data) {
    // Move received boundary data into DataBox
    db::mutate<Tags::BoundaryData<Dim, TemporalIdTag, BoundaryFields>>(
        make_not_null(&box), [&temporal_id, &element_id,
                              &boundary_data](const auto local_boundary_data) {
          (*local_boundary_data)[temporal_id][element_id] =
              std::move(boundary_data);
        });

    // Check if we have received data from all elements that cover the excision
    // boundary
    const auto& name = get<Tags::ExcisionName>(box);
    // const auto& domain = db::get<domain::Tags::Domain<Dim>>(box);
    // const auto& excision = domain.excision_spheres().at(name);
    // const auto& block_neighbors = excision.block_neighbors();
    const auto& all_boundary_data =
        get<Tags::BoundaryData<Dim, TemporalIdTag, BoundaryFields>>(box).at(
            temporal_id);
    // TODO: support refinement
    if (all_boundary_data.size() < 6) {
      return;
    }

    // const auto& strahlkorper =
    //     get<StrahlkorperTags::Strahlkorper<Frame::Inertial>>(box);
    // const auto& ylm = strahlkorper.ylm_spherepack();

    auto& reduction_writer = Parallel::get_parallel_component<
        observers::ObserverWriter<Metavariables>>(cache);
    Parallel::simple_action<observers::Actions::WriteSingletonReductionData>(
        // Node 0 is always the writer, so directly call the component on that
        // node
        reduction_writer[0], std::string{"/" + name},
        std::vector<std::string>{db::tag_name<TemporalIdTag>()},
        std::make_tuple(observation_value));
  }
};

}  // namespace ObserveExcision_detail

template <size_t Dim, typename TemporalIdTag, typename ObservationValueTag,
          typename ObservableTensorTagsList, typename NonTensorComputeTagsList,
          typename StrahlkorperComputeTags, typename ArraySectionIdTag = void>
class ObserveExcisions;

template <size_t Dim, typename TemporalIdTag, typename ObservationValueTag,
          typename... ObservableTensorTags, typename... NonTensorComputeTags,
          typename StrahlkorperComputeTags, typename ArraySectionIdTag>
class ObserveExcisions<Dim, TemporalIdTag, ObservationValueTag,
                       tmpl::list<ObservableTensorTags...>,
                       tmpl::list<NonTensorComputeTags...>,
                       StrahlkorperComputeTags, ArraySectionIdTag>
    : public Event {
 public:
  static constexpr Options::String help =
      "Observe quantities on excision surfaces.";

  /// The name of the subfile inside the HDF5 file
  struct SubfileName {
    using type = std::string;
    static constexpr Options::String help = {
        "The name of the subfile inside the HDF5 file without an extension and "
        "without a preceding '/'."};
  };

  using options = tmpl::list<SubfileName>;

  explicit ObserveExcisions(CkMigrateMessage* msg) : Event(msg) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ObserveExcisions);  // NOLINT

  ObserveExcisions() = default;
  ObserveExcisions(const std::string& subfile_name)
      : subfile_path_("/" + subfile_name) {}

  using boundary_fields = tmpl::list<>;

  template <typename Metavariables>
  using excision_observer_component = ObserveExcision_detail::ExcisionObserver<
      Metavariables, TemporalIdTag, boundary_fields, StrahlkorperComputeTags>;
  template <typename Metavariables>
  using parallel_components =
      tmpl::list<excision_observer_component<Metavariables>>;

  using compute_tags_for_observation_box =
      tmpl::list<ObservableTensorTags..., NonTensorComputeTags...>;

  using argument_tags = tmpl::list<::Tags::ObservationBox>;

  template <typename ComputeTagsList, typename DataBoxType,
            typename Metavariables, typename ParallelComponent>
  void operator()(const ObservationBox<ComputeTagsList, DataBoxType>& box,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ElementId<Dim>& element_id,
                  const ParallelComponent* const /*meta*/) const {
    // Skip observation on elements that are not part of a section
    const std::optional<std::string> section_observation_key =
        observers::get_section_observation_key<ArraySectionIdTag>(box);
    if (not section_observation_key.has_value()) {
      return;
    }

    const auto& domain = get<domain::Tags::Domain<Dim>>(box);
    const auto& excisions = domain.excision_spheres();
    // const auto& block = domain.blocks()[element_id.block_id()];
    const auto& mesh = get<domain::Tags::Mesh<Dim>>(box);
    ASSERT(
        mesh.quadrature(0) == Spectral::Quadrature::GaussLobatto,
        "Only implemented for Gauss-Lobatto quadrature at the moment. Other "
        "quadratures may need an interpolation to the boundary, which should "
        "be easy enough to add as well when needed.");

    for (const auto& [excision_name, excision] : excisions) {
      const std::optional<Direction<Dim>>& radial_direction =
          excision.radial_direction(element_id);
      if (not radial_direction.has_value()) {
        continue;
      }
      // Slice vars to boundary (add interpolation to boundary here if needed)
      // const size_t slice_index =
      //     index_to_slice_at(mesh.extents(), radial_direction.value());
      // data_on_slice(face_vars, volume_vars, mesh.extents(),
      //               radial_direction.dimension(), slice_index);
      // Contribute boundary vars to Strahlkorper
      Parallel::simple_action<ObserveExcision_detail::ObserveExcision<
          Dim, TemporalIdTag, boundary_fields>>(
          Parallel::get_parallel_component<
              excision_observer_component<Metavariables>>(cache),
          get<TemporalIdTag>(box), get<ObservationValueTag>(box), element_id,
          Variables<boundary_fields>{});
    }
  }

  using is_ready_argument_tags = tmpl::list<>;

  template <typename Metavariables, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ElementId<Dim>& /*element_id*/,
                const Component* const /*meta*/) const {
    return true;
  }

  bool needs_evolved_variables() const override { return true; }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override {
    Event::pup(p);
    p | subfile_path_;
  }

 private:
  std::string subfile_path_;
};
/// @}

template <size_t Dim, typename TemporalIdTag, typename ObservationValueTag,
          typename... ObservableTensorTags, typename... NonTensorComputeTags,
          typename StrahlkorperComputeTags, typename ArraySectionIdTag>
PUP::able::PUP_ID ObserveExcisions<Dim, TemporalIdTag, ObservationValueTag,
                                   tmpl::list<ObservableTensorTags...>,
                                   tmpl::list<NonTensorComputeTags...>,
                                   StrahlkorperComputeTags,
                                   ArraySectionIdTag>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace Events
