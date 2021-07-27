// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <regex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags/BlockNamesAndGroups.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"
#include "IO/Observer/GetSectionObservationKey.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace tuples {
template <typename... Tags>
struct TaggedTuple;
}  // namespace tuples
namespace Parallel {
template <typename Metavariables>
struct GlobalCache;
}  // namespace Parallel
/// \endcond

namespace Elasticity::Actions {

template <size_t Dim, typename ObservationValueTag,
          typename ArraySectionIdTag = void>
struct ObservePerLayer {
  using const_global_cache_tags = tmpl::list<domain::Tags::BlockNames<Dim>>;

  using ReductionData = Parallel::ReductionData<
      // Observation value (iteration ID)
      Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
      // Volume
      Parallel::ReductionDatum<double, funcl::Plus<>>,
      // Potential energy
      Parallel::ReductionDatum<double, funcl::Plus<>>>;

  // Registration. These two aliases are used in the metavariables.
  using observed_reduction_data_tags =
      observers::make_reduction_data_tags<tmpl::list<ReductionData>>;

  struct RegisterWithObservers {
    template <typename ParallelComponent, typename DbTagsList>
    static std::pair<observers::TypeOfObservation, observers::ObservationKey>
    register_info(const db::DataBox<DbTagsList>& box,
                  const ElementId<Dim>& element_id) {
      const auto& block_name =
          db::get<domain::Tags::BlockNames<Dim>>(box).at(element_id.block_id());
      const std::optional<std::string> subfile_path = get_subfile_path(
          block_name,
          observers::get_section_observation_key<ArraySectionIdTag>(box));
      return {observers::TypeOfObservation::Reduction,
              observers::ObservationKey(subfile_path.value_or("Unused"))};
    }
  };

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    // Determine the layer to construct a reduction ID
    const auto& block_name =
        db::get<domain::Tags::BlockNames<Dim>>(box).at(element_id.block_id());
    const std::optional<std::string> subfile_path = get_subfile_path(
        block_name,
        observers::get_section_observation_key<ArraySectionIdTag>(box));
    if (not subfile_path.has_value()) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }

    // Compute integrals over the element
    const auto& mesh = db::get<domain::Tags::Mesh<Dim>>(box);
    const auto& det_inv_jacobian = db::get<
        domain::Tags::DetInvJacobian<Frame::ElementLogical, Frame::Inertial>>(
        box);
    const auto& observation_value = db::get<ObservationValueTag>(box);
    const auto& potential_energy_density =
        db::get<Elasticity::Tags::PotentialEnergyDensity<Dim>>(box);

    const DataVector det_jacobian = 1. / get(det_inv_jacobian);
    const double local_volume = definite_integral(det_jacobian, mesh);
    const double local_potential_energy =
        definite_integral(get(potential_energy_density) * det_jacobian, mesh);

    // Send data to reduction observer
    auto& local_observer = *Parallel::local_branch(
        Parallel::get_parallel_component<observers::Observer<Metavariables>>(
            cache));
    Parallel::simple_action<observers::Actions::ContributeReductionData>(
        local_observer,
        observers::ObservationId(observation_value, *subfile_path),
        observers::ArrayComponentId{
            std::add_pointer_t<ParallelComponent>{nullptr},
            Parallel::ArrayIndex<ElementId<Dim>>(element_id)},
        *subfile_path,
        std::vector<std::string>{db::tag_name<ObservationValueTag>(), "Volume",
                                 "PotentialEnergy"},
        ReductionData{static_cast<double>(observation_value), local_volume,
                      local_potential_energy});
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }

 private:
  static std::optional<std::string> get_subfile_path(
      const std::string& block_name,
      const std::optional<std::string>& section_observation_key) {
    if (not section_observation_key.has_value()) {
      // Skip the observation on elements that are not part of the section (e.g.
      // multigrid levels)
      return std::nullopt;
    }
    const std::regex layer_pattern{"Layer[0-9]+"};
    std::smatch layer_match{};
    if (not std::regex_search(block_name, layer_match, layer_pattern)) {
      // Skip the observation if the domain is not partitioned into layers
      return std::nullopt;
    }
    return "/VolumeIntegrals" + std::string{layer_match[0]} +
           *section_observation_key;
  }
};

}  // namespace Elasticity::Actions
