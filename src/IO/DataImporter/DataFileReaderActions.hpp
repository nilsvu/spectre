// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <iterator>
#include <unordered_map>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/TensorData.hpp"
#include "ErrorHandling/Error.hpp"
#include "IO/DataImporter/Tags.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/VolumeData.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TaggedTuple.hpp"

#include "Parallel/Printf.hpp"

namespace importer {
/// Actions related to the data importer
namespace Actions {

/*!
 * \brief Invoked on the `importer::DataFileReader` component to store the
 * registered data.
 *
 * The `importer::Actions::RegisterWithImporter` action, which is performed on
 * each element of an array parallel component, invokes this action on the
 * `importer::DataFileReader` component.
 */
struct RegisterElementWithSelf {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<sizeof...(DbTags) != 0> = nullptr>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const observers::ArrayComponentId& array_component_id,
                    const std::string& grid_name) noexcept {
    Parallel::printf("Node %zu registered element %s\n",
                     static_cast<size_t>(Parallel::my_node()), grid_name);
    db::mutate<Tags::RegisteredElements>(
        make_not_null(&box),
        [&array_component_id, &grid_name ](
            const gsl::not_null<db::item_type<Tags::RegisteredElements>*>
                registered_elements) noexcept {
          (*registered_elements)[array_component_id] = grid_name;
        });
  }
};

}  // namespace Actions

/// Threaded actions related to the data importer
namespace ThreadedActions {

/*!
 * \brief Read a volume data file and distribute the data to the registered
 * elements.
 *
 * This action can be invoked on the `importer::DataFileReader` component once
 * all elements have been registered in a previous phase. It opens the data
 * file, reads the data for each registered element in turn and calls the
 * `CallbackAction` on each element providing the data.
 *
 * - The `ImporterOptionsGroup` parameter specifies the \ref OptionGroupsGroup
 * "options group" in the input file that provides the following run-time
 * options:
 *   - `importer::OptionTags::DataFileName`
 *   - `importer::OptionTags::VolumeDataSubgroup`
 *   - `importer::OptionTags::ObservationValue`
 * - The `FieldTagsList` parameter specifies a typelist of tensor tags that
 * are read from the file and provided to each element. It is assumed that the
 * tensor data is stored in datasets named `Tag::name() + suffix`, where the
 * `suffix` is empty for scalars or `"_"` followed by the
 * `Tensor::component_name` for each independent tensor component.
 * - The `CallbackAction` is invoked on each registered element of the
 * `CallbackComponent` with a
 * `tuples::tagged_tuple_from_typelist<FieldTagsList>` containing the
 * tensor data for that element. The `CallbackComponent` must the the same that
 * was encoded into the `observers::ArrayComponentId` used to register the
 * elements.
 */
template <typename ImporterOptionsGroup, typename FieldTagsList,
          typename CallbackAction, typename CallbackComponent>
struct ReadElementData {
 private:
  template <typename T>
  static std::string component_suffix(const T& tensor,
                                      size_t component_index) noexcept {
    return tensor.rank() == 0
               ? ""
               : "_" + tensor.component_name(
                           tensor.get_tensor_index(component_index));
  }
  static size_t get_observation_id(const h5::VolumeData& volume_file,
                                   const double observation_value) noexcept {
    for (auto& observation_id : volume_file.list_observation_ids()) {
      if (volume_file.get_observation_value(observation_id) ==
          observation_value) {
        return observation_id;
      }
    }
    ERROR("No observation with value " << observation_value
                                       << " found in volume file.");
  }

 public:
  using const_global_cache_tag_list =
      tmpl::list<OptionTags::DataFileName<ImporterOptionsGroup>,
                 OptionTags::VolumeDataSubgroup<ImporterOptionsGroup>,
                 OptionTags::ObservationValue<ImporterOptionsGroup>>;

  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<sizeof...(DbTags) != 0> = nullptr>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const gsl::not_null<CmiNodeLock*> node_lock) noexcept {
    Parallel::printf("Node %zu reads element data...\n",
                     static_cast<size_t>(Parallel::my_node()));
    Parallel::lock(node_lock);
    {
      // The scoping is to close the file before unlocking
      h5::H5File<h5::AccessType::ReadOnly> h5file(
          Parallel::get<OptionTags::DataFileName<ImporterOptionsGroup>>(cache));
      constexpr size_t version_number = 0;
      const auto& volume_file = h5file.get<h5::VolumeData>(
          "/" +
              Parallel::get<
                  OptionTags::VolumeDataSubgroup<ImporterOptionsGroup>>(cache),
          version_number);
      const auto observation_id = get_observation_id(
          volume_file,
          Parallel::get<OptionTags::ObservationValue<ImporterOptionsGroup>>(
              cache));
      for (auto& element_and_name : get<Tags::RegisteredElements>(box)) {
        const CkArrayIndex& raw_element_index =
            element_and_name.first.array_index();
        // Check if the parallel component of the registered element matches the
        // callback, because it's possible that elements from other components
        // with the same index are also registered.
        // Since the way the component is encoded in `ArrayComponentId` is
        // private to that class, we construct one and compare.
        if (element_and_name.first ==
            observers::ArrayComponentId(
                std::add_pointer_t<CallbackComponent>{nullptr},
                raw_element_index)) {
          const auto& grid_name = element_and_name.second;
          tuples::tagged_tuple_from_typelist<FieldTagsList> element_data{};
          tmpl::for_each<FieldTagsList>([
            &element_data, &volume_file, &observation_id, &grid_name
          ](auto field_tag_v) noexcept {
            using field_tag = tmpl::type_from<decltype(field_tag_v)>;
            auto& tensor_data = get<field_tag>(element_data);
            for (size_t i = 0; i < tensor_data.size(); i++) {
              tensor_data[i] = volume_file.get_tensor_component(
                  observation_id, grid_name,
                  field_tag::name() + component_suffix(tensor_data, i));
            }
          });
          const auto element_index =
              Parallel::ArrayIndex<typename CallbackComponent::array_index>(
                  raw_element_index)
                  .get_index();
          Parallel::printf("read %s data\n", element_index);
          Parallel::simple_action<CallbackAction>(
              Parallel::get_parallel_component<CallbackComponent>(
                  cache)[element_index],
              std::move(element_data));
        }
      }
    }
    Parallel::unlock(node_lock);
  }
};

}  // namespace ThreadedActions
}  // namespace importer
