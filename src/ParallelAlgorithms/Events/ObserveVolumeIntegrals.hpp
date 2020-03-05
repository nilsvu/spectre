// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Tags.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

namespace dg {
namespace Events {
template <size_t VolumeDim, typename ObservationValueTag, typename Tensors,
          typename EventRegistrars>
class ObserveVolumeIntegrals;

namespace Registrars {
template <size_t VolumeDim, typename ObservationValueTag, typename Tensors>
// Presence of size_t template argument requires to define this struct
// instead of using Registration::Registrar alias.
struct ObserveVolumeIntegrals {
  template <typename RegistrarList>
  using f = Events::ObserveVolumeIntegrals<VolumeDim, ObservationValueTag,
                                           Tensors, RegistrarList>;
};
}  // namespace Registrars

template <size_t VolumeDim, typename ObservationValueTag, typename Tensors,
          typename EventRegistrars =
              tmpl::list<Registrars::ObserveVolumeIntegrals<
                  VolumeDim, ObservationValueTag, Tensors>>>
class ObserveVolumeIntegrals;

/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief %Observe the volume integrals of the tensors over the domain.
 *
 * Writes reduction quantities:
 * - `ObservationValueTag`
 * - `Volume` = volume of the domain
 * - `VolInt(*)` = volume integral of the tensor
 *
 * \warning Currently, only one reduction observation event can be
 * triggered at a given observation value.  Causing multiple events to run at
 * once will produce unpredictable results.
 */
template <size_t VolumeDim, typename ObservationValueTag, typename... Tensors,
          typename EventRegistrars>
class ObserveVolumeIntegrals<VolumeDim, ObservationValueTag,
                             tmpl::list<Tensors...>, EventRegistrars>
    : public Event<EventRegistrars> {
 private:
  static constexpr size_t num_tensor_components =
      tmpl::fold<tmpl::integral_list<size_t, db::item_type<Tensors>::size()...>,
                 tmpl::size_t<0>,
                 tmpl::plus<tmpl::_state, tmpl::_element>>::value;

  using VolumeIntegralDatum = Parallel::ReductionDatum<double, funcl::Plus<>>;
  using ReductionData = tmpl::wrap<
      tmpl::flatten<tmpl::list<
          Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
          VolumeIntegralDatum,
          tmpl::filled_list<VolumeIntegralDatum, num_tensor_components>>>,
      Parallel::ReductionData>;

  template <size_t... Is>
  static ReductionData make_reduction_data(
      const double observation_value, const double local_volume,
      std::array<double, num_tensor_components>&& local_volume_integrals,
      std::index_sequence<Is...> /*meta*/) noexcept {
    return ReductionData{observation_value, local_volume,
                         local_volume_integrals.at(Is)...};
  }

  template <typename T>
  static std::string component_suffix(const T& tensor,
                                      size_t component_index) noexcept {
    return tensor.rank() == 0
               ? ""
               : "_" + tensor.component_name(
                           tensor.get_tensor_index(component_index));
  }

 public:
  /// \cond
  explicit ObserveVolumeIntegrals(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ObserveVolumeIntegrals);  // NOLINT
  /// \endcond

  using options = tmpl::list<>;
  static constexpr OptionString help =
      "Observe the volume integrals of the tensors over the domain.\n"
      "\n"
      "Writes reduction quantities:\n"
      " * ObservationValueTag\n"
      " * Volume = volume of the domain\n"
      " * VolInt(*) = volume integral of the tensor\n"
      "\n"
      "Warning: Currently, only one reduction observation event can be\n"
      "triggered at a given observation value.  Causing multiple events to\n"
      "run at once will produce unpredictable results.";

  ObserveVolumeIntegrals() = default;

  using observed_reduction_data_tags =
      observers::make_reduction_data_tags<tmpl::list<ReductionData>>;

  using argument_tags =
      tmpl::list<ObservationValueTag, domain::Tags::Mesh<VolumeDim>,
                 domain::Tags::ElementMap<VolumeDim, Frame::Inertial>,
                 domain::Tags::Coordinates<VolumeDim, Frame::Logical>,
                 Tensors...>;

  template <typename Metavariables, typename ArrayIndex,
            typename ParallelComponent>
  void operator()(
      const db::const_item_type<ObservationValueTag>& observation_value,
      const Mesh<VolumeDim>& mesh,
      const ElementMap<VolumeDim, Frame::Inertial>& element_map,
      const tnsr::I<DataVector, VolumeDim, Frame::Logical>& logical_coordinates,
      const db::const_item_type<Tensors>&... tensors,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/,
      const ParallelComponent* const /*meta*/) const noexcept {
    // Determinant of Jacobian is needed because integral is performed in
    // logical coords. Currently not initialized in the Metavariables.
    const DataVector det_jacobian =
        get(determinant(element_map.jacobian(logical_coordinates)));
    const double local_volume = definite_integral(det_jacobian, mesh);

    std::array<double, num_tensor_components> local_volume_integrals{};
    std::vector<std::string> reduction_names = {
        db::tag_name<ObservationValueTag>(), "Volume"};
    size_t integral_index = 0;
    const auto record_integrals = [&local_volume_integrals, &reduction_names,
                                   &det_jacobian, &mesh, &integral_index](
                                      const auto tensor_tag_v,
                                      const auto& tensor) noexcept {
      using tensor_tag = tmpl::type_from<decltype(tensor_tag_v)>;
      for (size_t i = 0; i < tensor.size(); ++i) {
        reduction_names.push_back("VolInt(" + db::tag_name<tensor_tag>() +
                                  component_suffix(tensor, i) + ")");
        local_volume_integrals[integral_index] =
            definite_integral(det_jacobian * tensor[i], mesh);
        integral_index++;
      }
      return 0;
    };
    expand_pack(record_integrals(tmpl::type_<Tensors>{}, tensors)...);

    // Send data to reduction observer
    auto& local_observer =
        *Parallel::get_parallel_component<observers::Observer<Metavariables>>(
             cache)
             .ckLocalBranch();
    Parallel::simple_action<observers::Actions::ContributeReductionData>(
        local_observer,
        observers::ObservationId(
            observation_value,
            typename Metavariables::element_observation_type{}),
        std::string{"/element_data"}, reduction_names,
        make_reduction_data(static_cast<double>(observation_value),
                            local_volume, std::move(local_volume_integrals),
                            std::make_index_sequence<num_tensor_components>{}));
  }
};

/// \cond
template <size_t VolumeDim, typename ObservationValueTag, typename... Tensors,
          typename EventRegistrars>
PUP::able::PUP_ID ObserveVolumeIntegrals<VolumeDim, ObservationValueTag,
                                         tmpl::list<Tensors...>,
                                         EventRegistrars>::my_PUP_ID =
    0;  // NOLINT
/// \endcond
}  // namespace Events
}  // namespace dg
