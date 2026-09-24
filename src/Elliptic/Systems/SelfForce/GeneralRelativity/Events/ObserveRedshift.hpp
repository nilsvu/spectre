// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <complex>
#include <cstddef>
#include <optional>
#include <pup.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/AnalyticData/CircularOrbit.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/AnalyticData/NumericData.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/Tags.hpp"
#include "Elliptic/Tags.hpp"
#include "IO/Observer/GetSectionObservationKey.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/String.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/TypeTraits.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/OptionalHelpers.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace GrSelfForce::Events {

namespace detail {
std::optional<std::complex<double>> extract_redshift(
    const Domain<2>& domain, const ElementId<2>& element_id,
    const AnalyticData::CircularOrbit& circular_orbit,
    const tnsr::aa<ComplexDataVector, 3>& field, const Mesh<2>& mesh);
}  // namespace detail

template <typename ArraySectionIdTag = void>
class ObserveRedshift : public Event {
 public:
  using BackgroundTag =
      elliptic::Tags::Background<elliptic::analytic_data::Background>;

  explicit ObserveRedshift(CkMigrateMessage* msg) : Event(msg) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ObserveRedshift);  // NOLINT

  using options = tmpl::list<>;

  static constexpr Options::String help =
      "Observe the redshift at the position of the particle.";

  ObserveRedshift() = default;

  using compute_tags_for_observation_box = tmpl::list<>;

  using return_tags = tmpl::list<>;
  using argument_tags = tmpl::list<::Tags::ObservationBox>;

  template <typename ComputeTagsList, typename DataBoxType,
            typename Metavariables, typename ParallelComponent>
  void operator()(const ObservationBox<ComputeTagsList, DataBoxType>& box,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ElementId<2>& element_id,
                  const ParallelComponent* const /*meta*/,
                  const ObservationValue& observation_value) const {
    // Skip observation on elements that are not part of the section
    const std::optional<std::string> section_observation_key =
        observers::get_section_observation_key<ArraySectionIdTag>(box);
    if (not section_observation_key.has_value()) {
      return;
    }
    const auto& background = get<BackgroundTag>(box);
    const auto* co_ptr =
        dynamic_cast<const AnalyticData::CircularOrbit*>(&background);
    const auto* nd_ptr =
        co_ptr ? nullptr
               : dynamic_cast<const AnalyticData::NumericData*>(&background);
    ASSERT(co_ptr != nullptr or nd_ptr != nullptr,
           "Background must be CircularOrbit or NumericData");
    const AnalyticData::CircularOrbit& circular_orbit =
        co_ptr ? *co_ptr : nd_ptr->circular_orbit();
    const auto& mesh = get<domain::Tags::Mesh<2>>(box);
    const auto redshift = detail::extract_redshift(
        get<domain::Tags::Domain<2>>(box), element_id, circular_orbit,
        get<Tags::MMode>(box), mesh);
    if (not redshift.has_value()) {
      return;
    }
    // Write result to file
    auto& reduction_writer = Parallel::get_parallel_component<
        observers::ObserverWriter<Metavariables>>(cache);
    Parallel::threaded_action<
        observers::ThreadedActions::WriteReductionDataRow>(
        reduction_writer[0], std::string{"/Redshift"},
        std::vector<std::string>{"IterationId", "NumberOfPoints",
                                 "Re(Redshift)", "Im(Redshift)"},
        std::make_tuple(observation_value.value, mesh.number_of_grid_points(),
                        redshift->real(), redshift->imag()));
  }

  using observation_registration_tags = tmpl::list<::Tags::DataBox>;

  template <typename DbTagsList>
  std::optional<
      std::pair<observers::TypeOfObservation, observers::ObservationKey>>
  get_observation_type_and_key_for_registration(
      const db::DataBox<DbTagsList>& box) const {
    const std::optional<std::string> section_observation_key =
        observers::get_section_observation_key<ArraySectionIdTag>(box);
    if (not section_observation_key.has_value()) {
      return std::nullopt;
    }
    return {
        {observers::TypeOfObservation::Reduction,
         observers::ObservationKey("ObserveRedshift" +
                                   section_observation_key.value() + ".dat")}};
  }

  using is_ready_argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*meta*/) const {
    return true;
  }

  bool needs_evolved_variables() const override { return false; }
};

/// \cond
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
template <typename ArraySectionIdTag>
PUP::able::PUP_ID ObserveRedshift<ArraySectionIdTag>::my_PUP_ID = 0;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)
/// \endcond
}  // namespace GrSelfForce::Events
