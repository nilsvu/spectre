// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
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
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/Trace.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/Tags.hpp"
#include "IO/Observer/GetSectionObservationKey.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
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

template <typename BackgroundTag, typename ArraySectionIdTag = void>
class ObserveRedshift : public Event {
 public:
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
    const auto& circular_orbit =
        dynamic_cast<const AnalyticData::CircularOrbit&>(background);
    // Get element-logical coords of puncture
    const auto& domain = get<domain::Tags::Domain<2>>(box);
    const auto puncture_position = circular_orbit.puncture_position();
    const auto& block = domain.blocks()[element_id.block_id()];
    const auto block_logical_coords =
        block_logical_coordinates_single_point(puncture_position, block);
    if (not block_logical_coords.has_value()) {
      return;
    }
    const auto puncture_logical_coords =
        element_logical_coordinates(block_logical_coords.value(), element_id);
    if (not puncture_logical_coords.has_value()) {
      return;
    }
    // Interpolate field to puncture position
    const auto& field = get<Tags::MMode>(box);
    const auto& mesh = get<domain::Tags::Mesh<2>>(box);
    const intrp::Irregular<2> interpolator(mesh,
                                           puncture_logical_coords.value());
    tnsr::aa<std::complex<double>, 3> field_at_puncture{};
    ComplexDataVector intrp_result(1_st);
    for (size_t i = 0; i < field.size(); ++i) {
      interpolator.interpolate(make_not_null(&intrp_result), field[i]);
      field_at_puncture[i] = intrp_result[0];
    }
    // Calculate redshift
    const double r0 = circular_orbit.orbital_radius();
    const double M = circular_orbit.black_hole_mass();
    const double spin = circular_orbit.black_hole_spin();
    const double a = M * spin;
    const double omega = 1. / (a + sqrt(cube(r0) / M));
    const double delta = square(r0) - 2.0 * M * r0 + a * a;
    const double sigma = square(r0);
    tnsr::aa<double, 3> kerr_metric{0.0};
    get<0, 0>(kerr_metric) = -(1.0 - 2.0 * M * r0 / sigma);
    get<0, 3>(kerr_metric) = -4.0 * M * a * r0 / sigma;
    get<1, 1>(kerr_metric) = sigma / delta;
    get<2, 2>(kerr_metric) = sigma;
    get<3, 3>(kerr_metric) =
        (square(r0) + square(a) + 2.0 * M * square(a) * r0 / sigma);
    const auto inv_kerr_metric = determinant_and_inverse(kerr_metric).second;
    auto hbar = field_at_puncture;
    get<0, 0>(hbar) *= 1.0 / r0;
    get<0, 1>(hbar) *= r0 / delta;
    get<1, 1>(hbar) *= cube(r0) / square(delta);
    get<1, 2>(hbar) *= square(r0) / delta;
    get<1, 3>(hbar) *= square(r0) / delta;
    get<2, 2>(hbar) *= r0;
    get<2, 3>(hbar) *= r0;
    get<3, 3>(hbar) *= r0;
    const auto trace_hbar =
        tenex::evaluate(hbar(ti::a, ti::b) * inv_kerr_metric(ti::A, ti::B));
    const auto h = tenex::evaluate<ti::a, ti::b>(
        hbar(ti::a, ti::b) - 0.5 * trace_hbar() * kerr_metric(ti::a, ti::b));
    tnsr::A<double, 3> u_particle{0.0};
    const double u_mag =
        sqrt(cube(r0) / M - 3.0 * square(r0) + 2.0 * a * sqrt(cube(r0) / M));
    get<0>(u_particle) = 1.0 / u_mag / omega;
    get<3>(u_particle) = 1.0 / u_mag;
    const auto redshift = tenex::evaluate(
        0.5 * h(ti::a, ti::b) * u_particle(ti::A) * u_particle(ti::B));
    // Write result to file
    auto& reduction_writer = Parallel::get_parallel_component<
        observers::ObserverWriter<Metavariables>>(cache);
    Parallel::threaded_action<
        observers::ThreadedActions::WriteReductionDataRow>(
        reduction_writer[0], std::string{"/Redshift"},
        std::vector<std::string>{"IterationId", "NumberOfPoints",
                                 "Re(Redshift)", "Im(Redshift)"},
        std::make_tuple(observation_value.value, mesh.number_of_grid_points(),
                        get(redshift).real(), get(redshift).imag()));
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

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
template <typename BackgroundTag, typename ArraySectionIdTag>
PUP::able::PUP_ID ObserveRedshift<BackgroundTag, ArraySectionIdTag>::my_PUP_ID =
    0;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)
/// \endcond
}  // namespace GrSelfForce::Events
