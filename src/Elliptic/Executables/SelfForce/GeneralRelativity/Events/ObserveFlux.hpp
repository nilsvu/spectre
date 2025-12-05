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
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/Faces.hpp"
#include "Domain/Tags/SurfaceJacobian.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/Tags.hpp"
#include "IO/Observer/GetSectionObservationKey.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/ProjectToBoundary.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
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
class ObserveFlux : public Event {
 public:
  using ReductionData = Parallel::ReductionData<
      // Observation value
      Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
      // Number of grid points
      Parallel::ReductionDatum<size_t, funcl::Plus<>>,
      // Energy flux
      Parallel::ReductionDatum<double, funcl::Plus<>>,
      // Surface area (should be 2)
      Parallel::ReductionDatum<double, funcl::Plus<>>>;

  explicit ObserveFlux(CkMigrateMessage* msg) : Event(msg) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ObserveFlux);  // NOLINT

  using options = tmpl::list<>;

  static constexpr Options::String help =
      "Observe the energy flux at the outer boundary.";

  ObserveFlux() = default;

  using observed_reduction_data_tags =
      observers::make_reduction_data_tags<tmpl::list<ReductionData>>;
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
    // Use only elements at the outer boundary
    const auto& direction = Direction<2>::upper_xi();
    const auto& element = get<domain::Tags::Element<2>>(box);
    const auto& mesh = get<domain::Tags::Mesh<2>>(box);
    double energy_flux = 0.0;
    double surface_area = 0.0;
    if (element.external_boundaries().contains(direction)) {
      // Get parameters
      const auto& background = get<BackgroundTag>(box);
      const auto& circular_orbit =
          dynamic_cast<const AnalyticData::CircularOrbit&>(background);
      const double r0 = circular_orbit.orbital_radius();
      const double M = circular_orbit.black_hole_mass();
      const double spin = circular_orbit.black_hole_spin();
      const double a = M * spin;
      const int m_mode = circular_orbit.m_mode_number();
      const double omega = 1. / (a + sqrt(cube(r0) / M));
      // Compute integral over the outer boundary
      const auto& field = get<Tags::MMode>(box);
      const auto field_on_face =
          dg::project_tensor_to_boundary(field, mesh, direction);
      const auto& face_jacobian = get<domain::Tags::Faces<
          2, domain::Tags::DetSurfaceJacobian<Frame::ElementLogical,
                                              Frame::Inertial>>>(box)
                                      .at(direction);
      const auto& x = get<domain::Tags::Faces<
          2, domain::Tags::Coordinates<2, Frame::Inertial>>>(box)
                          .at(direction);
      const auto& sin_theta = sin(get<1>(x));
      energy_flux =
          square(m_mode * omega) * 0.03125 *
          definite_integral(
              real(square(abs(get<2, 2>(field_on_face))) +
                   4. * square(abs(get<2, 3>(field_on_face))) +
                   square(abs(get<3, 3>(field_on_face))) -
                   get<2, 2>(field_on_face) * conj(get<3, 3>(field_on_face)) -
                   conj(get<2, 2>(field_on_face)) * get<3, 3>(field_on_face)) *
                  get(face_jacobian) * sin_theta,
              mesh.slice_away(0));
      surface_area =
          definite_integral(sin_theta * get(face_jacobian), mesh.slice_away(0));
    } else {
      energy_flux = 0.0;
      surface_area = 0.0;
    }
    // Send data to reduction observer
    auto& local_observer = *Parallel::local_branch(
        Parallel::get_parallel_component<
            tmpl::conditional_t<Parallel::is_nodegroup_v<ParallelComponent>,
                                observers::ObserverWriter<Metavariables>,
                                observers::Observer<Metavariables>>>(cache));
    const std::string subfile_name = "Flux";
    observers::ObservationId observation_id{observation_value.value,
                                            subfile_name + ".dat"};
    Parallel::ArrayComponentId array_component_id{
        std::add_pointer_t<ParallelComponent>{nullptr},
        Parallel::ArrayIndex<ElementId<2>>(element_id)};
    ReductionData reduction_data{observation_value.value,
                                 mesh.number_of_grid_points(), energy_flux,
                                 surface_area};
    std::vector<std::string> legend{"ObservationValue", "NumberOfPoints",
                                    "EnergyFlux", "SurfaceArea"};
    if constexpr (Parallel::is_nodegroup_v<ParallelComponent>) {
      Parallel::threaded_action<
          observers::ThreadedActions::CollectReductionDataOnNode>(
          local_observer, std::move(observation_id),
          std::move(array_component_id), subfile_name + ".dat",
          std::move(legend), std::move(reduction_data));
    } else {
      Parallel::simple_action<observers::Actions::ContributeReductionData>(
          local_observer, std::move(observation_id),
          std::move(array_component_id), subfile_name + ".dat",
          std::move(legend), std::move(reduction_data));
    }
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
    return {{observers::TypeOfObservation::Reduction,
             observers::ObservationKey(
                 "Flux" + section_observation_key.value() + ".dat")}};
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
PUP::able::PUP_ID ObserveFlux<BackgroundTag, ArraySectionIdTag>::my_PUP_ID = 0;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)
/// \endcond
}  // namespace GrSelfForce::Events
