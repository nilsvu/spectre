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
#include "Elliptic/Systems/SelfForce/Scalar/Tags.hpp"
#include "IO/Observer/GetSectionObservationKey.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
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

namespace ScalarSelfForce::Events {

/*!
 * \brief An event that observes the self-force at the position of the scalar
 * charge.
 *
 * \details This event interpolates the scalar m-mode field and its derivatives
 * to the position of the scalar charge (puncture) and computes the m-mode
 * contribution to the self-force acting on the charge. The self-force is then
 * written to a reduction file.
 *
 * The self-force is computed in the Boyer-Lindquist `r` and `theta`
 * coordinates (for scalar charge $q=1$) following Eq. (3.10) in
 * \cite Osburn:2022bby :
 * \begin{align}
 *   F_r^m &= \partial_r (\Psi_m^R / r) = \frac{1}{r \alpha}
 *     \partial_{r_*} \Psi_m^R - \frac{1}{r^2} \Psi_m^R \\
 *   F_\theta^m &= \partial_\theta (\Psi_m^R / r) = -\frac{1}{r}
 *     \partial_{\cos\theta} \Psi_m^R
 * \end{align}
 * with $\alpha = 1 - 2 M r / (r^2 + a^2)$. For modes $m > 0$ the self-force
 * includes an additional factor of 2 and a complex rotation by
 * $m \Delta\phi = \frac{m a}{r_+ - r_-}\ln\frac{r - r_+}{r - r_-}$.
 */
template <typename BackgroundTag, typename ArraySectionIdTag = void>
class ObserveSelfForce : public Event {
 public:
  explicit ObserveSelfForce(CkMigrateMessage* msg) : Event(msg) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ObserveSelfForce);  // NOLINT

  using options = tmpl::list<>;

  static constexpr Options::String help =
      "Observe the self-force at the position of the scalar charge.";

  ObserveSelfForce() = default;

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
    // Interpolate field and field derivative to puncture position
    const auto& field = get<Tags::MMode>(box);
    const auto& mesh = get<domain::Tags::Mesh<2>>(box);
    const auto& inv_jacobian =
        get<domain::Tags::InverseJacobian<2, Frame::ElementLogical,
                                          Frame::Inertial>>(box);
    const auto deriv_field = partial_derivative(field, mesh, inv_jacobian);
    const intrp::Irregular<2> interpolator(mesh,
                                           puncture_logical_coords.value());
    ComplexDataVector intrp_result{1_st};
    Scalar<std::complex<double>> field_at_puncture{};
    interpolator.interpolate(make_not_null(&intrp_result), get(field));
    get(field_at_puncture) = intrp_result[0];
    tnsr::i<std::complex<double>, 2> deriv_field_at_puncture{};
    interpolator.interpolate(make_not_null(&intrp_result), get<0>(deriv_field));
    get<0>(deriv_field_at_puncture) = intrp_result[0];
    // Assuming equatorial symmetry, so theta derivative is zero
    get<1>(deriv_field_at_puncture) = 0.;
    // Calculate self-force in r and theta coordinates
    tnsr::i<std::complex<double>, 2> self_force = deriv_field_at_puncture;
    const double r0 = circular_orbit.orbital_radius();
    const double M = circular_orbit.black_hole_mass();
    const double spin = circular_orbit.black_hole_spin();
    const double a = M * spin;
    const int m_mode = circular_orbit.m_mode_number();
    const double r_plus = M * (1. + sqrt(1. - square(spin)));
    const double r_minus = M * (1. - sqrt(1. - square(spin)));
    get<0>(self_force) /= r0;
    if (not circular_orbit.penetrating_horizon()) {
      const double alpha = 1. - 2. * M * r0 / (square(r0) + square(a));
      get<0>(self_force) /= alpha;
    }
    get<0>(self_force) -= get(field_at_puncture) / square(r0);
    if (m_mode > 0) {
      const double delta_phi =
          m_mode * a / (r_plus - r_minus) * log((r0 - r_plus) / (r0 - r_minus));
      const double delta_phi_dr = m_mode * a / (r0 - r_minus) / (r0 - r_plus);
      if (not equal_within_roundoff(puncture_position[1], 0.0)) {
        ERROR("Assuming puncture is at cos(theta) = 0, but puncture is at "
              << puncture_position[1]);
      }
      get<0>(self_force) +=
          std::complex<double>{0., delta_phi_dr / r0} * get(field_at_puncture);
      const std::complex<double> rotation =
          cos(delta_phi) + std::complex<double>(0., 1.) * sin(delta_phi);
      get<0>(self_force) *= 2. * rotation;
    }
    // Write result to file
    auto& reduction_writer = Parallel::get_parallel_component<
        observers::ObserverWriter<Metavariables>>(cache);
    Parallel::threaded_action<
        observers::ThreadedActions::WriteReductionDataRow>(
        reduction_writer[0], std::string{"/SelfForce"},
        std::vector<std::string>{"IterationId", "Re(RegularFieldAtPuncture)",
                                 "Im(RegularFieldAtPuncture)",
                                 "Re(DerivRegularFieldAtPuncture_rstar)",
                                 "Im(DerivRegularFieldAtPuncture_rstar)",
                                 "Re(DerivRegularFieldAtPuncture_costheta)",
                                 "Im(DerivRegularFieldAtPuncture_costheta)",
                                 "Re(SelfForce_r)", "Im(SelfForce_r)",
                                 "Re(SelfForce_theta)", "Im(SelfForce_theta)"},
        std::make_tuple(observation_value.value, get(field_at_puncture).real(),
                        get(field_at_puncture).imag(),
                        get<0>(deriv_field_at_puncture).real(),
                        get<0>(deriv_field_at_puncture).imag(),
                        get<1>(deriv_field_at_puncture).real(),
                        get<1>(deriv_field_at_puncture).imag(),
                        get<0>(self_force).real(), get<0>(self_force).imag(),
                        get<1>(self_force).real(), get<1>(self_force).imag()));
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
         observers::ObservationKey("ObserveSelfForce" +
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
PUP::able::PUP_ID
    ObserveSelfForce<BackgroundTag, ArraySectionIdTag>::my_PUP_ID = 0;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)
/// \endcond
}  // namespace ScalarSelfForce::Events
