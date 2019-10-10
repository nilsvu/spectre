// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/Xcts/NeutronStarBinary.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "ErrorHandling/Error.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/TovStar.hpp"
#include "PointwiseFunctions/OrbitalDynamics/ForceBalance.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Xcts {
namespace AnalyticData {

NeutronStarBinary::NeutronStarBinary(
    const double separation, const double eccentricity,
    const std::array<double, 2> central_rest_mass_densities,
    const double polytropic_constant, const double polytropic_exponent) noexcept
    : separation_(separation),
      eccentricity_(eccentricity),
      central_rest_mass_densities_(std::move(central_rest_mass_densities)),
      polytropic_constant_(polytropic_constant),
      polytropic_exponent_(polytropic_exponent),
      equation_of_state_{polytropic_constant_, polytropic_exponent_},
      stars_{{{central_rest_mass_densities_[0], polytropic_constant,
               polytropic_exponent},
              {central_rest_mass_densities_[1], polytropic_constant,
               polytropic_exponent}}} {
  // Compute star centers such that the center-of-mass (at rest) is at the
  // origin
  const std::array<double, 2> masses{
      {stars_[0].radial_solution().total_mass(),
       stars_[1].radial_solution().total_mass()}};
  const double total_mass = masses[0] + masses[1];
  star_centers_ = std::array<double, 2>{{-separation_ * masses[1] / total_mass,
                                         separation_ * masses[0] / total_mass}};
}

double NeutronStarBinary::separation() const noexcept { return separation_; }

double NeutronStarBinary::eccentricity() const noexcept {
  return eccentricity_;
}

const std::array<double, 2>& NeutronStarBinary::central_rest_mass_densities()
    const noexcept {
  return central_rest_mass_densities_;
}

const std::array<double, 2>& NeutronStarBinary::star_centers() const noexcept {
  return star_centers_;
}

const EquationsOfState::PolytropicFluid<true>&
NeutronStarBinary::equation_of_state() const noexcept {
  return equation_of_state_;
}

double NeutronStarBinary::center_of_mass_estimate() const noexcept {
  return 0.;
}

double NeutronStarBinary::angular_velocity_estimate() const noexcept {
  const double total_mass_estimate = stars_[0].radial_solution().total_mass() +
                                     stars_[1].radial_solution().total_mass();
  return sqrt(total_mass_estimate / cube(separation_));
}

std::array<tnsr::I<DataVector, 3, Frame::Inertial>, 2>
NeutronStarBinary::star_centered_coordinates(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x) const noexcept {
  auto x_centered_left = x;
  auto x_centered_right = x;
  get<0>(x_centered_left) -= star_centers()[0];
  get<0>(x_centered_right) -= star_centers()[1];
  return {{std::move(x_centered_left), std::move(x_centered_right)}};
}

Variables<tmpl::list<gr::Tags::EnergyDensity<DataVector>,
                     gr::Tags::StressTrace<DataVector>,
                     gr::Tags::MomentumDensity<3, Frame::Inertial, DataVector>>>
NeutronStarBinary::matter_sources(
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coords,
    const double& center_of_mass, const double& angular_velocity,
    const std::array<double, 2>& injection_energy) const noexcept {
  const auto star_centered_coords = star_centered_coordinates(inertial_coords);
  const std::array<DataVector, 2> coord_separation_from_star{
      {get(magnitude(star_centered_coords[0])),
       get(magnitude(star_centered_coords[1]))}};
  const double eccentricity = eccentricity_;
  const DataVector star_centers{NeutronStarBinary::star_centers()[0],
                                NeutronStarBinary::star_centers()[1]};
  // From the variable data, compute the specific enthalpy
  const auto lapse = get(lapse_times_conformal_factor) / get(conformal_factor);
  const auto shift_squared = get(dot_product(shift, shift));
  const auto conformal_factor_pow_4 = pow<4>(get(conformal_factor));
  const auto tangential_velocity = orbital::tangential_velocity(
      center_of_mass, angular_velocity, star_centers, eccentricity);
  auto specific_enthalpy =
      make_with_value<Scalar<DataVector>>(conformal_factor, 1.);
  DataVector lorentz_factor{lapse.size()};
  auto spatial_velocity =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(conformal_factor,
                                                               0.);
  get<0>(spatial_velocity) = get<0>(shift) / lapse;
  get<1>(spatial_velocity) = get<1>(shift) / lapse;
  get<2>(spatial_velocity) = get<2>(shift) / lapse;
  for (size_t i = 0; i < get_size(get(specific_enthalpy)); i++) {
    const double injection_energy_of_closest_star =
        coord_separation_from_star[0][i] < coord_separation_from_star[1][i]
            ? injection_energy[0]
            : injection_energy[1];
    const double tangential_velocity_of_closest_star =
        coord_separation_from_star[0][i] < coord_separation_from_star[1][i]
            ? tangential_velocity[0]
            : tangential_velocity[1];
    const double center_of_closest_star =
        coord_separation_from_star[0][i] < coord_separation_from_star[1][i]
            ? star_centers[0]
            : star_centers[1];
    lorentz_factor[i] =
        lapse[i] /
        sqrt(square(lapse[i]) -
             conformal_factor_pow_4[i] *
                 (shift_squared[i] +
                  2. * get<1>(shift)[i] * tangential_velocity_of_closest_star +
                  square(tangential_velocity_of_closest_star)));
    get<1>(spatial_velocity)[i] +=
        tangential_velocity_of_closest_star / lapse[i];
    const auto denominator =
        lorentz_factor[i] * lapse[i] -
        lorentz_factor[i] / lapse[i] * conformal_factor_pow_4[i] *
            (shift_squared[i] +
             tangential_velocity_of_closest_star * get<1>(shift)[i] -
             angular_velocity * get<1>(inertial_coords)[i] * get<0>(shift)[i] +
             angular_velocity *
                 (get<0>(inertial_coords)[i] -
                  eccentricity * (center_of_closest_star - center_of_mass) -
                  center_of_mass) *
                 (get<1>(shift)[i] + tangential_velocity_of_closest_star));
    if (denominator < injection_energy_of_closest_star) {
      get(specific_enthalpy)[i] =
          injection_energy_of_closest_star / denominator;
    }
  }
  // Use the specific enthalpy to compute the quantities that feed back into
  // the XCTS equations, i.e. the energy density and the stress trace.
  auto background_fields = make_with_value<Variables<tmpl::list<
      gr::Tags::EnergyDensity<DataVector>, gr::Tags::StressTrace<DataVector>,
      gr::Tags::MomentumDensity<3, Frame::Inertial, DataVector>>>>(
      conformal_factor, 0.);
  const auto& eos = equation_of_state();
  const auto rest_mass_density =
      eos.rest_mass_density_from_enthalpy(specific_enthalpy);
  const auto density_times_enthalpy =
      get(rest_mass_density) * get(specific_enthalpy);
  const auto pressure = eos.pressure_from_density(rest_mass_density);
  const auto lorentz_factor_squared = square(lorentz_factor);
  get(get<gr::Tags::EnergyDensity<DataVector>>(background_fields)) =
      density_times_enthalpy * (1. + square(lorentz_factor - 1.)) -
      get(pressure);
  get(get<gr::Tags::StressTrace<DataVector>>(background_fields)) =
      density_times_enthalpy * (lorentz_factor_squared - 1.) +
      3. * get(pressure);
  const auto momentum_density_prefactor =
      density_times_enthalpy * lorentz_factor_squared * conformal_factor_pow_4;
  auto& momentum_density =
      get<gr::Tags::MomentumDensity<3, Frame::Inertial, DataVector>>(
          background_fields);
  get<0>(momentum_density) =
      momentum_density_prefactor * get<0>(spatial_velocity);
  get<1>(momentum_density) =
      momentum_density_prefactor * get<1>(spatial_velocity);
  get<2>(momentum_density) =
      momentum_density_prefactor * get<2>(spatial_velocity);
  return background_fields;
}

void NeutronStarBinary::pup(PUP::er& p) noexcept {
  p | separation_;
  p | eccentricity_;
  p | central_rest_mass_densities_;
  p | polytropic_constant_;
  p | polytropic_exponent_;
  p | equation_of_state_;
  p | stars_;
  p | star_centers_;
}

bool operator==(const NeutronStarBinary& lhs,
                const NeutronStarBinary& rhs) noexcept {
  // there is no comparison operator for the EoS, but should be okay as
  // the `polytropic_exponent`s and `polytropic_constant`s are compared
  return lhs.central_rest_mass_densities_ ==
             rhs.central_rest_mass_densities_ and
         lhs.separation_ == rhs.separation_ and
         lhs.eccentricity_ == rhs.eccentricity_ and
         lhs.polytropic_constant_ == rhs.polytropic_constant_ and
         lhs.polytropic_exponent_ == rhs.polytropic_exponent_;
}

bool operator!=(const NeutronStarBinary& lhs,
                const NeutronStarBinary& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace AnalyticData
}  // namespace Xcts
/// \endcond
