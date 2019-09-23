// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/Xcts/NeutronStarHeadOnCollision.hpp"

#include "DataStructures/DataVector.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/TovStar.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace Xcts {
namespace AnalyticData {

NeutronStarHeadOnCollision::NeutronStarHeadOnCollision(
    const double separation,
    const std::array<double, 2> central_rest_mass_densities,
    const double polytropic_constant, const double polytropic_exponent) noexcept
    : separation_(separation),
      central_rest_mass_densities_(std::move(central_rest_mass_densities)),
      polytropic_constant_(polytropic_constant),
      polytropic_exponent_(polytropic_exponent),
      equation_of_state_{polytropic_constant_, polytropic_exponent_},
      stars_{{{central_rest_mass_densities_[0], polytropic_constant,
               polytropic_exponent},
              {central_rest_mass_densities_[1], polytropic_constant,
               polytropic_exponent}}} {
  // Compute star centers such that the center-of-mass is at the origin
  const std::array<double, 2> masses{
      {stars_[0].radial_solution().total_mass(),
       stars_[1].radial_solution().total_mass()}};
  const double total_mass = masses[0] + masses[1];
  star_centers_ = std::array<double, 2>{{-separation_ * masses[1] / total_mass,
                                         separation_ * masses[0] / total_mass}};
}

void NeutronStarHeadOnCollision::pup(PUP::er& p) noexcept {
  p | separation_;
  p | central_rest_mass_densities_;
  p | polytropic_constant_;
  p | polytropic_exponent_;
  p | equation_of_state_;
  p | stars_;
  p | star_centers_;
}

bool operator==(const NeutronStarHeadOnCollision& lhs,
                const NeutronStarHeadOnCollision& rhs) noexcept {
  // there is no comparison operator for the EoS, but should be okay as
  // the `polytropic_exponent`s and `polytropic_constant`s are compared
  return lhs.central_rest_mass_densities_ ==
             rhs.central_rest_mass_densities_ and
         lhs.separation_ == rhs.separation_ and
         lhs.polytropic_constant_ == rhs.polytropic_constant_ and
         lhs.polytropic_exponent_ == rhs.polytropic_exponent_;
}

bool operator!=(const NeutronStarHeadOnCollision& lhs,
                const NeutronStarHeadOnCollision& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace AnalyticData
}  // namespace Xcts
/// \endcond
