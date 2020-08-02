// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/BoundaryConditions/Elasticity/LaserBeam.hpp"

#include <algorithm>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Elliptic/BoundaryConditions.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/TMPL.hpp"

template <>
Elasticity::BoundaryConditions::MirrorSuspension
Options::create_from_yaml<Elasticity::BoundaryConditions::MirrorSuspension>::
    create<void>(const Options::Option& options) {
  const auto mirror_suspension = options.parse_as<std::string>();
  if (mirror_suspension == "AttachedOnBack") {
    return Elasticity::BoundaryConditions::MirrorSuspension::AttachedOnBack;
  } else if (mirror_suspension == "AttachedOnSides") {
    return Elasticity::BoundaryConditions::MirrorSuspension::AttachedOnSides;
  }
  PARSE_ERROR(options.context(),
              "Invalid mirror suspension type '"
                  << mirror_suspension
                  << "'. Available: 'AttachedOnBack', 'AttachedOnSides'");
}

namespace Elasticity::BoundaryConditions {

elliptic::BoundaryCondition mirror_suspension_boundary_condition_type(
    const MirrorSuspension mirror_suspension,
    const Direction<3>& direction) noexcept {
  if (direction == Direction<3>::lower_zeta()) {
    return elliptic::BoundaryCondition::Neumann;
  }
  switch (mirror_suspension) {
    case MirrorSuspension::AttachedOnBack:
      return direction == Direction<3>::upper_zeta()
                 ? elliptic::BoundaryCondition::Dirichlet
                 : elliptic::BoundaryCondition::Neumann;
    case MirrorSuspension::AttachedOnSides:
      return direction == Direction<3>::upper_zeta()
                 ? elliptic::BoundaryCondition::Neumann
                 : elliptic::BoundaryCondition::Dirichlet;
    default:
      ERROR("Invalid MirrorSuspension");
  }
}

LaserBeam::LaserBeam(double beam_width,
                     MirrorSuspension mirror_suspension) noexcept
    : beam_width_(beam_width), mirror_suspension_(mirror_suspension) {}

void LaserBeam::apply(
    const gsl::not_null<tnsr::I<DataVector, 3>*> displacement,
    // This is n_i F^{ij} = n_i Y^{ijkl}(x) S_{kl} = -n_i T^{ij}
    const gsl::not_null<tnsr::I<DataVector, 3>*> minus_n_dot_stress,
    const tnsr::i<DataVector, 3>& normal, const Direction<3>& direction,
    const tnsr::I<DataVector, 3>& x) const noexcept {
  if (direction == Direction<3>::lower_zeta()) {
    // Laser beam incident on mirror surface
    // Just checking that the face normal is indeed (0, 0, 1)
#ifdef SPECTRE_DEBUG
    for (size_t i = 0; i < normal.begin()->size(); ++i) {
      ASSERT(equal_within_roundoff(get<0>(normal)[i], 0.) and
                 equal_within_roundoff(get<1>(normal)[i], 0.) and
                 equal_within_roundoff(get<2>(normal)[i], 1.),
             "Expected face normal (0,0,1) at mirror surface, but got ("
                 << get<0>(normal)[i] << "," << get<1>(normal)[i] << ","
                 << get<2>(normal)[i] << ").");
    }
#endif  // SPECTRE_DEBUG
    get<0>(*minus_n_dot_stress) = 0.;
    get<1>(*minus_n_dot_stress) = 0.;
    const DataVector r = get(magnitude(x));
    get<2>(*minus_n_dot_stress) =
        -exp(-square(r) / square(beam_width_)) / M_PI / square(beam_width_);
    return;
  }
  switch (mirror_suspension_) {
    case MirrorSuspension::AttachedOnBack:
      if (direction == Direction<3>::upper_zeta()) {
        std::fill(displacement->begin(), displacement->end(), 0.);
      } else {
        std::fill(minus_n_dot_stress->begin(), minus_n_dot_stress->end(), 0.);
      }
      break;
    case MirrorSuspension::AttachedOnSides:
      if (direction == Direction<3>::upper_zeta()) {
        std::fill(minus_n_dot_stress->begin(), minus_n_dot_stress->end(), 0.);
      } else {
        std::fill(displacement->begin(), displacement->end(), 0.);
      }
      break;
    default:
      ERROR("Invalid MirrorSuspension");
  }
}

LaserBeam::Linearization LaserBeam::linearization() const noexcept {
  return Linearization{mirror_suspension_};
}

void LaserBeam::pup(PUP::er& p) noexcept {
  p | beam_width_;
  p | mirror_suspension_;
}

LinearizedLaserBeam::LinearizedLaserBeam(
    MirrorSuspension mirror_suspension) noexcept
    : mirror_suspension_(mirror_suspension) {}

// The linearization of the variable-independent conditions is just zero
void LinearizedLaserBeam::apply(
    const gsl::not_null<tnsr::I<DataVector, 3>*> displacement,
    const gsl::not_null<tnsr::I<DataVector, 3>*> minus_n_dot_stress,
    const tnsr::i<DataVector, 3>& /*normal*/, const Direction<3>& direction,
    const tnsr::I<DataVector, 3>& x) const noexcept {
  if (boundary_condition_type(x, direction, Tags::Displacement<3>{}) ==
      elliptic::BoundaryCondition::Dirichlet) {
    std::fill(displacement->begin(), displacement->end(), 0.);
  } else {
    std::fill(minus_n_dot_stress->begin(), minus_n_dot_stress->end(), 0.);
  }
}

void LinearizedLaserBeam::pup(PUP::er& p) noexcept { p | mirror_suspension_; }

}  // namespace Elasticity::BoundaryConditions
