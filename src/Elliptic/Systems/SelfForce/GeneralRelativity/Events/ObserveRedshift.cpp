// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/SelfForce/GeneralRelativity/Events/ObserveRedshift.hpp"

#include <complex>
#include <cstddef>
#include <optional>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/Trace.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/ElementLogicalCoordinates.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/AnalyticData/CircularOrbit.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Math.hpp"

namespace GrSelfForce::Events::detail {

std::optional<std::complex<double>> extract_redshift(
    const Domain<2>& domain, const ElementId<2>& element_id,
    const AnalyticData::CircularOrbit& circular_orbit,
    const tnsr::aa<ComplexDataVector, 3>& field, const Mesh<2>& mesh) {
  // Get element-logical coords of puncture
  const auto puncture_position = circular_orbit.puncture_position();
  const auto& block = domain.blocks()[element_id.block_id()];
  const auto block_logical_coords =
      block_logical_coordinates_single_point(puncture_position, block);
  if (not block_logical_coords.has_value()) {
    return std::nullopt;
  }
  const auto puncture_logical_coords =
      element_logical_coordinates(block_logical_coords.value(), element_id);
  if (not puncture_logical_coords.has_value()) {
    return std::nullopt;
  }
  // Interpolate field to puncture position
  const intrp::Irregular<2> interpolator(mesh, puncture_logical_coords.value());
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
  return get(redshift);
}

}  // namespace GrSelfForce::Events::detail
