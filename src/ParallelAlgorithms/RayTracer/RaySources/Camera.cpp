// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/RayTracer/RaySources/Camera.hpp"

#include <array>
#include <memory>
#include <pup.h>
#include <pup_stl.h>

#include "DataStructures/Tensor/EagerMath/CrossProduct.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/GramSchmidtOrthonormalize.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "ParallelAlgorithms/RayTracer/BackgroundSpacetimes/BackgroundSpacetime.hpp"
#include "ParallelAlgorithms/RayTracer/RaySources/RaySource.hpp"
#include "PointwiseFunctions/GeneralRelativity/InverseSpacetimeMetric.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace ray_tracing {

Camera::Camera(std::array<double, 3> position, std::array<double, 3> focus,
               std::array<double, 3> up, bool parallel_rays,
               double opening_angle, std::array<size_t, 2> resolution,
               std::array<double, 2> time_range, double interval,
               double integration_time, const bool only_upper_half)
    : RaySource(),
      position_(position),
      four_velocity_({{1.0, 0.0, 0.0, 0.0}}),
      direction_({{0., focus[0] - position[0], focus[1] - position[1],
                   focus[2] - position[2]}}),
      up_({{0., up[0], up[1], up[2]}}),
      right_({{0., direction_[1] * up_[2] - direction_[2] * up_[1],
               direction_[2] * up_[0] - direction_[0] * up_[2],
               direction_[0] * up_[1] - direction_[1] * up_[0]}}),
      parallel_rays_(parallel_rays),
      opening_angle_(opening_angle),
      resolution_(resolution),
      time_range_(time_range),
      interval_(interval),
      integration_time_(integration_time),
      only_upper_half_(only_upper_half) {}

Camera::Camera(CkMigrateMessage* msg) : RaySource(msg) {}

std::unique_ptr<RaySource> Camera::get_clone() const {
  return std::make_unique<Camera>(*this);
}

void Camera::pup(PUP::er& p) {
  RaySource::pup(p);
  p | position_;
  p | four_velocity_;
  p | direction_;
  p | up_;
  p | right_;
  p | parallel_rays_;
  p | opening_angle_;
  p | resolution_;
  p | time_range_;
  p | interval_;
  p | integration_time_;
  p | only_upper_half_;
  // Also serialize cached quantities that are computed during initialization
  p | time_;
  p | spacetime_metric_;
}

size_t Camera::num_frames() const {
  return static_cast<size_t>(time_range_[1] - time_range_[0]) / interval_;
}

size_t Camera::num_rays(const size_t /*frame*/) const {
  return resolution_[0] * resolution_[1] * (only_upper_half_ ? 0.5 : 1.0);
}

std::array<double, 2> Camera::time_bounds(const size_t frame) const {
  const double time = time_range_[1] - frame * interval_;
  return {{time - integration_time_, time}};
}

void Camera::initialize(const size_t frame,
                        const BackgroundSpacetime& background_spacetime) {
  time_ = time_range_[1] - frame * interval_;
  const auto background_vars = background_spacetime.variables(position_, time_);
  const auto inv_spacetime_metric = gr::inverse_spacetime_metric(
      get<gr::Tags::Lapse<double>>(background_vars),
      get<gr::Tags::Shift<double, 3, ::Frame::Inertial>>(background_vars),
      get<gr::Tags::InverseSpatialMetric<double, 3, ::Frame::Inertial>>(
          background_vars));
  Scalar<double> det_spacetime_metric{};
  determinant_and_inverse(make_not_null(&det_spacetime_metric),
                          make_not_null(&spacetime_metric_),
                          inv_spacetime_metric);
  gram_schmidt_orthonormalize(
      std::array{make_not_null(&four_velocity_), make_not_null(&direction_),
                 make_not_null(&up_)},
      spacetime_metric_);
  cross_product(make_not_null(&right_), four_velocity_, direction_, up_,
                inv_spacetime_metric, det_spacetime_metric);
}

tuples::tagged_tuple_from_typelist<typename Camera::tags> Camera::operator()(
    size_t ray_index,
    const BackgroundSpacetime& /*background_spacetime*/) const {
  const size_t x_index = ray_index % resolution_[0];
  const size_t y_index = ray_index / resolution_[0];
  const double x_frac =
      resolution_[0] > 1 ? static_cast<double>(x_index) / (resolution_[0] - 1)
                         : 0.5;
  const double y_frac =
      resolution_[1] > 1 ? static_cast<double>(y_index) / (resolution_[1] - 1)
                         : 0.5;
  if (parallel_rays_) {
    auto position = tenex::evaluate<ti::I>(
        position_(ti::I) +
        (2.0 * x_frac - 1.0) * opening_angle_ * right_(ti::I) +
        (2.0 * y_frac - 1.0) * opening_angle_ * up_(ti::I));
    auto momentum = raise_or_lower_index(direction_, spacetime_metric_);
    return tuples::tagged_tuple_from_typelist<typename Camera::tags>{
        time_, std::move(position), tenex::evaluate<ti::i>(-momentum(ti::i)),
        -integration_time_};
  } else {
    const double x_angle = (2.0 * x_frac - 1.0) * tan(opening_angle_ / 2.0);
    const double y_angle = (2.0 * y_frac - 1.0) * tan(opening_angle_ / 2.0);
    const double norm = sqrt(1.0 + square(x_angle) + square(y_angle));
    const tnsr::a<double, 3> momentum = raise_or_lower_index(
        tenex::evaluate<ti::A>(four_velocity_(ti::A) -
                               (direction_(ti::A) + x_angle * right_(ti::A) +
                                y_angle * up_(ti::A)) /
                                   norm),
        spacetime_metric_);
    return tuples::tagged_tuple_from_typelist<typename Camera::tags>{
        time_, position_, tenex::evaluate<ti::i>(momentum(ti::i)),
        -integration_time_};
  }
}

PUP::able::PUP_ID Camera::my_PUP_ID = 0;

}  // namespace ray_tracing
