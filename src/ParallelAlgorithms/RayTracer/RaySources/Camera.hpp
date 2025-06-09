// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/String.hpp"
#include "ParallelAlgorithms/RayTracer/BackgroundSpacetimes/BackgroundSpacetime.hpp"
#include "ParallelAlgorithms/RayTracer/RaySources/RaySource.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace ray_tracing {

class Camera : public RaySource {
 public:
  static constexpr Options::String help = "A camera that emits rays";

  struct Position {
    using type = std::array<double, 3>;
    static constexpr Options::String help = "Position of the camera";
  };

  struct Focus {
    using type = std::array<double, 3>;
    static constexpr Options::String help = "Focus of the camera";
  };

  struct Up {
    using type = std::array<double, 3>;
    static constexpr Options::String help = "Up direction of the camera";
  };

  struct ParallelRays {
    using type = bool;
    static constexpr Options::String help =
        "If true, rays are emitted parallel to the camera's direction and "
        "the 'OpeningAngle' is the extent of the camera in the up "
        "direction (max impact parameter). "
        "If false, rays are emitted in a cone defined by the opening angle.";
  };

  struct OpeningAngle {
    using type = double;
    static constexpr Options::String help = "Opening angle of the camera";
  };

  struct Resolution {
    using type = std::array<size_t, 2>;
    static constexpr Options::String help =
        "Resolution of the camera in pixels";
  };

  struct TimeRange {
    using type = std::array<double, 2>;
    static constexpr Options::String help =
        "Time at which to evaluate the camera";
  };

  struct Interval {
    using type = double;
    static constexpr Options::String help =
        "Time interval between evaluations of the camera";
  };

  struct IntegrationTime {
    using type = double;
    static constexpr Options::String help =
        "Maximum time to integrate the geodesics";
  };

  struct OnlyUpperHalf {
    using type = bool;
    static constexpr Options::String help =
        "Due to symmetry, only emit rays in the upper half of the camera";
  };

  using options =
      tmpl::list<Position, Focus, Up, ParallelRays, OpeningAngle, Resolution,
                 TimeRange, Interval, IntegrationTime, OnlyUpperHalf>;

  Camera() = default;
  Camera(const Camera& /*rhs*/) = default;
  Camera& operator=(const Camera& /*rhs*/) = default;
  Camera(Camera&& /*rhs*/) = default;
  Camera& operator=(Camera&& /*rhs*/) = default;
  ~Camera() override = default;

  /*!
   * \brief Construct a Camera
   *
   * \param position Position of the camera in Cartesian inertial coordinates.
   * \param focus Point the camera is pointing at.
   * \param up Up direction of the camera.
   * \param parallel_rays If true, rays are emitted parallel to the camera's
   * direction and the `opening_angle` is the extent of the camera in the up
   * direction (max impact parameter). If false, rays are emitted in a cone
   * defined by the opening angle.
   * \param opening_angle Opening angle in radians.
   * \param resolution Number of pixels in the horizontal and vertical
   * directions.
   * \param time_range Time range
   * \param interval Time interval between frames
   * \param integration_time Maximum time to integrate the geodesics.
   * \param only_upper_half If true, only emit rays in the upper half of the
   * camera.
   */
  Camera(std::array<double, 3> position, std::array<double, 3> focus,
         std::array<double, 3> up, bool parallel_rays, double opening_angle,
         std::array<size_t, 2> resolution, std::array<double, 2> time_range,
         double interval, double integration_time,
         bool only_upper_half = false);

  const auto& position() const { return position_; }
  const auto& direction() const { return direction_; }
  const auto& up() const { return up_; }
  const auto& right() const { return right_; }
  double opening_angle() const { return opening_angle_; }
  const auto& resolution() const { return resolution_; }

  /// \cond
  explicit Camera(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Camera);
  /// \endcond

  std::unique_ptr<RaySource> get_clone() const override;

  size_t num_frames() const override;

  size_t num_rays(size_t frame) const override;

  std::array<double, 2> time_bounds(const size_t frame) const override;

  void initialize(size_t frame,
                  const BackgroundSpacetime& background_spacetime) override;

  tuples::tagged_tuple_from_typelist<tags> operator()(
      size_t ray_index,
      const BackgroundSpacetime& background_spacetime) const override;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  tnsr::I<double, 3> position_{};
  tnsr::A<double, 3> four_velocity_{};
  tnsr::A<double, 3> direction_{};
  tnsr::A<double, 3> up_{};
  tnsr::A<double, 3> right_{};
  bool parallel_rays_{};
  double opening_angle_{};
  std::array<size_t, 2> resolution_{};
  std::array<double, 2> time_range_{};
  double interval_{};
  double integration_time_{};
  bool only_upper_half_{false};
  // State
  double time_{};
  tnsr::aa<double, 3> spacetime_metric_{};
};

}  // namespace ray_tracing
