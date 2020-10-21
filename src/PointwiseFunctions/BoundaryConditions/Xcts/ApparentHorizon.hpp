// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <ostream>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Elliptic/BoundaryConditions.hpp"
#include "Elliptic/Protocols.hpp"
#include "Elliptic/Systems/Xcts/Equations.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Xcts::BoundaryConditions {

template <Xcts::Geometry ConformalGeometry>
void apparent_horizon(
    gsl::not_null<Scalar<DataVector>*> conformal_factor,
    gsl::not_null<Scalar<DataVector>*> lapse_times_conformal_factor,
    gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
    gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
    gsl::not_null<Scalar<DataVector>*>
        n_dot_lapse_times_conformal_factor_gradient,
    gsl::not_null<tnsr::I<DataVector, 3>*> n_dot_longitudinal_shift_excess,
    const tnsr::i<DataVector, 3>& inward_pointing_face_normal,
    const tnsr::I<DataVector, 3>& x,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const tnsr::I<DataVector, 3>& shift_background,
    const tnsr::II<DataVector, 3>& longitudinal_shift_background,
    const std::optional<tnsr::II<DataVector, 3>>& inv_conformal_metric,
    const std::optional<tnsr::Ijj<DataVector, 3>>&
        conformal_christoffel_second_kind) noexcept;

template <Xcts::Geometry ConformalGeometry>
void linearized_apparent_horizon(
    gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
    gsl::not_null<Scalar<DataVector>*> lapse_times_conformal_factor_correction,
    gsl::not_null<tnsr::I<DataVector, 3>*> shift_correction,
    gsl::not_null<Scalar<DataVector>*>
        n_dot_conformal_factor_gradient_correction,
    gsl::not_null<Scalar<DataVector>*>
        n_dot_lapse_times_conformal_factor_gradient_correction,
    gsl::not_null<tnsr::I<DataVector, 3>*> n_dot_longitudinal_shift_correction,
    const tnsr::i<DataVector, 3>& inward_pointing_face_normal,
    const tnsr::I<DataVector, 3>& x,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const tnsr::II<DataVector, 3>& longitudinal_shift_background,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, 3>& n_dot_longitudinal_shift_excess,
    const std::optional<tnsr::II<DataVector, 3>>& inv_conformal_metric,
    const std::optional<tnsr::Ijj<DataVector, 3>>&
        conformal_christoffel_second_kind) noexcept;

template <Xcts::Geometry ConformalGeometry>
struct LinearizedApparentHorizon;

template <Xcts::Geometry ConformalGeometry>
struct ApparentHorizon {
 public:
  struct EnclosingRadius {
    using type = double;
    static constexpr Options::String help{
        "Any radius between the excision surface and the outer boundary"};
  };

  using options = tmpl::list<EnclosingRadius>;
  static constexpr Options::String help{"AH boundary conditions"};

  using Linearization = LinearizedApparentHorizon<ConformalGeometry>;

  ApparentHorizon() = default;
  ApparentHorizon(const ApparentHorizon&) noexcept = default;
  ApparentHorizon& operator=(const ApparentHorizon&) noexcept = default;
  ApparentHorizon(ApparentHorizon&&) noexcept = default;
  ApparentHorizon& operator=(ApparentHorizon&&) noexcept = default;
  ~ApparentHorizon() noexcept = default;

  explicit ApparentHorizon(double enclosing_radius) noexcept;

  Linearization linearization() const noexcept;

  template <typename Tag>
  elliptic::BoundaryCondition boundary_condition_type(
      const tnsr::I<DataVector, 3>& x, const Direction<3>& /*direction*/,
      Tag /*meta*/) const noexcept {
    static_assert(
        tmpl::list_contains_v<
            tmpl::list<Xcts::Tags::ConformalFactor<DataVector>,
                       Xcts::Tags::LapseTimesConformalFactor<DataVector>,
                       Xcts::Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>,
            Tag>,
        "Unsupported tag");
    const DataVector r = get(magnitude(x));
    if (r[0] > enclosing_radius_) {
      return elliptic::BoundaryCondition::Dirichlet;
    }
    if constexpr (std::is_same_v<Tag,
                                 Xcts::Tags::ConformalFactor<DataVector>> or
                  std::is_same_v<
                      Tag, Xcts::Tags::LapseTimesConformalFactor<DataVector>>) {
      return elliptic::BoundaryCondition::Neumann;
    } else {
      return elliptic::BoundaryCondition::Dirichlet;
    }
  }

  using argument_tags = tmpl::flatten<tmpl::list<
      domain::Tags::Coordinates<3, Frame::Inertial>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>,
      Xcts::Tags::ShiftBackground<DataVector, 3, Frame::Inertial>,
      Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
          DataVector, 3, Frame::Inertial>,
      tmpl::conditional_t<ConformalGeometry == Xcts::Geometry::NonEuclidean,
                          tmpl::list<Xcts::Tags::InverseConformalMetric<
                                         DataVector, 3, Frame::Inertial>,
                                     Xcts::Tags::ConformalChristoffelSecondKind<
                                         DataVector, 3, Frame::Inertial>>,
                          tmpl::list<>>>>;
  using volume_tags = tmpl::list<>;

  template <typename... NonEuclideanArgs>
  void apply(
      const gsl::not_null<Scalar<DataVector>*> conformal_factor,
      const gsl::not_null<Scalar<DataVector>*> lapse_times_conformal_factor,
      const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
      const gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
      const gsl::not_null<Scalar<DataVector>*>
          n_dot_lapse_times_conformal_factor_gradient,
      const gsl::not_null<tnsr::I<DataVector, 3>*>
          n_dot_longitudinal_shift_excess,
      const tnsr::i<DataVector, 3>& inward_pointing_face_normal,
      const tnsr::I<DataVector, 3>& x,
      const Scalar<DataVector>& extrinsic_curvature_trace,
      const tnsr::I<DataVector, 3>& shift_background,
      const tnsr::II<DataVector, 3>& longitudinal_shift_background,
      const NonEuclideanArgs&... non_euclidean_args) const noexcept {
    const DataVector r = get(magnitude(x));
    if (r[0] > enclosing_radius_) {
      get(*conformal_factor) = 1.;
      get(*lapse_times_conformal_factor) = 1.;
      for (size_t i = 0; i < 3; ++i) {
        shift_excess->get(i) = 0.;
      }
    } else {
      if constexpr (ConformalGeometry == Xcts::Geometry::Euclidean) {
        apparent_horizon<ConformalGeometry>(
            conformal_factor, lapse_times_conformal_factor, shift_excess,
            n_dot_conformal_factor_gradient,
            n_dot_lapse_times_conformal_factor_gradient,
            n_dot_longitudinal_shift_excess, inward_pointing_face_normal, x,
            extrinsic_curvature_trace, shift_background,
            longitudinal_shift_background, std::nullopt, std::nullopt);
      } else {
        apparent_horizon<ConformalGeometry>(
            conformal_factor, lapse_times_conformal_factor, shift_excess,
            n_dot_conformal_factor_gradient,
            n_dot_lapse_times_conformal_factor_gradient,
            n_dot_longitudinal_shift_excess, inward_pointing_face_normal, x,
            extrinsic_curvature_trace, shift_background,
            longitudinal_shift_background, non_euclidean_args...);
      }
    }
  }

  void pup(PUP::er& p) noexcept;

 private:
  double enclosing_radius_;
};

template <Xcts::Geometry ConformalGeometry>
struct LinearizedApparentHorizon {
 public:
  struct EnclosingRadius {
    using type = double;
    static constexpr Options::String help{
        "Any radius between the excision surface and the outer boundary"};
  };

  using options = tmpl::list<EnclosingRadius>;
  static constexpr Options::String help{"AH boundary conditions"};

  LinearizedApparentHorizon() = default;
  LinearizedApparentHorizon(const LinearizedApparentHorizon&) noexcept =
      default;
  LinearizedApparentHorizon& operator=(
      const LinearizedApparentHorizon&) noexcept = default;
  LinearizedApparentHorizon(LinearizedApparentHorizon&&) noexcept = default;
  LinearizedApparentHorizon& operator=(LinearizedApparentHorizon&&) noexcept =
      default;
  ~LinearizedApparentHorizon() noexcept = default;

  explicit LinearizedApparentHorizon(double enclosing_radius) noexcept;

  template <typename Tag>
  elliptic::BoundaryCondition boundary_condition_type(
      const tnsr::I<DataVector, 3>& x, const Direction<3>& /*direction*/,
      Tag /*meta*/) const noexcept {
    const DataVector r = get(magnitude(x));
    if (r[0] > enclosing_radius_) {
      return elliptic::BoundaryCondition::Dirichlet;
    }
    if constexpr (std::is_same_v<Tag,
                                 Xcts::Tags::ConformalFactor<DataVector>> or
                  std::is_same_v<
                      Tag, Xcts::Tags::LapseTimesConformalFactor<DataVector>>) {
      return elliptic::BoundaryCondition::Neumann;
    } else {
      return elliptic::BoundaryCondition::Dirichlet;
    }
  }

  using argument_tags = tmpl::flatten<tmpl::list<
      domain::Tags::Coordinates<3, Frame::Inertial>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>,
      Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
          DataVector, 3, Frame::Inertial>,
      Xcts::Tags::ConformalFactor<DataVector>,
      Xcts::Tags::LapseTimesConformalFactor<DataVector>,
      ::Tags::NormalDotFlux<
          Xcts::Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>,
      tmpl::conditional_t<ConformalGeometry == Xcts::Geometry::NonEuclidean,
                          tmpl::list<Xcts::Tags::InverseConformalMetric<
                                         DataVector, 3, Frame::Inertial>,
                                     Xcts::Tags::ConformalChristoffelSecondKind<
                                         DataVector, 3, Frame::Inertial>>,
                          tmpl::list<>>>>;
  using volume_tags = tmpl::list<>;

  template <typename... NonEuclideanArgs>
  void apply(
      const gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
      const gsl::not_null<Scalar<DataVector>*>
          lapse_times_conformal_factor_correction,
      const gsl::not_null<tnsr::I<DataVector, 3>*> shift_correction,
      const gsl::not_null<Scalar<DataVector>*>
          n_dot_conformal_factor_gradient_correction,
      const gsl::not_null<Scalar<DataVector>*>
          n_dot_lapse_times_conformal_factor_gradient_correction,
      const gsl::not_null<tnsr::I<DataVector, 3>*>
          n_dot_longitudinal_shift_correction,
      const tnsr::i<DataVector, 3>& inward_pointing_face_normal,
      const tnsr::I<DataVector, 3>& x,
      const Scalar<DataVector>& extrinsic_curvature_trace,
      const tnsr::II<DataVector, 3>& longitudinal_shift_background,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& lapse_times_conformal_factor,
      const tnsr::I<DataVector, 3>& n_dot_longitudinal_shift_excess,
      const NonEuclideanArgs&... non_euclidean_args) const noexcept {
    const DataVector r = get(magnitude(x));
    if (r[0] > enclosing_radius_) {
      get(*conformal_factor_correction) = 0.;
      get(*lapse_times_conformal_factor_correction) = 0.;
      for (size_t i = 0; i < 3; ++i) {
        shift_correction->get(i) = 0.;
      }
    } else {
      if constexpr (ConformalGeometry == Xcts::Geometry::Euclidean) {
        linearized_apparent_horizon<ConformalGeometry>(
            conformal_factor_correction,
            lapse_times_conformal_factor_correction, shift_correction,
            n_dot_conformal_factor_gradient_correction,
            n_dot_lapse_times_conformal_factor_gradient_correction,
            n_dot_longitudinal_shift_correction, inward_pointing_face_normal, x,
            extrinsic_curvature_trace, longitudinal_shift_background,
            conformal_factor, lapse_times_conformal_factor,
            n_dot_longitudinal_shift_excess, std::nullopt, std::nullopt);
      } else {
        linearized_apparent_horizon<ConformalGeometry>(
            conformal_factor_correction,
            lapse_times_conformal_factor_correction, shift_correction,
            n_dot_conformal_factor_gradient_correction,
            n_dot_lapse_times_conformal_factor_gradient_correction,
            n_dot_longitudinal_shift_correction, inward_pointing_face_normal, x,
            extrinsic_curvature_trace, longitudinal_shift_background,
            conformal_factor, lapse_times_conformal_factor,
            n_dot_longitudinal_shift_excess, non_euclidean_args...);
      }
    }
  }

  void pup(PUP::er& p) noexcept;

 private:
  double enclosing_radius_;
};

}  // namespace Xcts::BoundaryConditions
