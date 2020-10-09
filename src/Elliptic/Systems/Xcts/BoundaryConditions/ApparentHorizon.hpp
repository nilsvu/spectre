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
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
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

struct LinearizedApparentHorizon;

struct ApparentHorizon {
 public:
  struct EnclosingRadius {
    using type = double;
    static constexpr Options::String help{
        "Any radius between the excision surface and the outer boundary"};
  };

  using options = tmpl::list<EnclosingRadius>;
  static constexpr Options::String help{"AH boundary conditions"};

  using linearization = LinearizedApparentHorizon;

  ApparentHorizon() = default;
  ApparentHorizon(const ApparentHorizon&) noexcept = default;
  ApparentHorizon& operator=(const ApparentHorizon&) noexcept = default;
  ApparentHorizon(ApparentHorizon&&) noexcept = default;
  ApparentHorizon& operator=(ApparentHorizon&&) noexcept = default;
  ~ApparentHorizon() noexcept = default;

  ApparentHorizon(const double enclosing_radius) noexcept
      : enclosing_radius_(enclosing_radius) {}

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

  using argument_tags =
      tmpl::list<domain::Tags::Coordinates<3, Frame::Inertial>,
                 gr::Tags::TraceExtrinsicCurvature<DataVector>>;
  using volume_tags = tmpl::list<>;

  void apply(
      const gsl::not_null<Scalar<DataVector>*> conformal_factor,
      const gsl::not_null<Scalar<DataVector>*> lapse_times_conformal_factor,
      const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
      const gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
      const gsl::not_null<Scalar<DataVector>*>
          n_dot_lapse_times_conformal_factor_gradient,
      const gsl::not_null<tnsr::I<DataVector, 3>*> n_dot_longitudinal_shift,
      const tnsr::i<DataVector, 3>& face_normal,
      const tnsr::I<DataVector, 3>& x,
      const Scalar<DataVector>& extrinsic_curvature_trace) const noexcept {
    const DataVector r = get(magnitude(x));
    if (r[0] > enclosing_radius_) {
      get(*conformal_factor) = 1.;
      get(*lapse_times_conformal_factor) = 1.;
      for (size_t i = 0; i < 3; ++i) {
        shift_excess->get(i) = 0.;
      }
    } else {
      // Conformal factor
      Scalar<DataVector> nn_dot_longitudinal_shift{face_normal.begin()->size()};
      normal_dot_flux(make_not_null(&nn_dot_longitudinal_shift), face_normal,
                      *n_dot_longitudinal_shift);
      get(*n_dot_conformal_factor_gradient) =
          -0.5 * get(*conformal_factor) / r +
          get(extrinsic_curvature_trace) * cube(get(*conformal_factor)) / 6. -
          pow<4>(get(*conformal_factor)) / 8. /
              get(*lapse_times_conformal_factor) *
              get(nn_dot_longitudinal_shift);

      // Lapse
      get(*n_dot_lapse_times_conformal_factor_gradient) = 0.;

      // Shift
      DataVector beta_orthogonal =
          get(*lapse_times_conformal_factor) / cube(get(*conformal_factor));
      for (size_t i = 0; i < 3; ++i) {
        shift_excess->get(i) = beta_orthogonal * face_normal.get(i);
      }
    }
  }

  void pup(PUP::er& p) noexcept { p | enclosing_radius_; }

  double enclosing_radius_;
};

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

  LinearizedApparentHorizon(const double enclosing_radius) noexcept
      : enclosing_radius_(enclosing_radius) {}

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

  using argument_tags =
      tmpl::list<domain::Tags::Coordinates<3, Frame::Inertial>,
                 gr::Tags::TraceExtrinsicCurvature<DataVector>,
                 Xcts::Tags::ConformalFactor<DataVector>,
                 Xcts::Tags::LapseTimesConformalFactor<DataVector>,
                 ::Tags::NormalDotFlux<
                     Xcts::Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>>;
  using volume_tags = tmpl::list<>;

  void apply(
      const gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
      const gsl::not_null<Scalar<DataVector>*>
          lapse_times_conformal_factor_correction,
      const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess_correction,
      const gsl::not_null<Scalar<DataVector>*>
          n_dot_conformal_factor_gradient_correction,
      const gsl::not_null<Scalar<DataVector>*>
          n_dot_lapse_times_conformal_factor_gradient_correction,
      const gsl::not_null<tnsr::I<DataVector, 3>*>
          n_dot_longitudinal_shift_correction,
      const tnsr::i<DataVector, 3, Frame::Inertial>& face_normal,
      const tnsr::I<DataVector, 3, Frame::Inertial>& x,
      const Scalar<DataVector>& extrinsic_curvature_trace,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& lapse_times_conformal_factor,
      const tnsr::I<DataVector, 3>& minus_n_dot_longitudinal_shift)
      const noexcept {
    const DataVector r = get(magnitude(x));
    if (r[0] > enclosing_radius_) {
      get(*conformal_factor_correction) = 0.;
      get(*lapse_times_conformal_factor_correction) = 0.;
      for (size_t i = 0; i < 3; ++i) {
        shift_excess_correction->get(i) = 0.;
      }
    } else {
      // Conformal factor
      Scalar<DataVector> minus_nn_dot_longitudinal_shift{
          face_normal.begin()->size()};
      normal_dot_flux(make_not_null(&minus_nn_dot_longitudinal_shift),
                      face_normal, minus_n_dot_longitudinal_shift);
      Scalar<DataVector> nn_dot_longitudinal_shift_correction{
          face_normal.begin()->size()};
      normal_dot_flux(make_not_null(&nn_dot_longitudinal_shift_correction),
                      face_normal, *n_dot_longitudinal_shift_correction);
      get(*n_dot_conformal_factor_gradient_correction) =
          -0.5 * get(*conformal_factor_correction) / r +
          0.5 * get(extrinsic_curvature_trace) * square(get(conformal_factor)) *
              get(*conformal_factor_correction) +
          0.5 * pow<3>(get(conformal_factor)) /
              get(lapse_times_conformal_factor) *
              get(minus_nn_dot_longitudinal_shift) *
              get(*conformal_factor_correction) -
          0.125 * pow<4>(get(conformal_factor)) /
              square(get(lapse_times_conformal_factor)) *
              get(minus_nn_dot_longitudinal_shift) *
              get(*lapse_times_conformal_factor_correction) -
          0.125 * pow<4>(get(conformal_factor)) /
              get(lapse_times_conformal_factor) *
              get(nn_dot_longitudinal_shift_correction);

      // Lapse
      get(*n_dot_lapse_times_conformal_factor_gradient_correction) = 0.;

      // Shift
      DataVector beta_orthogonal =
          get(*lapse_times_conformal_factor_correction) /
              cube(get(conformal_factor)) -
          3. * get(lapse_times_conformal_factor) /
              pow<4>(get(conformal_factor)) * get(*conformal_factor_correction);
      for (size_t i = 0; i < 3; ++i) {
        shift_excess_correction->get(i) = beta_orthogonal * face_normal.get(i);
      }
    }
  }

  void pup(PUP::er& p) noexcept { p | enclosing_radius_; }

  double enclosing_radius_;
};

}  // namespace Xcts::BoundaryConditions
