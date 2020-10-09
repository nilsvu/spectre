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
#include "ErrorHandling/Assert.hpp"
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

struct LinearizedApparentHorizons;

struct ApparentHorizons {
 public:
  struct Positions {
    using type = std::array<double, 2>;
    static constexpr Options::String help{"pos"};
  };

  struct EnclosingRadius {
    using type = double;
    static constexpr Options::String help{"enclosing radius"};
  };

  struct NegativeExpansion {
    using type = double;
    static constexpr Options::String help = "Negative exp";
  };

  using options = tmpl::list<Positions, EnclosingRadius, NegativeExpansion>;
  static constexpr Options::String help{
      "Black hole binary initial data in general relativity"};

  using linearization = LinearizedApparentHorizons;

  ApparentHorizons() = default;
  ApparentHorizons(const ApparentHorizons&) noexcept = default;
  ApparentHorizons& operator=(const ApparentHorizons&) noexcept = default;
  ApparentHorizons(ApparentHorizons&&) noexcept = default;
  ApparentHorizons& operator=(ApparentHorizons&&) noexcept = default;
  ~ApparentHorizons() noexcept = default;

  ApparentHorizons(std::array<double, 2> positions, double enclosing_radius,
                   double negative_expansion) noexcept
      : positions_(std::move(positions)),
        enclosing_radius_(enclosing_radius),
        negative_expansion_(negative_expansion) {
    ASSERT(positions_[0] < 0. and positions_[1] > 0., "Invalid pos");
  }

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
                 Xcts::Tags::ShiftBackground<DataVector, 3, Frame::Inertial>>;
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
      const Scalar<DataVector>& extrinsic_curvature_trace,
      const tnsr::I<DataVector, 3>& shift_background) const noexcept {
    const DataVector r = get(magnitude(x));
    if (r[0] > enclosing_radius_) {
      get(*conformal_factor) = 1.;
      get(*lapse_times_conformal_factor) = 1.;
      get<0>(*shift_excess) = 0.;
      get<1>(*shift_excess) = 0.;
      get<2>(*shift_excess) = 0.;
      // get<0>(shift_excess) = background.angular_velocity() *
      // center_of_mass_[1]; get<1>(shift_excess) =
      //     -background.angular_velocity() * center_of_mass_[0];
      // get<2>(shift_excess) = 0.;
      // for (size_t i = 0; i < 3; ++i) {
      //   shift_excess.get(i) += background.radial_velocity() * x.get(i);
      // }
    } else {
      auto x_centered = x;
      if (get<0>(x)[0] > 0) {
        get<0>(x_centered) -= positions_[1];
      } else {
        get<0>(x_centered) -= positions_[0];
      }
      const DataVector r_centered = get(magnitude(x_centered));

      // Conformal factor
      Scalar<DataVector> nn_dot_longitudinal_shift{face_normal.begin()->size()};
      normal_dot_flux(make_not_null(&nn_dot_longitudinal_shift), face_normal,
                      *n_dot_longitudinal_shift);
      get(*n_dot_conformal_factor_gradient) =
          -0.5 * get(*conformal_factor) / r_centered +
          get(extrinsic_curvature_trace) * cube(get(*conformal_factor)) / 6. -
          0.125 * pow<4>(get(*conformal_factor)) /
              get(*lapse_times_conformal_factor) *
              get(nn_dot_longitudinal_shift);

      // Lapse
      get(*n_dot_lapse_times_conformal_factor_gradient) = 0.;

      // Shift
      DataVector beta_orthogonal =
          get(*lapse_times_conformal_factor) / cube(get(*conformal_factor));
      for (size_t i = 0; i < 3; ++i) {
        shift_excess->get(i) =
            beta_orthogonal * face_normal.get(i) - shift_background.get(i);
      }
    }
  }

  void pup(PUP::er& p) noexcept {
    p | positions_;
    p | enclosing_radius_;
    p | negative_expansion_;
  }

 private:
  std::array<double, 2> positions_;
  double enclosing_radius_;
  double negative_expansion_;
};

struct LinearizedApparentHorizons {
 public:
  struct Positions {
    using type = std::array<double, 2>;
    static constexpr Options::String help{"pos"};
  };

  struct EnclosingRadius {
    using type = double;
    static constexpr Options::String help{"enclosing radius"};
  };

  using options = tmpl::list<Positions, EnclosingRadius>;
  static constexpr Options::String help{"AH boundary conditions"};

  LinearizedApparentHorizons() = default;
  LinearizedApparentHorizons(const LinearizedApparentHorizons&) noexcept =
      default;
  LinearizedApparentHorizons& operator=(
      const LinearizedApparentHorizons&) noexcept = default;
  LinearizedApparentHorizons(LinearizedApparentHorizons&&) noexcept = default;
  LinearizedApparentHorizons& operator=(LinearizedApparentHorizons&&) noexcept =
      default;
  ~LinearizedApparentHorizons() noexcept = default;

  LinearizedApparentHorizons(std::array<double, 2> positions,
                             const double enclosing_radius) noexcept
      : positions_(std::move(positions)), enclosing_radius_(enclosing_radius) {}

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
      const tnsr::i<DataVector, 3>& face_normal,
      const tnsr::I<DataVector, 3>& x,
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
      auto x_centered = x;
      if (get<0>(x)[0] > 0) {
        get<0>(x_centered) -= positions_[1];
      } else {
        get<0>(x_centered) -= positions_[0];
      }
      const DataVector r_centered = get(magnitude(x_centered));

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
          -0.5 * get(*conformal_factor_correction) / r_centered +
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

  void pup(PUP::er& p) noexcept {
    p | positions_;
    p | enclosing_radius_;
  }

 private:
  std::array<double, 2> positions_;
  double enclosing_radius_;
};

}  // namespace Xcts::BoundaryConditions
