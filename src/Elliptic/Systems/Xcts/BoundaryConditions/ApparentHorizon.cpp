// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Xcts/BoundaryConditions/ApparentHorizon.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/Gsl.hpp"

namespace Xcts::BoundaryConditions::detail {

void add_normal_gradient_term_flat_cartesian(
    const gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
    const Scalar<DataVector>& conformal_factor,
    const DataVector& euclidean_radius) noexcept {
  get(*n_dot_conformal_factor_gradient) +=
      0.5 * get(conformal_factor) / euclidean_radius;
}

void add_normal_gradient_term_curved(
    const gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::i<DataVector, 3>& minus_conformal_horizon_normal,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind,
    const tnsr::I<DataVector, 3>& inertial_coords) noexcept {
  DataVector non_euclidean_radius_square{inertial_coords.begin()->size(), 0.};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      non_euclidean_radius_square += inv_conformal_metric.get(i, j) *
                                     inertial_coords.get(i) *
                                     inertial_coords.get(j);
    }
  }
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < non_euclidean_radius_square.size(); ++j) {
      ASSERT(equal_within_roundoff(minus_conformal_horizon_normal.get(i)[j],
                                   -inertial_coords.get(i)[j] /
                                       sqrt(non_euclidean_radius_square[j])),
             "Horizon normal is incorrect at point ("
                 << get<0>(inertial_coords)[j] << ","
                 << get<1>(inertial_coords)[j] << ","
                 << get<2>(inertial_coords)[j] << ") in dim=" << i
                 << ". Expected " << inertial_coords.get(i)[j] << " / "
                 << sqrt(non_euclidean_radius_square[j]) << ", got "
                 << -minus_conformal_horizon_normal.get(i)[j]);
    }
  }
  Scalar<DataVector> projected_normal_gradient{inertial_coords.begin()->size(),
                                               0.};
  for (size_t i = 0; i < 3; ++i) {
    get(projected_normal_gradient) +=
        (inv_conformal_metric.get(i, i) -
         square(minus_conformal_horizon_normal.get(i))) /
        sqrt(non_euclidean_radius_square);
    for (size_t j = 0; j <= i; ++j) {
      DataVector projection = inv_conformal_metric.get(i, j) -
                              minus_conformal_horizon_normal.get(i) *
                                  minus_conformal_horizon_normal.get(j);
      for (size_t k = 0; k < 3; ++k) {
        get(projected_normal_gradient) +=
            projection * minus_conformal_horizon_normal.get(k) *
            conformal_christoffel_second_kind.get(k, i, j);
      }
    }
  }
  get(*n_dot_conformal_factor_gradient) +=
      0.25 * get(conformal_factor) * get(projected_normal_gradient);
}

template <Xcts::Geometry ConformalGeometry>
void apparent_horizon_impl(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor,
    const gsl::not_null<Scalar<DataVector>*> lapse_times_conformal_factor,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
    const gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_lapse_times_conformal_factor_gradient,
    const gsl::not_null<tnsr::I<DataVector, 3>*>
        n_dot_longitudinal_shift_excess,
    const tnsr::i<DataVector, 3>& face_normal, const tnsr::I<DataVector, 3>& x,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const tnsr::I<DataVector, 3>& shift_background,
    const tnsr::II<DataVector, 3>& longitudinal_shift_background,
    const std::optional<std::reference_wrapper<const tnsr::II<DataVector, 3>>>
        inv_conformal_metric,
    const std::optional<std::reference_wrapper<const tnsr::Ijj<DataVector, 3>>>
        conformal_christoffel_second_kind) noexcept {
  // The conformal unit normal to the horizon surface s_i. Note that the face
  // normal points _out_ of the computational domain, i.e. _into_ the excised
  // region.
  const auto& minus_conformal_horizon_normal = face_normal;
  // Conformal factor
  tnsr::I<DataVector, 3> n_dot_longitudinal_shift{x.begin()->size()};
  normal_dot_flux(make_not_null(&n_dot_longitudinal_shift),
                  minus_conformal_horizon_normal,
                  longitudinal_shift_background);
  for (size_t i = 0; i < 3; ++i) {
    n_dot_longitudinal_shift.get(i) += n_dot_longitudinal_shift_excess->get(i);
  }
  Scalar<DataVector> nn_dot_longitudinal_shift{x.begin()->size()};
  normal_dot_flux(make_not_null(&nn_dot_longitudinal_shift),
                  minus_conformal_horizon_normal, n_dot_longitudinal_shift);
  get(*n_dot_conformal_factor_gradient) =
      -get(extrinsic_curvature_trace) * cube(get(*conformal_factor)) / 6. +
      pow<4>(get(*conformal_factor)) / 8. / get(*lapse_times_conformal_factor) *
          get(nn_dot_longitudinal_shift);
  if constexpr (ConformalGeometry == Xcts::Geometry::FlatCartesian) {
    const DataVector euclidean_radius = get(magnitude(x));
    add_normal_gradient_term_flat_cartesian(
        n_dot_conformal_factor_gradient, *conformal_factor, euclidean_radius);
  } else {
    add_normal_gradient_term_curved(
        n_dot_conformal_factor_gradient, *conformal_factor,
        minus_conformal_horizon_normal, *inv_conformal_metric,
        *conformal_christoffel_second_kind, x);
  }

  // Lapse
  get(*n_dot_lapse_times_conformal_factor_gradient) = 0.;

  // Shift
  DataVector beta_orthogonal =
      get(*lapse_times_conformal_factor) / cube(get(*conformal_factor));
  for (size_t i = 0; i < 3; ++i) {
    shift_excess->get(i) =
        -beta_orthogonal * minus_conformal_horizon_normal.get(i) -
        shift_background.get(i);
  }
}

template <Xcts::Geometry ConformalGeometry>
void linearized_apparent_horizon_impl(
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
    const tnsr::i<DataVector, 3>& face_normal, const tnsr::I<DataVector, 3>& x,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const tnsr::II<DataVector, 3>& longitudinal_shift_background,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, 3>& n_dot_longitudinal_shift_excess,
    const std::optional<std::reference_wrapper<const tnsr::II<DataVector, 3>>>
        inv_conformal_metric,
    const std::optional<std::reference_wrapper<const tnsr::Ijj<DataVector, 3>>>
        conformal_christoffel_second_kind) noexcept {
  // The conformal unit normal to the horizon surface s_i
  const auto& minus_conformal_horizon_normal = face_normal;
  // Conformal factor
  tnsr::I<DataVector, 3> n_dot_longitudinal_shift{x.begin()->size()};
  normal_dot_flux(make_not_null(&n_dot_longitudinal_shift),
                  minus_conformal_horizon_normal,
                  longitudinal_shift_background);
  for (size_t i = 0; i < 3; ++i) {
    n_dot_longitudinal_shift.get(i) += n_dot_longitudinal_shift_excess.get(i);
  }
  Scalar<DataVector> nn_dot_longitudinal_shift{x.begin()->size()};
  normal_dot_flux(make_not_null(&nn_dot_longitudinal_shift),
                  minus_conformal_horizon_normal, n_dot_longitudinal_shift);
  Scalar<DataVector> nn_dot_longitudinal_shift_correction{x.begin()->size()};
  normal_dot_flux(make_not_null(&nn_dot_longitudinal_shift_correction),
                  minus_conformal_horizon_normal,
                  *n_dot_longitudinal_shift_correction);
  get(*n_dot_conformal_factor_gradient_correction) =
      -0.5 * get(extrinsic_curvature_trace) * square(get(conformal_factor)) *
          get(*conformal_factor_correction) +
      0.5 * pow<3>(get(conformal_factor)) / get(lapse_times_conformal_factor) *
          get(nn_dot_longitudinal_shift) * get(*conformal_factor_correction) -
      0.125 * pow<4>(get(conformal_factor)) /
          square(get(lapse_times_conformal_factor)) *
          get(nn_dot_longitudinal_shift) *
          get(*lapse_times_conformal_factor_correction) +
      0.125 * pow<4>(get(conformal_factor)) /
          get(lapse_times_conformal_factor) *
          get(nn_dot_longitudinal_shift_correction);
  if constexpr (ConformalGeometry == Xcts::Geometry::FlatCartesian) {
    const DataVector euclidean_radius = get(magnitude(x));
    add_normal_gradient_term_flat_cartesian(
        n_dot_conformal_factor_gradient_correction,
        *conformal_factor_correction, euclidean_radius);
  } else {
    add_normal_gradient_term_curved(
        n_dot_conformal_factor_gradient_correction,
        *conformal_factor_correction, minus_conformal_horizon_normal,
        *inv_conformal_metric, *conformal_christoffel_second_kind, x);
  }

  // Lapse
  get(*n_dot_lapse_times_conformal_factor_gradient_correction) = 0.;

  // Shift
  DataVector beta_orthogonal = get(*lapse_times_conformal_factor_correction) /
                                   cube(get(conformal_factor)) -
                               3. * get(lapse_times_conformal_factor) /
                                   pow<4>(get(conformal_factor)) *
                                   get(*conformal_factor_correction);
  for (size_t i = 0; i < 3; ++i) {
    shift_correction->get(i) =
        -beta_orthogonal * minus_conformal_horizon_normal.get(i);
  }
}

template <Xcts::Geometry ConformalGeometry>
void ApparentHorizonImpl<ConformalGeometry>::apply(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor,
    const gsl::not_null<Scalar<DataVector>*> lapse_times_conformal_factor,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
    const gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_lapse_times_conformal_factor_gradient,
    const gsl::not_null<tnsr::I<DataVector, 3>*>
        n_dot_longitudinal_shift_excess,
    const tnsr::i<DataVector, 3>& face_normal, const tnsr::I<DataVector, 3>& x,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const tnsr::I<DataVector, 3>& shift_background,
    const tnsr::II<DataVector, 3>& longitudinal_shift_background)
    const noexcept {
  apparent_horizon_impl<ConformalGeometry>(
      conformal_factor, lapse_times_conformal_factor, shift_excess,
      n_dot_conformal_factor_gradient,
      n_dot_lapse_times_conformal_factor_gradient,
      n_dot_longitudinal_shift_excess, face_normal, x,
      extrinsic_curvature_trace, shift_background,
      longitudinal_shift_background, std::nullopt, std::nullopt);
}

template <Xcts::Geometry ConformalGeometry>
void ApparentHorizonImpl<ConformalGeometry>::apply(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor,
    const gsl::not_null<Scalar<DataVector>*> lapse_times_conformal_factor,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
    const gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_lapse_times_conformal_factor_gradient,
    const gsl::not_null<tnsr::I<DataVector, 3>*>
        n_dot_longitudinal_shift_excess,
    const tnsr::i<DataVector, 3>& face_normal, const tnsr::I<DataVector, 3>& x,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const tnsr::I<DataVector, 3>& shift_background,
    const tnsr::II<DataVector, 3>& longitudinal_shift_background,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind)
    const noexcept {
  apparent_horizon_impl<ConformalGeometry>(
      conformal_factor, lapse_times_conformal_factor, shift_excess,
      n_dot_conformal_factor_gradient,
      n_dot_lapse_times_conformal_factor_gradient,
      n_dot_longitudinal_shift_excess, face_normal, x,
      extrinsic_curvature_trace, shift_background,
      longitudinal_shift_background, inv_conformal_metric,
      conformal_christoffel_second_kind);
}

template <Xcts::Geometry ConformalGeometry>
void ApparentHorizonImpl<ConformalGeometry>::apply_linearized(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
    const gsl::not_null<Scalar<DataVector>*>
        lapse_times_conformal_factor_correction,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess_correction,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_conformal_factor_gradient_correction,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_lapse_times_conformal_factor_gradient_correction,
    const gsl::not_null<tnsr::I<DataVector, 3>*>
        n_dot_longitudinal_shift_excess_correction,
    const tnsr::i<DataVector, 3>& face_normal, const tnsr::I<DataVector, 3>& x,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const tnsr::II<DataVector, 3>& longitudinal_shift_background,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, 3>& n_dot_longitudinal_shift_excess)
    const noexcept {
  linearized_apparent_horizon_impl<ConformalGeometry>(
      conformal_factor_correction, lapse_times_conformal_factor_correction,
      shift_excess_correction, n_dot_conformal_factor_gradient_correction,
      n_dot_lapse_times_conformal_factor_gradient_correction,
      n_dot_longitudinal_shift_excess_correction, face_normal, x,
      extrinsic_curvature_trace, longitudinal_shift_background,
      conformal_factor, lapse_times_conformal_factor,
      n_dot_longitudinal_shift_excess, std::nullopt, std::nullopt);
}

template <Xcts::Geometry ConformalGeometry>
void ApparentHorizonImpl<ConformalGeometry>::apply_linearized(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
    const gsl::not_null<Scalar<DataVector>*>
        lapse_times_conformal_factor_correction,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess_correction,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_conformal_factor_gradient_correction,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_lapse_times_conformal_factor_gradient_correction,
    const gsl::not_null<tnsr::I<DataVector, 3>*>
        n_dot_longitudinal_shift_excess_correction,
    const tnsr::i<DataVector, 3>& face_normal, const tnsr::I<DataVector, 3>& x,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const tnsr::II<DataVector, 3>& longitudinal_shift_background,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, 3>& n_dot_longitudinal_shift_excess,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind)
    const noexcept {
  linearized_apparent_horizon_impl<ConformalGeometry>(
      conformal_factor_correction, lapse_times_conformal_factor_correction,
      shift_excess_correction, n_dot_conformal_factor_gradient_correction,
      n_dot_lapse_times_conformal_factor_gradient_correction,
      n_dot_longitudinal_shift_excess_correction, face_normal, x,
      extrinsic_curvature_trace, longitudinal_shift_background,
      conformal_factor, lapse_times_conformal_factor,
      n_dot_longitudinal_shift_excess, inv_conformal_metric,
      conformal_christoffel_second_kind);
}

template <Xcts::Geometry ConformalGeometry>
void ApparentHorizonImpl<ConformalGeometry>::pup(PUP::er& /*p*/) noexcept {}

template <Xcts::Geometry ConformalGeometry>
bool operator==(
    const ApparentHorizonImpl<ConformalGeometry>& /*lhs*/,
    const ApparentHorizonImpl<ConformalGeometry>& /*rhs*/) noexcept {
  return true;
}

template <Xcts::Geometry ConformalGeometry>
bool operator!=(const ApparentHorizonImpl<ConformalGeometry>& lhs,
                const ApparentHorizonImpl<ConformalGeometry>& rhs) noexcept {
  return not(lhs == rhs);
}

template class ApparentHorizonImpl<Xcts::Geometry::FlatCartesian>;
template class ApparentHorizonImpl<Xcts::Geometry::Curved>;

}  // namespace Xcts::BoundaryConditions::detail
