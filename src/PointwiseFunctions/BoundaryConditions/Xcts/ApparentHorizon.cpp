// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/BoundaryConditions/Xcts/ApparentHorizon.hpp"

#include <utility>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Equations.hpp"
#include "ErrorHandling/Assert.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Xcts::BoundaryConditions {

void add_normal_gradient_term_euclidean(
    const gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
    const Scalar<DataVector>& conformal_factor,
    const DataVector& euclidean_radius) noexcept {
  get(*n_dot_conformal_factor_gradient) -=
      0.5 * get(conformal_factor) / euclidean_radius;
}

void add_normal_gradient_term_non_euclidean(
    const gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::i<DataVector, 3>& conformal_horizon_normal,
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
      ASSERT(equal_within_roundoff(conformal_horizon_normal.get(i)[j],
                                   inertial_coords.get(i)[j] /
                                       sqrt(non_euclidean_radius_square[j])),
             "Horizon normal is incorrect at point ("
                 << get<0>(inertial_coords)[j] << ","
                 << get<1>(inertial_coords)[j] << ","
                 << get<2>(inertial_coords)[j] << ") in dim=" << i
                 << ". Expected " << inertial_coords.get(i)[j] << " / "
                 << sqrt(non_euclidean_radius_square[j]) << ", got "
                 << conformal_horizon_normal.get(i)[j]);
    }
  }
  Scalar<DataVector> projected_normal_gradient{inertial_coords.begin()->size(),
                                               0.};
  for (size_t i = 0; i < 3; ++i) {
    get(projected_normal_gradient) +=
        (inv_conformal_metric.get(i, i) -
         square(conformal_horizon_normal.get(i))) /
        sqrt(non_euclidean_radius_square);
    for (size_t j = 0; j <= i; ++j) {
      DataVector projection =
          inv_conformal_metric.get(i, j) -
          conformal_horizon_normal.get(i) * conformal_horizon_normal.get(j);
      for (size_t k = 0; k < 3; ++k) {
        get(projected_normal_gradient) -=
            projection * conformal_horizon_normal.get(k) *
            conformal_christoffel_second_kind.get(k, i, j);
      }
    }
  }
  get(*n_dot_conformal_factor_gradient) -=
      0.25 * get(conformal_factor) * get(projected_normal_gradient);
}

template <Xcts::Geometry ConformalGeometry>
void apparent_horizon(
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
    const std::optional<tnsr::II<DataVector, 3>>& inv_conformal_metric,
    const std::optional<tnsr::Ijj<DataVector, 3>>&
        conformal_christoffel_second_kind) noexcept {
  // The conformal unit normal to the horizon surface s_i
  const auto& conformal_horizon_normal = inward_pointing_face_normal;
  // Conformal factor
  tnsr::I<DataVector, 3> n_dot_longitudinal_shift{x.begin()->size()};
  normal_dot_flux(make_not_null(&n_dot_longitudinal_shift),
                  conformal_horizon_normal, longitudinal_shift_background);
  for (size_t i = 0; i < 3; ++i) {
    n_dot_longitudinal_shift.get(i) += n_dot_longitudinal_shift_excess->get(i);
  }
  Scalar<DataVector> nn_dot_longitudinal_shift{x.begin()->size()};
  normal_dot_flux(make_not_null(&nn_dot_longitudinal_shift),
                  conformal_horizon_normal, n_dot_longitudinal_shift);
  get(*n_dot_conformal_factor_gradient) =
      get(extrinsic_curvature_trace) * cube(get(*conformal_factor)) / 6. -
      pow<4>(get(*conformal_factor)) / 8. / get(*lapse_times_conformal_factor) *
          get(nn_dot_longitudinal_shift);
  if constexpr (ConformalGeometry == Xcts::Geometry::Euclidean) {
    const DataVector euclidean_radius = get(magnitude(x));
    add_normal_gradient_term_euclidean(n_dot_conformal_factor_gradient,
                                       *conformal_factor, euclidean_radius);
  } else {
    add_normal_gradient_term_non_euclidean(
        n_dot_conformal_factor_gradient, *conformal_factor,
        conformal_horizon_normal, *inv_conformal_metric,
        *conformal_christoffel_second_kind, x);
  }

  // Lapse
  get(*n_dot_lapse_times_conformal_factor_gradient) = 0.;

  // Shift
  DataVector beta_orthogonal =
      get(*lapse_times_conformal_factor) / cube(get(*conformal_factor));
  for (size_t i = 0; i < 3; ++i) {
    shift_excess->get(i) = beta_orthogonal * conformal_horizon_normal.get(i) -
                           shift_background.get(i);
  }
}

template <Xcts::Geometry ConformalGeometry>
void linearized_apparent_horizon(
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
    const std::optional<tnsr::II<DataVector, 3>>& inv_conformal_metric,
    const std::optional<tnsr::Ijj<DataVector, 3>>&
        conformal_christoffel_second_kind) noexcept {
  // The conformal unit normal to the horizon surface s_i
  const auto& conformal_horizon_normal = inward_pointing_face_normal;
  // Conformal factor
  tnsr::I<DataVector, 3> n_dot_longitudinal_shift{x.begin()->size()};
  normal_dot_flux(make_not_null(&n_dot_longitudinal_shift),
                  conformal_horizon_normal, longitudinal_shift_background);
  for (size_t i = 0; i < 3; ++i) {
    n_dot_longitudinal_shift.get(i) += n_dot_longitudinal_shift_excess.get(i);
  }
  Scalar<DataVector> nn_dot_longitudinal_shift{x.begin()->size()};
  normal_dot_flux(make_not_null(&nn_dot_longitudinal_shift),
                  conformal_horizon_normal, n_dot_longitudinal_shift);
  Scalar<DataVector> nn_dot_longitudinal_shift_correction{x.begin()->size()};
  normal_dot_flux(make_not_null(&nn_dot_longitudinal_shift_correction),
                  conformal_horizon_normal,
                  *n_dot_longitudinal_shift_correction);
  get(*n_dot_conformal_factor_gradient_correction) =
      0.5 * get(extrinsic_curvature_trace) * square(get(conformal_factor)) *
          get(*conformal_factor_correction) -
      0.5 * pow<3>(get(conformal_factor)) / get(lapse_times_conformal_factor) *
          get(nn_dot_longitudinal_shift) * get(*conformal_factor_correction) +
      0.125 * pow<4>(get(conformal_factor)) /
          square(get(lapse_times_conformal_factor)) *
          get(nn_dot_longitudinal_shift) *
          get(*lapse_times_conformal_factor_correction) -
      0.125 * pow<4>(get(conformal_factor)) /
          get(lapse_times_conformal_factor) *
          get(nn_dot_longitudinal_shift_correction);
  if constexpr (ConformalGeometry == Xcts::Geometry::Euclidean) {
    const DataVector euclidean_radius = get(magnitude(x));
    add_normal_gradient_term_euclidean(
        n_dot_conformal_factor_gradient_correction,
        *conformal_factor_correction, euclidean_radius);
  } else {
    add_normal_gradient_term_non_euclidean(
        n_dot_conformal_factor_gradient_correction,
        *conformal_factor_correction, conformal_horizon_normal,
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
        beta_orthogonal * conformal_horizon_normal.get(i);
  }
}

template <Xcts::Geometry ConformalGeometry>
ApparentHorizon<ConformalGeometry>::ApparentHorizon(const double enclosing_radius) noexcept
    : enclosing_radius_(enclosing_radius) {}

template <Xcts::Geometry ConformalGeometry>
typename ApparentHorizon<ConformalGeometry>::Linearization
ApparentHorizon<ConformalGeometry>::linearization() const noexcept {
  return Linearization{enclosing_radius_};
}

template <Xcts::Geometry ConformalGeometry>
void ApparentHorizon<ConformalGeometry>::pup(PUP::er& p) noexcept {
  p | enclosing_radius_;
}

template <Xcts::Geometry ConformalGeometry>
LinearizedApparentHorizon<ConformalGeometry>::LinearizedApparentHorizon(
    const double enclosing_radius) noexcept
    : enclosing_radius_(enclosing_radius) {}

template <Xcts::Geometry ConformalGeometry>
void LinearizedApparentHorizon<ConformalGeometry>::pup(PUP::er& p) noexcept {
  p | enclosing_radius_;
}

#define GEOM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template void apparent_horizon<GEOM(data)>(                                 \
      gsl::not_null<Scalar<DataVector>*> conformal_factor,                    \
      gsl::not_null<Scalar<DataVector>*> lapse_times_conformal_factor,        \
      gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,                    \
      gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,     \
      gsl::not_null<Scalar<DataVector>*>                                      \
          n_dot_lapse_times_conformal_factor_gradient,                        \
      gsl::not_null<tnsr::I<DataVector, 3>*> n_dot_longitudinal_shift_excess, \
      const tnsr::i<DataVector, 3>& inward_pointing_face_normal,              \
      const tnsr::I<DataVector, 3>& x,                                        \
      const Scalar<DataVector>& extrinsic_curvature_trace,                    \
      const tnsr::I<DataVector, 3>& shift_background,                         \
      const tnsr::II<DataVector, 3>& longitudinal_shift_background,           \
      const std::optional<tnsr::II<DataVector, 3>>& inv_conformal_metric,     \
      const std::optional<tnsr::Ijj<DataVector, 3>>&                          \
          conformal_christoffel_second_kind) noexcept;                        \
  template void linearized_apparent_horizon<GEOM(data)>(                      \
      gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,         \
      gsl::not_null<Scalar<DataVector>*>                                      \
          lapse_times_conformal_factor_correction,                            \
      gsl::not_null<tnsr::I<DataVector, 3>*> shift_correction,                \
      gsl::not_null<Scalar<DataVector>*>                                      \
          n_dot_conformal_factor_gradient_correction,                         \
      gsl::not_null<Scalar<DataVector>*>                                      \
          n_dot_lapse_times_conformal_factor_gradient_correction,             \
      gsl::not_null<tnsr::I<DataVector, 3>*>                                  \
          n_dot_longitudinal_shift_correction,                                \
      const tnsr::i<DataVector, 3>& inward_pointing_face_normal,              \
      const tnsr::I<DataVector, 3>& x,                                        \
      const Scalar<DataVector>& extrinsic_curvature_trace,                    \
      const tnsr::II<DataVector, 3>& longitudinal_shift_background,           \
      const Scalar<DataVector>& conformal_factor,                             \
      const Scalar<DataVector>& lapse_times_conformal_factor,                 \
      const tnsr::I<DataVector, 3>& n_dot_longitudinal_shift_excess,          \
      const std::optional<tnsr::II<DataVector, 3>>& inv_conformal_metric,     \
      const std::optional<tnsr::Ijj<DataVector, 3>>&                          \
          conformal_christoffel_second_kind) noexcept;                        \
  template class ApparentHorizon<GEOM(data)>;                                 \
  template class LinearizedApparentHorizon<GEOM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (Geometry::Euclidean, Geometry::NonEuclidean))

#undef INSTANTIATE
#undef GEOM

}  // namespace Xcts::BoundaryConditions
