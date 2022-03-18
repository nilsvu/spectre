// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Xcts/SpacetimeQuantities.hpp"

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "PointwiseFunctions/Elasticity/Strain.hpp"
#include "PointwiseFunctions/Xcts/LongitudinalOperator.hpp"

namespace Xcts {

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<tnsr::i<DataVector, 3>*> deriv_conformal_factor,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<Tags::ConformalFactor<DataVector>, tmpl::size_t<3>,
                  Frame::Inertial> /*meta*/) const {
  partial_derivative(deriv_conformal_factor, conformal_factor, mesh,
                     inv_jacobian);
}

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<Scalar<DataVector>*>
        conformal_laplacian_of_conformal_factor,
    const gsl::not_null<Cache*> cache,
    detail::ConformalLaplacianOfConformalFactor<DataVector> /*meta*/) const {
  const auto& deriv_conformal_factor = cache->get_var(
      *this, ::Tags::deriv<Tags::ConformalFactor<DataVector>,
                           tmpl::size_t<3>, Frame::Inertial>{});
  const auto conformal_factor_flux = TensorExpressions::evaluate<ti_I>(
      inv_conformal_metric(ti_I, ti_J) * deriv_conformal_factor(ti_j));
  const auto deriv_conformal_factor_flux =
      partial_derivative(conformal_factor_flux, mesh, inv_jacobian);
  TensorExpressions::evaluate(
      conformal_laplacian_of_conformal_factor,
      deriv_conformal_factor_flux(ti_i, ti_I) +
          conformal_christoffel_contracted(ti_i) * conformal_factor_flux(ti_I));
}

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<tnsr::iJ<DataVector, 3>*> deriv_shift_excess,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<Tags::ShiftExcess<DataVector, 3, Frame::Inertial>,
                  tmpl::size_t<3>, Frame::Inertial> /*meta*/) const {
  partial_derivative(deriv_shift_excess, shift_excess, mesh, inv_jacobian);
}

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<tnsr::ii<DataVector, 3>*> shift_strain,
    const gsl::not_null<Cache*> cache,
    Tags::ShiftStrain<DataVector, 3, Frame::Inertial> /*meta*/) const {
  const auto& deriv_shift_excess = cache->get_var(
      *this, ::Tags::deriv<Tags::ShiftExcess<DataVector, 3, Frame::Inertial>,
                           tmpl::size_t<3>, Frame::Inertial>{});
  Elasticity::strain(shift_strain, deriv_shift_excess, conformal_metric,
                     deriv_conformal_metric, conformal_christoffel_first_kind,
                     shift_excess);
}

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<tnsr::II<DataVector, 3>*> longitudinal_shift_excess,
    const gsl::not_null<Cache*> cache,
    Tags::LongitudinalShiftExcess<DataVector, 3, Frame::Inertial> /*meta*/)
    const {
  const auto& shift_strain = cache->get_var(
      *this, Tags::ShiftStrain<DataVector, 3, Frame::Inertial>{});
  Xcts::longitudinal_operator(longitudinal_shift_excess, shift_strain,
                        inv_conformal_metric);
}

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<tnsr::ii<DataVector, 3>*> spatial_metric,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector> /*meta*/) const {
  *spatial_metric = conformal_metric;
  for (auto& spatial_metric_component : *spatial_metric) {
    spatial_metric_component *= pow<4>(get(conformal_factor));
  }
}

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<tnsr::II<DataVector, 3>*> inv_spatial_metric,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector> /*meta*/)
    const {
  *inv_spatial_metric = inv_conformal_metric;
  for (auto& inv_spatial_metric_component : *inv_spatial_metric) {
    inv_spatial_metric_component /= pow<4>(get(conformal_factor));
  }
}

// void SpacetimeQuantitiesComputer::operator()(
//     const gsl::not_null<tnsr::ijj<DataVector, 3>*> deriv_spatial_metric,
//     const gsl::not_null<Cache*> cache,
//     ::Tags::deriv<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
//                   tmpl::size_t<3>, Frame::Inertial> /*meta*/) const {
//   const auto& conformal_metric = cache->get_var(
//       *this, Tags::ConformalMetric<DataVector, 3, Frame::Inertial>{});
//   const auto& deriv_conformal_metric = cache->get_var(
//       *this,
//       ::Tags::deriv<Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
//                     tmpl::size_t<3>, Frame::Inertial>{});
//   const auto& conformal_factor =
//       cache->get_var(*this, Tags::ConformalFactor<DataVector>{});
//   const auto& deriv_conformal_factor =
//       cache->get_var(*this, ::Tags::deriv<Tags::ConformalFactor<DataVector>,
//                                           tmpl::size_t<3>,
//                                           Frame::Inertial>{});
//   TensorExpressions::evaluate<ti_i, ti_j, ti_k>(
//       deriv_spatial_metric,
//       pow<4>(conformal_factor()) * deriv_conformal_metric(ti_i, ti_j, ti_k) +
//           4. * pow<3>(conformal_factor()) * deriv_conformal_factor(ti_i) *
//               conformal_metric(ti_j, ti_k));
// }

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<Scalar<DataVector>*> lapse,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::Lapse<DataVector> /*meta*/) const {
  get(*lapse) = get(lapse_times_conformal_factor) / get(conformal_factor);
}

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::Shift<3, Frame::Inertial, DataVector> /*meta*/) const {
  for (size_t i = 0; i < 3; ++i) {
    shift->get(i) = shift_excess.get(i) + shift_background.get(i);
  }
}

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<tnsr::ii<DataVector, 3>*> extrinsic_curvature,
    const gsl::not_null<Cache*> cache,
    gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector> /*meta*/)
    const {
  const auto& lapse = cache->get_var(*this, gr::Tags::Lapse<DataVector>{});
  const auto& longitudinal_shift_excess = cache->get_var(
      *this, Tags::LongitudinalShiftExcess<DataVector, 3, Frame::Inertial>{});
  const auto& spatial_metric = cache->get_var(
      *this, gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>{});
  TensorExpressions::evaluate<ti_i, ti_j>(
      extrinsic_curvature,
      pow<4>(conformal_factor()) / (2. * lapse()) *
              conformal_metric(ti_i, ti_k) * conformal_metric(ti_j, ti_l) *
              (longitudinal_shift_excess(ti_K, ti_L) +
               longitudinal_shift_background_minus_dt_conformal_metric(ti_K,
                                                                       ti_L)) +
          spatial_metric(ti_i, ti_j) * trace_extrinsic_curvature() / 3.);
}

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<tnsr::ijj<DataVector, 3>*> deriv_extrinsic_curvature,
    const gsl::not_null<Cache*> cache,
    ::Tags::deriv<gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector>,
                  tmpl::size_t<3>, Frame::Inertial> /*meta*/) const {
  // const auto& extrinsic_curvature = cache->get_var(
  //     *this, gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector>{});
  // const auto& christoffel = cache->get_var(
  //     *this,
  //     gr::Tags::SpatialChristoffelSecondKind<3, Frame::Inertial,
  //     DataVector>{});
  // partial_derivative(deriv_extrinsic_curvature, extrinsic_curvature, mesh,
  //                    inv_jacobian);
  // for (size_t i = 0; i < 3; ++i) {
  //   for (size_t j = 0; j < 3; ++j) {
  //     for (size_t k = 0; k <= j; ++k) {
  //       for (size_t l = 0; l < 3; ++l) {
  //         deriv_extrinsic_curvature->get(i, j, k) -=
  //             christoffel.get(l, i, j) * extrinsic_curvature.get(l, k);
  //         deriv_extrinsic_curvature->get(i, j, k) -=
  //             christoffel.get(l, i, k) * extrinsic_curvature.get(j, l);
  //       }
  //     }
  //   }
  // }
}

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
    const gsl::not_null<Cache*> cache,
    gr::Tags::HamiltonianConstraint<DataVector> /*meta*/) const {
  const auto& conformal_laplacian_of_conformal_factor = cache->get_var(
      *this, detail::ConformalLaplacianOfConformalFactor<DataVector>{});
  const auto& lapse = cache->get_var(*this, gr::Tags::Lapse<DataVector>{});
  const auto& inv_spatial_metric = cache->get_var(
      *this, gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>{});
  const auto& extrinsic_curvature = cache->get_var(
      *this, gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector>{});
  // Eq. 3.12 in BaumgarteShapiro
  TensorExpressions::evaluate(
      hamiltonian_constraint,
      conformal_laplacian_of_conformal_factor() -
          conformal_factor() * conformal_ricci_scalar() / 8. -
          pow<5>(conformal_factor()) * square(trace_extrinsic_curvature()) / 8. +
          pow<5>(conformal_factor()) * inv_spatial_metric(ti_I, ti_K) *
              inv_spatial_metric(ti_J, ti_L) * extrinsic_curvature(ti_i, ti_j) *
              extrinsic_curvature(ti_k, ti_l) / 8. +
          2. * M_PI * pow<5>(conformal_factor()) * energy_density());
}

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<tnsr::i<DataVector, 3>*> momentum_constraint,
    const gsl::not_null<Cache*> cache,
    gr::Tags::MomentumConstraint<3, Frame::Inertial, DataVector> /*meta*/)
    const {
  // const auto& deriv_extrinsic_curvature = cache->get_var(
  //     *this, ::Tags::deriv<
  //                gr::Tags::ExtrinsicCurvature<3, Frame::Inertial,
  //                DataVector>, tmpl::size_t<3>, Frame::Inertial>{});
  // const auto& inv_spatial_metric = cache->get_var(
  //     *this, gr::Tags::InverseSpatialMetric<3, Frame::Inertial,
  //     DataVector>{});
  // for (size_t i = 0; i < 3; ++i) {
  //   momentum_constraint->get(i) = 0.;
  //   for (size_t j = 0; j < 3; ++j) {
  //     for (size_t k = 0; k < 3; ++k) {
  //       momentum_constraint->get(i) += inv_spatial_metric.get(j, k) *
  //                                      (deriv_extrinsic_curvature.get(j, k,
  //                                      i) -
  //                                       deriv_extrinsic_curvature.get(i, j,
  //                                       k));
  //     }
  //   }
  // }
}

}  // namespace Xcts
