// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "IO/Importers/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "Parallel/AlgorithmMetafunctions.hpp"
#include "Parallel/GlobalCache.hpp"
#include "PointwiseFunctions/GeneralRelativity/DerivativesOfSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TimeDerivativeOfSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/TimeDerivativeOfSpatialMetric.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace GeneralizedHarmonic::Actions {

template <typename ImporterOptionsGroup, typename NumericInitialDataTags,
          typename GhTags>
struct SetNumericInitialData {
  static constexpr size_t Dim = 3;
  using inbox_tags =
      tmpl::list<importers::Tags::VolumeData<ImporterOptionsGroup,
                                             NumericInitialDataTags>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, Parallel::AlgorithmExecution>
  apply(db::DataBox<DbTagsList>& box,
        tuples::TaggedTuple<InboxTags...>& inboxes,
        const Parallel::GlobalCache<Metavariables>& /*cache*/,
        const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
        const ParallelComponent* const /*meta*/) {
    auto& inbox = tuples::get<importers::Tags::VolumeData<
        ImporterOptionsGroup, NumericInitialDataTags>>(inboxes);
    // Using `0` for the temporal ID since we only read the volume data once, so
    // there's no need to keep track of the temporal ID.
    if (inbox.find(0_st) == inbox.end()) {
      return {std::move(box), Parallel::AlgorithmExecution::Retry};
    }
    const auto numeric_initial_data =
        variables_from_tagged_tuple(std::move(inbox.extract(0_st).mapped()));

    // Take numerical derivatives of initial data
    const auto& mesh = db::get<domain::Tags::Mesh<Dim>>(box);
    const auto& inv_jacobian =
        db::get<domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                              Frame::Inertial>>(box);
    // TODO: No need to take deriv of extr curv
    const auto deriv_numeric_initial_data =
        partial_derivatives<NumericInitialDataTags>(numeric_initial_data, mesh,
                                                    inv_jacobian);

    // Compute GH quantities from initial data
    db::mutate<gr::Tags::SpacetimeMetric<3, Frame::Inertial, DataVector>,
               Tags::Pi<3, Frame::Inertial>, Tags::Phi<3, Frame::Inertial>>(
        make_not_null(&box),
        [](const gsl::not_null<tnsr::aa<DataVector, 3>*> spacetime_metric,
           const gsl::not_null<tnsr::aa<DataVector, 3>*> pi,
           const gsl::not_null<tnsr::iaa<DataVector, 3>*> phi,
           const tnsr::ii<DataVector, 3>& spatial_metric,
           const tnsr::ijj<DataVector, 3>& deriv_spatial_metric,
           const Scalar<DataVector>& lapse,
           const tnsr::i<DataVector, 3>& deriv_lapse,
           const tnsr::I<DataVector, 3>& shift,
           const tnsr::iJ<DataVector, 3>& deriv_shift,
           const tnsr::ii<DataVector, 3>& extrinsic_curvature) {
          gr::spacetime_metric(spacetime_metric, lapse, shift, spatial_metric);
          const auto dt_spatial_metric = gr::time_derivative_of_spatial_metric(
              lapse, shift, deriv_shift, spatial_metric, deriv_spatial_metric,
              extrinsic_curvature);
          // Choose dt_lapse and dt_shift = 0 in inertial frame (for now)
          const auto dt_lapse = make_with_value<Scalar<DataVector>>(lapse, 0.);
          const auto dt_shift =
              make_with_value<tnsr::I<DataVector, 3>>(shift, 0.);
          const auto derivs_spacetime_metric =
              gr::derivatives_of_spacetime_metric(
                  lapse, dt_lapse, deriv_lapse, shift, dt_shift, deriv_shift,
                  spatial_metric, dt_spatial_metric, deriv_spatial_metric);
          // Phi
          for (size_t i = 0; i < Dim; ++i) {
            for (size_t a = 0; a < Dim + 1; ++a) {
              for (size_t b = 0; b <= a; ++b) {
                phi->get(i, a, b) = derivs_spacetime_metric.get(i + 1, a, b);
              }
            }
          }
          // Pi
          for (size_t a = 0; a < Dim + 1; ++a) {
            for (size_t b = 0; b <= a; ++b) {
              pi->get(a, b) = -derivs_spacetime_metric.get(0, a, b);
              for (size_t i = 0; i <= Dim; ++i) {
                pi->get(a, b) -= shift.get(i) * phi->get(i, a, b);
              }
              pi->get(a, b) /= get(lapse);
            }
          }
        },
        get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>(
            numeric_initial_data),
        get<::Tags::deriv<
            gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
            tmpl::size_t<Dim>, Frame::Inertial>>(deriv_numeric_initial_data),
        get<gr::Tags::Lapse<DataVector>>(numeric_initial_data),
        get<::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<Dim>,
                          Frame::Inertial>>(deriv_numeric_initial_data),
        get<gr::Tags::Shift<3, Frame::Inertial, DataVector>>(
            numeric_initial_data),
        get<::Tags::deriv<gr::Tags::Shift<3, Frame::Inertial, DataVector>,
                          tmpl::size_t<Dim>, Frame::Inertial>>(
            deriv_numeric_initial_data),
        get<gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector>>(
            numeric_initial_data));

    return {std::move(box), Parallel::AlgorithmExecution::Continue};
  }
};

}  // namespace GeneralizedHarmonic::Actions
