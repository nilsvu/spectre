// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/Xcts/AnalyticData.hpp"

#include <algorithm>
#include <ostream>
#include <pup.h>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "Options/ParseOptions.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
namespace Xcts::AnalyticData::detail {

void DerivativeVariables::operator()(
    const gsl::not_null<tnsr::iJkk<DataVector, Dim>*>
        deriv_conformal_christoffel_second_kind,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<
        Tags::ConformalChristoffelSecondKind<DataVector, Dim, Frame::Inertial>,
        tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const noexcept {
  Variables<tmpl::list<
      Tags::ConformalChristoffelSecondKind<DataVector, Dim, Frame::Inertial>>>
      vars{mesh.number_of_grid_points()};
  get<Tags::ConformalChristoffelSecondKind<DataVector, Dim, Frame::Inertial>>(
      vars) = conformal_christoffel_second_kind;
  const auto derivs = partial_derivatives<tmpl::list<
      Tags::ConformalChristoffelSecondKind<DataVector, Dim, Frame::Inertial>>>(
      vars, mesh, inv_jacobian);
  *deriv_conformal_christoffel_second_kind = get<::Tags::deriv<
      Tags::ConformalChristoffelSecondKind<DataVector, Dim, Frame::Inertial>,
      tmpl::size_t<Dim>, Frame::Inertial>>(derivs);
}

void DerivativeVariables::operator()(
    const gsl::not_null<tnsr::ii<DataVector, Dim>*> conformal_ricci_tensor,
    const gsl::not_null<Cache*> cache,
    Tags::ConformalRicciTensor<DataVector, Dim, Frame::Inertial> /*meta*/)
    const noexcept {
  const auto& deriv_conformal_christoffel_second_kind = cache->get_var(
      ::Tags::deriv<Tags::ConformalChristoffelSecondKind<DataVector, Dim,
                                                         Frame::Inertial>,
                    tmpl::size_t<Dim>, Frame::Inertial>{});
  gr::ricci_tensor(conformal_ricci_tensor, conformal_christoffel_second_kind,
                   deriv_conformal_christoffel_second_kind);
}

void DerivativeVariables::operator()(
    const gsl::not_null<Scalar<DataVector>*> conformal_ricci_scalar,
    const gsl::not_null<Cache*> cache,
    Tags::ConformalRicciScalar<DataVector> /*meta*/) const noexcept {
  const auto& conformal_ricci_tensor = cache->get_var(
      Tags::ConformalRicciTensor<DataVector, Dim, Frame::Inertial>{});
  trace(conformal_ricci_scalar, conformal_ricci_tensor, inv_conformal_metric);
}

void DerivativeVariables::operator()(
    const gsl::not_null<tnsr::i<DataVector, Dim>*>
        deriv_extrinsic_curvature_trace,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataVector>,
                  tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const noexcept {
  Variables<tmpl::list<gr::Tags::TraceExtrinsicCurvature<DataVector>>> vars{
      mesh.number_of_grid_points()};
  get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(vars) =
      extrinsic_curvature_trace;
  const auto derivs = partial_derivatives<
      tmpl::list<gr::Tags::TraceExtrinsicCurvature<DataVector>>>(vars, mesh,
                                                                 inv_jacobian);
  *deriv_extrinsic_curvature_trace =
      get<::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataVector>,
                        tmpl::size_t<Dim>, Frame::Inertial>>(derivs);
}

void DerivativeVariables::operator()(
    const gsl::not_null<Scalar<DataVector>*>
        shift_dot_deriv_extrinsic_curvature_trace,
    const gsl::not_null<Cache*> cache,
    Tags::ShiftDotDerivExtrinsicCurvatureTrace<DataVector> /*meta*/)
    const noexcept {
  const auto& deriv_extrinsic_curvature_trace = cache->get_var(
      ::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataVector>,
                    tmpl::size_t<3>, Frame::Inertial>{});
  dot_product(shift_dot_deriv_extrinsic_curvature_trace, shift,
              deriv_extrinsic_curvature_trace);
}

}  // namespace Xcts::AnalyticData::detail
/// \endcond
