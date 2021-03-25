// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/Xcts/Binary.hpp"

#include <algorithm>
#include <ostream>
#include <pup.h>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Options/ParseOptions.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Xcts/LongitudinalOperator.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
namespace Xcts::AnalyticData::detail {

void BinaryVariables::operator()(
    const gsl::not_null<tnsr::II<DataType, 3>*> inv_conformal_metric,
    const gsl::not_null<Cache*> /*cache*/,
    Tags::InverseConformalMetric<DataType, 3, Frame::Inertial> /*meta*/)
    const noexcept {
  Scalar<DataVector> unused_det{x.begin()->size()};
  determinant_and_inverse(make_not_null(&unused_det), inv_conformal_metric,
                          conformal_metric);
}

void BinaryVariables::operator()(
    const gsl::not_null<tnsr::ijj<DataType, Dim>*>
        conformal_christoffel_first_kind,
    const gsl::not_null<Cache*> /*cache*/,
    Tags::ConformalChristoffelFirstKind<
        DataType, Dim, Frame::Inertial> /*meta*/) const noexcept {
  gr::christoffel_first_kind(conformal_christoffel_first_kind,
                             deriv_conformal_metric);
}

void BinaryVariables::operator()(
    const gsl::not_null<tnsr::Ijj<DataType, Dim>*>
        conformal_christoffel_second_kind,
    const gsl::not_null<Cache*> cache,
    Tags::ConformalChristoffelSecondKind<
        DataType, Dim, Frame::Inertial> /*meta*/) const noexcept {
  const auto& conformal_christoffel_first_kind = cache->get_var(
      Tags::ConformalChristoffelFirstKind<DataType, Dim, Frame::Inertial>{});
  const auto& inv_conformal_metric = cache->get_var(
      Tags::InverseConformalMetric<DataType, Dim, Frame::Inertial>{});
  raise_or_lower_first_index(conformal_christoffel_second_kind,
                             conformal_christoffel_first_kind,
                             inv_conformal_metric);
}

void BinaryVariables::operator()(
    const gsl::not_null<tnsr::i<DataType, Dim>*>
        conformal_christoffel_contracted,
    const gsl::not_null<Cache*> cache,
    Tags::ConformalChristoffelContracted<
        DataType, Dim, Frame::Inertial> /*meta*/) const noexcept {
  const auto& conformal_christoffel_second_kind = cache->get_var(
      Tags::ConformalChristoffelSecondKind<DataType, Dim, Frame::Inertial>{});
  for (size_t i = 0; i < Dim; ++i) {
    conformal_christoffel_contracted->get(i) =
        conformal_christoffel_second_kind.get(0, i, 0);
    for (size_t j = 1; j < Dim; ++j) {
      conformal_christoffel_contracted->get(i) +=
          conformal_christoffel_second_kind.get(j, i, j);
    }
  }
}

void BinaryVariables::operator()(
    const gsl::not_null<tnsr::I<DataVector, Dim>*> shift_background,
    const gsl::not_null<Cache*> /*cache*/,
    Tags::ShiftBackground<DataType, Dim, Frame::Inertial> /*meta*/)
    const noexcept {
  get<0>(*shift_background) = -angular_velocity * get<1>(x);
  get<1>(*shift_background) = angular_velocity * get<0>(x);
  get<2>(*shift_background) = 0.;
}

void BinaryVariables::operator()(
    const gsl::not_null<tnsr::iJ<DataVector, Dim>*> deriv_shift_background,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<Tags::ShiftBackground<DataType, Dim, Frame::Inertial>,
                  tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const noexcept {
  std::fill(deriv_shift_background->begin(), deriv_shift_background->end(), 0.);
  get<1, 0>(*deriv_shift_background) = -angular_velocity;
  get<0, 1>(*deriv_shift_background) = angular_velocity;
}

void BinaryVariables::operator()(
    const gsl::not_null<tnsr::II<DataVector, Dim>*>
        longitudinal_shift_background,
    const gsl::not_null<Cache*> cache,
    Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
        DataVector, Dim, Frame::Inertial> /*meta*/) const noexcept {
  const auto& shift_background =
      cache->get_var(Tags::ShiftBackground<DataType, Dim, Frame::Inertial>{});
  const auto& deriv_shift_background = cache->get_var(
      ::Tags::deriv<Tags::ShiftBackground<DataType, Dim, Frame::Inertial>,
                    tmpl::size_t<Dim>, Frame::Inertial>{});
  const auto& inv_conformal_metric = cache->get_var(
      Tags::InverseConformalMetric<DataType, Dim, Frame::Inertial>{});
  const auto& conformal_christoffel_first_kind = cache->get_var(
      Tags::ConformalChristoffelFirstKind<DataType, Dim, Frame::Inertial>{});
  auto shift_background_strain =
      make_with_value<tnsr::ii<DataVector, Dim>>(x, 0.);
  Xcts::shift_strain(make_not_null(&shift_background_strain),
                     deriv_shift_background, conformal_metric,
                     deriv_conformal_metric, conformal_christoffel_first_kind,
                     shift_background);
  Xcts::longitudinal_operator(longitudinal_shift_background,
                              shift_background_strain, inv_conformal_metric);
}

}  // namespace Xcts::AnalyticData::detail
/// \endcond
