// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Xcts/BoundaryConditions/Flatness.hpp"

#include <algorithm>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace Xcts::BoundaryConditions::detail {

void FlatnessImpl::apply(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor,
    const gsl::not_null<Scalar<DataVector>*>
    /*n_dot_conformal_factor_gradient*/) noexcept {
  get(*conformal_factor) = 1.;
}

void FlatnessImpl::apply(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor,
    const gsl::not_null<Scalar<DataVector>*> lapse_times_conformal_factor,
    const gsl::not_null<Scalar<DataVector>*>
    /*n_dot_conformal_factor_gradient*/,
    const gsl::not_null<Scalar<DataVector>*>
    /*n_dot_lapse_times_conformal_factor_gradient*/) noexcept {
  get(*conformal_factor) = 1.;
  get(*lapse_times_conformal_factor) = 1.;
}

void FlatnessImpl::apply(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor,
    const gsl::not_null<Scalar<DataVector>*> lapse_times_conformal_factor,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess) noexcept {
  get(*conformal_factor) = 1.;
  get(*lapse_times_conformal_factor) = 1.;
  std::fill(shift_excess->begin(), shift_excess->end(), 0.);
}

void FlatnessImpl::apply_linearized(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
    const gsl::not_null<Scalar<DataVector>*>
    /*n_dot_conformal_factor_gradient_correction*/) noexcept {
  get(*conformal_factor_correction) = 0.;
}

void FlatnessImpl::apply_linearized(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
    const gsl::not_null<Scalar<DataVector>*>
        lapse_times_conformal_factor_correction,
    const gsl::not_null<Scalar<DataVector>*>
    /*n_dot_conformal_factor_gradient_correction*/,
    const gsl::not_null<Scalar<DataVector>*>
    /*n_dot_lapse_times_conformal_factor_gradient_correction*/) noexcept {
  get(*conformal_factor_correction) = 0.;
  get(*lapse_times_conformal_factor_correction) = 0.;
}

void FlatnessImpl::apply_linearized(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
    const gsl::not_null<Scalar<DataVector>*>
        lapse_times_conformal_factor_correction,
    const gsl::not_null<tnsr::I<DataVector, 3>*>
        shift_excess_correction) noexcept {
  get(*conformal_factor_correction) = 0.;
  get(*lapse_times_conformal_factor_correction) = 0.;
  std::fill(shift_excess_correction->begin(), shift_excess_correction->end(),
            0.);
}

bool operator==(const FlatnessImpl& /*lhs*/,
                const FlatnessImpl& /*rhs*/) noexcept {
  return true;
}

bool operator!=(const FlatnessImpl& lhs, const FlatnessImpl& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace Xcts::BoundaryConditions::detail
