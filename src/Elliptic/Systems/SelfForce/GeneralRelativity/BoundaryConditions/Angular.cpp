// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/SelfForce/GeneralRelativity/BoundaryConditions/Angular.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace GrSelfForce::BoundaryConditions {

Angular::Angular(int m_mode_number) : m_mode_number_(m_mode_number) {}

void Angular::apply(
    const gsl::not_null<tnsr::aa<DataVector, 3>*> field_re,
    const gsl::not_null<tnsr::aa<DataVector, 3>*> field_im,
    const gsl::not_null<tnsr::aa<DataVector, 3>*> n_dot_field_gradient_re,
    const gsl::not_null<tnsr::aa<DataVector, 3>*> n_dot_field_gradient_im,
    const tnsr::iaa<DataVector, 3>& deriv_field_re,
    const tnsr::iaa<DataVector, 3>& deriv_field_im) const {
  // TODO
  // if (m_mode_number_ == 0) {
  //   get<0>(*n_dot_field_gradient) = 0.;
  //   get<1>(*n_dot_field_gradient) = 0.;
  // } else {
  //   get<0>(*field) = 0.;
  //   get<1>(*field) = 0.;
  // }
}

void Angular::apply_linearized(
    const gsl::not_null<tnsr::aa<DataVector, 3>*> field_correction_re,
    const gsl::not_null<tnsr::aa<DataVector, 3>*> field_correction_im,
    const gsl::not_null<tnsr::aa<DataVector, 3>*>
        n_dot_field_correction_gradient_re,
    const gsl::not_null<tnsr::aa<DataVector, 3>*>
        n_dot_field_correction_gradient_im,
    const tnsr::iaa<DataVector, 3>& deriv_field_correction_re,
    const tnsr::iaa<DataVector, 3>& deriv_field_correction_im) const {
  apply(field_correction_re, field_correction_im,
        n_dot_field_correction_gradient_re, n_dot_field_correction_gradient_im,
        deriv_field_correction_re, deriv_field_correction_im);
}

bool operator==(const Angular& lhs, const Angular& rhs) {
  return lhs.m_mode_number_ == rhs.m_mode_number_;
}

bool operator!=(const Angular& lhs, const Angular& rhs) {
  return not(lhs == rhs);
}

PUP::able::PUP_ID Angular::my_PUP_ID = 0;  // NOLINT

}  // namespace GrSelfForce::BoundaryConditions
