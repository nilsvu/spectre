// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/SelfForce/Scalar/BoundaryConditions/Angular.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace ScalarSelfForce::BoundaryConditions {

Angular::Angular(int m_mode_number) : m_mode_number_(m_mode_number) {}

Angular::Angular(CkMigrateMessage* m) : Base(m) {}

void Angular::apply(
    const gsl::not_null<Scalar<ComplexDataVector>*> field,
    const gsl::not_null<Scalar<ComplexDataVector>*> n_dot_field_gradient,
    const tnsr::i<ComplexDataVector, 2>& /*deriv_field*/) const {
  // for (size_t i = 0; i < get(*field).size(); ++i) {
  //   if (not equal_within_roundoff(get(*n_dot_field_gradient)[i], 0.0)) {
  //     ERROR(
  //         "Angular boundary condition should only be applied at poles, "
  //         "where the flux through the boundary is zero, but it is "
  //         << get(*n_dot_field_gradient)[i]);
  //   }
  // }
}

void Angular::apply_linearized(
    const gsl::not_null<Scalar<ComplexDataVector>*> field_correction,
    const gsl::not_null<Scalar<ComplexDataVector>*>
        n_dot_field_gradient_correction,
    const tnsr::i<ComplexDataVector, 2>& deriv_field_correction) const {
  apply(field_correction, n_dot_field_gradient_correction,
        deriv_field_correction);
}

void Angular::pup(PUP::er& p) {
  Base::pup(p);
  p | m_mode_number_;
}

bool operator==(const Angular& lhs, const Angular& rhs) {
  return lhs.m_mode_number_ == rhs.m_mode_number_;
}

bool operator!=(const Angular& lhs, const Angular& rhs) {
  return not(lhs == rhs);
}

PUP::able::PUP_ID Angular::my_PUP_ID = 0;  // NOLINT

}  // namespace ScalarSelfForce::BoundaryConditions
