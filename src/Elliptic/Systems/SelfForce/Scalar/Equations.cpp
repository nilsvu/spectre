// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/SelfForce/Scalar/Equations.hpp"

#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"

namespace ScalarSelfForce {

void fluxes(const gsl::not_null<tnsr::I<ComplexDataVector, 2>*> flux,
            const Scalar<ComplexDataVector>& alpha,
            const tnsr::i<ComplexDataVector, 2>& field_gradient) {
  get<0>(*flux) = get<0>(field_gradient);
  get<1>(*flux) = get(alpha) * get<1>(field_gradient);
}

void fluxes_on_face(const gsl::not_null<tnsr::I<ComplexDataVector, 2>*> flux,
                    const Scalar<ComplexDataVector>& alpha,
                    const tnsr::I<DataVector, 2>& face_normal_vector,
                    const Scalar<ComplexDataVector>& field) {
  get<0>(*flux) = get<0>(face_normal_vector) * get(field);
  get<1>(*flux) = get(alpha) * get<1>(face_normal_vector) * get(field);
}

void add_sources(const gsl::not_null<Scalar<ComplexDataVector>*> source,
                 const Scalar<ComplexDataVector>& beta,
                 const tnsr::i<ComplexDataVector, 2>& gamma,
                 const Scalar<ComplexDataVector>& field,
                 const tnsr::I<ComplexDataVector, 2>& flux) {
  get(*source) += get(beta) * get(field) + get<0>(gamma) * get<0>(flux) +
                  get<1>(gamma) * get<1>(flux);
}

void Fluxes::apply(const gsl::not_null<tnsr::I<ComplexDataVector, 2>*> flux,
                   const Scalar<ComplexDataVector>& alpha,
                   const Scalar<ComplexDataVector>& /*field*/,
                   const tnsr::i<ComplexDataVector, 2>& field_gradient) {
  fluxes(flux, alpha, field_gradient);
}

void Fluxes::apply(const gsl::not_null<tnsr::I<ComplexDataVector, 2>*> flux,
                   const Scalar<ComplexDataVector>& alpha,
                   const tnsr::i<DataVector, 2>& /*face_normal*/,
                   const tnsr::I<DataVector, 2>& face_normal_vector,
                   const Scalar<ComplexDataVector>& field) {
  fluxes_on_face(flux, alpha, face_normal_vector, field);
}

void Sources::apply(
    const gsl::not_null<Scalar<ComplexDataVector>*> scalar_equation,
    const Scalar<ComplexDataVector>& beta,
    const tnsr::i<ComplexDataVector, 2>& gamma,
    const Scalar<ComplexDataVector>& field,
    const tnsr::I<ComplexDataVector, 2>& flux) {
  add_sources(scalar_equation, beta, gamma, field, flux);
}

}  // namespace ScalarSelfForce
