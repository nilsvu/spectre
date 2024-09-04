// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/SelfForce/GeneralRelativity/Equations.hpp"

#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"

namespace GrSelfForce {

void fluxes(const gsl::not_null<tnsr::Iaa<ComplexDataVector, 3>*> flux,
            const Scalar<ComplexDataVector>& alpha,
            const tnsr::iaa<ComplexDataVector, 3>& field_gradient) {
  for (size_t a = 0; a < 4; ++a) {
    for (size_t b = 0; b <= a; ++b) {
      flux->get(0, a, b) = field_gradient.get(0, a, b);
      flux->get(1, a, b) = get(alpha) * field_gradient.get(1, a, b);
      flux->get(2, a, b) = field_gradient.get(2, a, b);
    }
  }
}

void fluxes_on_face(const gsl::not_null<tnsr::Iaa<ComplexDataVector, 3>*> flux,
                    const Scalar<ComplexDataVector>& alpha,
                    const tnsr::I<DataVector, 3>& face_normal_vector,
                    const tnsr::aa<ComplexDataVector, 3>& field) {
  for (size_t a = 0; a < 4; ++a) {
    for (size_t b = 0; b <= a; ++b) {
      flux->get(0, a, b) = get<0>(face_normal_vector) * field.get(a, b);
      flux->get(1, a, b) =
          get(alpha) * get<1>(face_normal_vector) * field.get(a, b);
      flux->get(2, a, b) = get<2>(face_normal_vector) * field.get(a, b);
    }
  }
}

void add_sources(const gsl::not_null<tnsr::aa<ComplexDataVector, 3>*> source,
                 const tnsr::aaBB<ComplexDataVector, 3>& beta,
                 const tnsr::aaBB<ComplexDataVector, 3>& gamma_rstar,
                 const tnsr::aaBB<ComplexDataVector, 3>& gamma_theta,
                 const tnsr::aa<ComplexDataVector, 3>& field,
                 const tnsr::Iaa<ComplexDataVector, 3>& flux) {
  for (size_t a = 0; a < 4; ++a) {
    for (size_t b = 0; b <= a; ++b) {
      for (size_t c = 0; c < 4; ++c) {
        for (size_t d = 0; d <= c; ++d) {
          source->get(a, b) += beta.get(a, b, c, d) * field.get(c, d) +
                               gamma_rstar.get(a, b, c, d) * flux.get(0, c, d) +
                               gamma_theta.get(a, b, c, d) * flux.get(1, c, d);
        }
      }
    }
  }
}

void Fluxes::apply(const gsl::not_null<tnsr::Iaa<DataVector, 3>*> flux_re,
                   const gsl::not_null<tnsr::Iaa<DataVector, 3>*> flux_im,
                   const Scalar<ComplexDataVector>& alpha,
                   const tnsr::aa<DataVector, 3>& /*field_re*/,
                   const tnsr::aa<DataVector, 3>& /*field_im*/,
                   const tnsr::iaa<DataVector, 3>& field_gradient_re,
                   const tnsr::iaa<DataVector, 3>& field_gradient_im) {
  tnsr::Iaa<ComplexDataVector, 3> flux_complex{
      field_gradient_re.begin()->size()};
  tnsr::iaa<ComplexDataVector, 3> field_gradient_complex{
      field_gradient_re.begin()->size()};
  for (size_t i = 0; i < field_gradient_re.size(); ++i) {
    field_gradient_complex[i] =
        field_gradient_re[i] +
        std::complex<double>(0.0, 1.0) * field_gradient_im[i];
  }
  fluxes(make_not_null(&flux_complex), alpha, field_gradient_complex);
  for (size_t i = 0; i < flux_complex.size(); ++i) {
    (*flux_re)[i] = real(flux_complex[i]);
    (*flux_im)[i] = imag(flux_complex[i]);
  }
}

void Fluxes::apply(const gsl::not_null<tnsr::Iaa<DataVector, 3>*> flux_re,
                   const gsl::not_null<tnsr::Iaa<DataVector, 3>*> flux_im,
                   const Scalar<ComplexDataVector>& alpha,
                   const tnsr::i<DataVector, 3>& /*face_normal*/,
                   const tnsr::I<DataVector, 3>& face_normal_vector,
                   const tnsr::aa<DataVector, 3>& field_re,
                   const tnsr::aa<DataVector, 3>& field_im) {
  tnsr::Iaa<ComplexDataVector, 3> flux_complex{field_re.begin()->size()};
  tnsr::aa<ComplexDataVector, 3> field_complex{field_re.begin()->size()};
  for (size_t i = 0; i < field_re.size(); ++i) {
    field_complex[i] =
        field_re[i] + std::complex<double>(0.0, 1.0) * field_im[i];
  }
  fluxes_on_face(make_not_null(&flux_complex), alpha, face_normal_vector,
                 field_complex);
  for (size_t i = 0; i < flux_complex.size(); ++i) {
    (*flux_re)[i] = real(flux_complex[i]);
    (*flux_im)[i] = imag(flux_complex[i]);
  }
}

void Sources::apply(const gsl::not_null<tnsr::aa<DataVector, 3>*> sources_re,
                    const gsl::not_null<tnsr::aa<DataVector, 3>*> sources_im,
                    const tnsr::aaBB<ComplexDataVector, 3>& beta,
                    const tnsr::aaBB<ComplexDataVector, 3>& gamma_rstar,
                    const tnsr::aaBB<ComplexDataVector, 3>& gamma_theta,
                    const tnsr::aa<DataVector, 3>& field_re,
                    const tnsr::aa<DataVector, 3>& field_im,
                    const tnsr::Iaa<DataVector, 3>& flux_re,
                    const tnsr::Iaa<DataVector, 3>& flux_im) {
  tnsr::aa<ComplexDataVector, 3> sources_complex{field_re.begin()->size()};
  tnsr::aa<ComplexDataVector, 3> field_complex{field_re.begin()->size()};
  for (size_t i = 0; i < field_re.size(); ++i) {
    sources_complex[i] =
        (*sources_re)[i] + std::complex<double>(0.0, 1.0) * (*sources_im)[i];
    field_complex[i] =
        field_re[i] + std::complex<double>(0.0, 1.0) * field_im[i];
  }
  tnsr::Iaa<ComplexDataVector, 3> flux_complex{flux_re.begin()->size()};
  for (size_t i = 0; i < flux_re.size(); ++i) {
    flux_complex[i] = flux_re[i] + std::complex<double>(0.0, 1.0) * flux_im[i];
  }
  add_sources(make_not_null(&sources_complex), beta, gamma_rstar, gamma_theta,
              field_complex, flux_complex);
  for (size_t i = 0; i < sources_complex.size(); ++i) {
    (*sources_re)[i] = real(sources_complex[i]);
    (*sources_im)[i] = imag(sources_complex[i]);
  }
}

}  // namespace GrSelfForce
