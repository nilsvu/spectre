// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarSelfForce {

/*!
 * \brief The first-order flux
 * $F^i=\{\partial_{r_\star}, \alpha\partial_{\cos\theta}\}\Psi_m$.
 */
void fluxes(gsl::not_null<tnsr::I<ComplexDataVector, 2>*> flux,
            const Scalar<ComplexDataVector>& alpha,
            const tnsr::i<ComplexDataVector, 2>& field_gradient);

/*!
 * \brief The first-order flux on an element face
 * $F^i=\{n_{r_\star}, \alpha n_{\cos\theta}\}\Psi_m$.
 */
void fluxes_on_face(gsl::not_null<tnsr::I<ComplexDataVector, 2>*> flux,
                    const Scalar<ComplexDataVector>& alpha,
                    const tnsr::I<DataVector, 2>& face_normal_vector,
                    const Scalar<ComplexDataVector>& field);

/*!
 * \brief The source term $\beta \Psi_m + \gamma_i F^i$.
 */
void add_sources(gsl::not_null<Scalar<ComplexDataVector>*> source,
                 const Scalar<ComplexDataVector>& beta,
                 const tnsr::i<ComplexDataVector, 2>& gamma,
                 const Scalar<ComplexDataVector>& field,
                 const tnsr::I<ComplexDataVector, 2>& flux);

struct Fluxes {
  using argument_tags = tmpl::list<Tags::Alpha>;
  using volume_tags = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  static constexpr bool is_trivial = false;
  static constexpr bool is_discontinuous = false;
  static void apply(gsl::not_null<tnsr::I<ComplexDataVector, 2>*> flux,
                    const Scalar<ComplexDataVector>& alpha,
                    const Scalar<ComplexDataVector>& /*field*/,
                    const tnsr::i<ComplexDataVector, 2>& field_gradient);
  static void apply(const gsl::not_null<tnsr::I<ComplexDataVector, 2>*> flux,
                    const Scalar<ComplexDataVector>& alpha,
                    const tnsr::i<DataVector, 2>& /*face_normal*/,
                    const tnsr::I<DataVector, 2>& face_normal_vector,
                    const Scalar<ComplexDataVector>& field);
};

struct Sources {
  using argument_tags = tmpl::list<Tags::Beta, Tags::Gamma>;
  static void apply(gsl::not_null<Scalar<ComplexDataVector>*> scalar_equation,
                    const Scalar<ComplexDataVector>& beta,
                    const tnsr::i<ComplexDataVector, 2>& gamma,
                    const Scalar<ComplexDataVector>& field,
                    const tnsr::I<ComplexDataVector, 2>& flux);
};

}  // namespace ScalarSelfForce
