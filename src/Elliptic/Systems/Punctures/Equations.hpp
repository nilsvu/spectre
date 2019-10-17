// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Elliptic/Systems/Punctures/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace Punctures {

void sources(const gsl::not_null<Scalar<DataVector>*> source_for_field,
             const Scalar<DataVector>& alpha, const Scalar<DataVector>& beta,
             const Scalar<DataVector>& field) noexcept;

void linearized_sources(
    const gsl::not_null<Scalar<DataVector>*> source_for_field_correction,
    const Scalar<DataVector>& field, const Scalar<DataVector>& alpha,
    const Scalar<DataVector>& beta,
    const Scalar<DataVector>& field_correction) noexcept;

struct Sources {
  using argument_tags = tmpl::list<Tags::Alpha, Tags::Beta>;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*> source_for_field,
      const Scalar<DataVector>& alpha, const Scalar<DataVector>& beta,
      const Scalar<DataVector>& field,
      const tnsr::i<DataVector, 3,
                    Frame::Inertial>& /*field_gradient*/) noexcept {
    sources(source_for_field, alpha, beta, field);
  }
};

struct LinearizedSources {
  using argument_tags = tmpl::list<Tags::Field, Tags::Alpha, Tags::Beta>;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*> source_for_field_correction,
      const Scalar<DataVector>& field, const Scalar<DataVector>& alpha,
      const Scalar<DataVector>& beta,
      const Scalar<DataVector>& field_correction,
      const tnsr::i<DataVector, 3,
                    Frame::Inertial>& /*field_gradient_correction*/) noexcept {
    linearized_sources(source_for_field_correction, field, alpha, beta,
                       field_correction);
  }
};

}  // namespace Punctures
