// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/Poisson/ProductOfSinusoids.hpp"

#include <cmath>
#include <pup.h>  // IWYU pragma: keep

#include "DataStructures/DataVector.hpp"     // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdArrayHelpers.hpp"
// IWYU pragma: no_forward_declare Tensor

namespace Poisson {
namespace Solutions {
namespace ProductOfSinusoids_detail {

template <size_t Dim>
Scalar<DataVector> variable(
    Tags::Field /* meta */, const tnsr::I<DataVector, Dim>& x,
    const std::array<double, Dim>& wave_numbers) noexcept {
  auto field = make_with_value<Scalar<DataVector>>(x, 1.);
  for (size_t d = 0; d < Dim; d++) {
    field.get() *= sin(gsl::at(wave_numbers, d) * x.get(d));
  }
  return field;
}

template <size_t Dim>
tnsr::i<DataVector, Dim> variable(
    ::Tags::deriv<Tags::Field, tmpl::size_t<Dim>, Frame::Inertial> /* meta */,
    const tnsr::I<DataVector, Dim>& x,
    const std::array<double, Dim>& wave_numbers) noexcept {
  auto field_gradient =
      make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(x, 1.);
  for (size_t d = 0; d < Dim; d++) {
    field_gradient.get(d) *=
        gsl::at(wave_numbers, d) * cos(gsl::at(wave_numbers, d) * x.get(d));
    for (size_t other_d = 0; other_d < Dim; other_d++) {
      if (other_d != d) {
        field_gradient.get(d) *=
            sin(gsl::at(wave_numbers, other_d) * x.get(other_d));
      }
    }
  }
  return field_gradient;
}

template <size_t Dim>
Scalar<DataVector> variable(
    ::Tags::FixedSource<Tags::Field> /* meta */,
    const tnsr::I<DataVector, Dim>& x,
    const std::array<double, Dim>& wave_numbers) noexcept {
  auto field_source = variable(Tags::Field{}, x, wave_numbers);
  field_source.get() *= square(magnitude(wave_numbers));
  return field_source;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define TAG(data) BOOST_PP_TUPLE_ELEM(1, data)

template <size_t Dim>
using FieldDeriv =
    ::Tags::deriv<Tags::Field, tmpl::size_t<Dim>, Frame::Inertial>;

#define INSTANTIATE_SCALARS(_, data)                                 \
  template db::item_type<TAG(data)> variable(                        \
      TAG(data) /* meta */, const tnsr::I<DataVector, DIM(data)>& x, \
      const std::array<double, DIM(data)>& wave_numbers) noexcept;

#define INSTANTIATE_TENSORS(_, data)                    \
  template db::item_type<TAG(data) < DIM(data)>>        \
      variable(TAG(data) < DIM(data) > /* meta */,      \
               const tnsr::I<DataVector, DIM(data)>& x, \
               const std::array<double, DIM(data)>& wave_numbers) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_SCALARS, (1, 2, 3),
                        (Tags::Field, ::Tags::FixedSource<Tags::Field>))
GENERATE_INSTANTIATIONS(INSTANTIATE_TENSORS, (1, 2, 3), (FieldDeriv))

#undef DIM
#undef TAG
#undef INSTANTIATE_SCALARS
#undef INSTANTIATE_TENSORS

}  // namespace ProductOfSinusoids_detail
}  // namespace Solutions
}  // namespace Poisson
