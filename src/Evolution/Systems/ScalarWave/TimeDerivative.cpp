// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarWave/TimeDerivative.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace ScalarWave {
template <size_t Dim>
template <typename DataType>
evolution::dg::TimeDerivativeDecisions<Dim> TimeDerivative<Dim>::apply(
    const gsl::not_null<Scalar<DataType>*> dt_psi,
    const gsl::not_null<Scalar<DataType>*> dt_pi,
    const gsl::not_null<tnsr::i<DataType, Dim, Frame::Inertial>*> dt_phi,

    const gsl::not_null<Scalar<DataType>*> result_gamma2,

    const tnsr::i<DataType, Dim, Frame::Inertial>& d_psi,
    const tnsr::i<DataType, Dim, Frame::Inertial>& d_pi,
    const tnsr::ij<DataType, Dim, Frame::Inertial>& d_phi,

    const Scalar<DataType>& pi,
    const tnsr::i<DataType, Dim, Frame::Inertial>& phi,
    const Scalar<DataType>& gamma2) {
  // The constraint damping parameter gamma2 is needed for boundary corrections,
  // which means we need it as a temporary tag in order to project it to the
  // boundary. We prevent slicing/projecting directly from the volume to prevent
  // people from adding many compute tags to the DataBox, instead preferring
  // quantities be computed inside the TimeDerivative/Flux/Source structs. This
  // keeps related code together and makes figuring out where something is
  // computed a lot easier.
  *result_gamma2 = gamma2;

  get(*dt_psi) = -get(pi);
  get(*dt_pi) = -get<0, 0>(d_phi);
  for (size_t d = 1; d < Dim; ++d) {
    get(*dt_pi) -= d_phi.get(d, d);
  }
  for (size_t d = 0; d < Dim; ++d) {
    dt_phi->get(d) = -d_pi.get(d) + get(gamma2) * (d_psi.get(d) - phi.get(d));
  }
  return {true};
}

template class TimeDerivative<1>;
template class TimeDerivative<2>;
template class TimeDerivative<3>;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(_, data)                                                \
  template evolution::dg::TimeDerivativeDecisions<DIM(data)>                  \
  TimeDerivative<DIM(data)>::apply(                                           \
      const gsl::not_null<Scalar<DTYPE(data)>*>,                              \
      const gsl::not_null<Scalar<DTYPE(data)>*>,                              \
      const gsl::not_null<tnsr::i<DTYPE(data), DIM(data), Frame::Inertial>*>, \
      const gsl::not_null<Scalar<DTYPE(data)>*>,                              \
      const tnsr::i<DTYPE(data), DIM(data), Frame::Inertial>&,                \
      const tnsr::i<DTYPE(data), DIM(data), Frame::Inertial>&,                \
      const tnsr::ij<DTYPE(data), DIM(data), Frame::Inertial>&,               \
      const Scalar<DTYPE(data)>&,                                             \
      const tnsr::i<DTYPE(data), DIM(data), Frame::Inertial>&,                \
      const Scalar<DTYPE(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (double, DataVector))

#undef INSTANTIATION
#undef DIM
#undef DTYPE
}  // namespace ScalarWave
