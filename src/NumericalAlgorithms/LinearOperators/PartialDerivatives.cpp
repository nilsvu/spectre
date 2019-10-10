// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Mesh.hpp"
#include "Utilities/GenerateInstantiations.hpp"

template <size_t Dim, typename Symm, typename IndexList,
          typename DerivativeFrame, typename ResultTensor>
ResultTensor partial_derivatives(
    const Tensor<DataVector, Symm, IndexList>& tensor, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::Logical, DerivativeFrame>&
        inverse_jacobian) noexcept {
  using tensor_tag = Tags::TempTensor<0, Tensor<DataVector, Symm, IndexList>>;
  Variables<tmpl::list<tensor_tag>> temp_vars{mesh.number_of_grid_points()};
  get<tensor_tag>(temp_vars) = tensor;
  return get<Tags::deriv<tensor_tag, tmpl::size_t<Dim>, DerivativeFrame>>(
      partial_derivatives<tmpl::list<tensor_tag>>(temp_vars, mesh,
                                                  inverse_jacobian));
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FR(data) BOOST_PP_TUPLE_ELEM(1, data)
#define TNSR(data) BOOST_PP_TUPLE_ELEM(2, data)
#define INSTANTIATE(_, data)                                                   \
  template TensorMetafunctions::prepend_spatial_index<TNSR(data) < DataVector, \
                                                      DIM(data), FR(data)>,    \
      DIM(data), UpLo::Lo,                                                     \
      FR(data) >                                                               \
          partial_derivatives(                                                 \
              const TNSR(data) < DataVector, DIM(data), FR(data) > &,          \
              const Mesh<DIM(data)>&,                                          \
              const InverseJacobian<DataVector, DIM(data), Frame::Logical,     \
                                    FR(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Inertial),
                        (tnsr::i, tnsr::I))

#undef DIM
#undef FR
#undef TNSR
#undef INSTANTIATE
