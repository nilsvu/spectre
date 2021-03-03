// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Elasticity/BoundaryConditions/CoupledPoissonTest.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/IndexToSliceAt.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace Elasticity::BoundaryConditions::detail {

template <size_t Dim>
void CoupledPoissonTestImpl<Dim>::apply(
    const gsl::not_null<tnsr::I<DataVector, Dim>*> displacement,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> n_dot_minus_stress,
    const Direction<Dim>& direction, const tnsr::I<DataVector, Dim>& x,
    const tnsr::i<DataVector, Dim>& face_normal, const Mesh<Dim>& mesh,
    const std::optional<std::reference_wrapper<const Variables<db::wrap_tags_in<
        ::Tags::Analytic,
        tmpl::list<Tags::Displacement<Dim>, Tags::MinusStress<Dim>>>>>>
        analytic_solutions) const noexcept {
  const size_t slice_index = index_to_slice_at(mesh.extents(), direction);
  const auto analytic_solutions_on_face =
      data_on_slice(analytic_solutions->get(), mesh.extents(),
                    direction.dimension(), slice_index);
  if (direction == Direction<Dim>::upper_xi()) {
    get<0>(*displacement) = -get<1>(*displacement);
    get<1>(*displacement) =
        get<1>(get<::Tags::Analytic<Tags::Displacement<Dim>>>(
            analytic_solutions_on_face));
  } else if (direction == Direction<Dim>::upper_eta()) {
    normal_dot_flux(n_dot_minus_stress, face_normal,
                    get<::Tags::Analytic<Tags::MinusStress<Dim>>>(
                        analytic_solutions_on_face));
  } else {
    *displacement = get<::Tags::Analytic<Tags::Displacement<Dim>>>(
        analytic_solutions_on_face);
  }
}

template <size_t Dim>
void CoupledPoissonTestImpl<Dim>::apply_linearized(
    const gsl::not_null<tnsr::I<DataVector, Dim>*> displacement_correction,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> n_dot_minus_stress_correction,
    const Direction<Dim>& direction) noexcept {
  if (direction == Direction<Dim>::upper_xi()) {
    get<0>(*displacement_correction) = -get<1>(*displacement_correction);
    get<1>(*displacement_correction) = 0.;
  } else if (direction == Direction<Dim>::upper_eta()) {
    std::fill(n_dot_minus_stress_correction->begin(),
              n_dot_minus_stress_correction->end(), 0.);
  } else {
    std::fill(displacement_correction->begin(), displacement_correction->end(),
              0.);
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data) template class CoupledPoissonTestImpl<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (2))

#undef DIM
#undef INSTANTIATE

}  // namespace Elasticity::BoundaryConditions::detail
