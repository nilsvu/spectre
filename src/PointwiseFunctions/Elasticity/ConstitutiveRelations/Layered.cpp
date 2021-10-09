// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/Layered.hpp"

#include <array>
#include <memory>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/ConstitutiveRelation.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Elasticity::ConstitutiveRelations {

void Layered::stress(const gsl::not_null<tnsr::II<DataVector, 3>*> stress,
                     const tnsr::ii<DataVector, 3>& strain,
                     const tnsr::I<DataVector, 3>& x) const {
  const auto& z = get<2>(x);
  tnsr::II<DataVector, 3> stress_i{1_st};
  tnsr::ii<DataVector, 3> strain_i{1_st};
  tnsr::I<DataVector, 3> x_i{1_st};
  for (size_t m_i = 0; m_i < layer_boundaries_.size() + 1; ++m_i) {
    const auto& material = *(materials_[m_i]);
    const double lower_bound = m_i == 0
                                   ? -std::numeric_limits<double>::infinity()
                                   : layer_boundaries_[m_i - 1];
    const double upper_bound = m_i == layer_boundaries_.size()
                                   ? std::numeric_limits<double>::infinity()
                                   : layer_boundaries_[m_i];
    for (size_t i = 0; i < z.size(); ++i) {
      if (z[i] > lower_bound and z[i] <= upper_bound) {
        for (size_t k = 0; k < Dim; ++k) {
          for (size_t l = 0; l <= k; ++l) {
            strain_i.get(k, l)[0] = strain.get(k, l)[i];
          }
          x_i.get(k)[0] = x.get(k)[i];
        }
        material.stress(make_not_null(&stress_i), strain_i, x_i);
        for (size_t k = 0; k < Dim; ++k) {
          for (size_t l = 0; l <= k; ++l) {
            stress->get(k, l)[i] = stress_i.get(k, l)[0];
          }
        }
      }
    }
  }
}

PUP::able::PUP_ID Layered::my_PUP_ID = 0;
}  // namespace Elasticity::ConstitutiveRelations
