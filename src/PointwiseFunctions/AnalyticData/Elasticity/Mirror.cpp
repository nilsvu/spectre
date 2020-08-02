// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/Elasticity/Mirror.hpp"

#include <cstddef>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Elasticity::AnalyticData {

Mirror::Mirror(constitutive_relation_type constitutive_relation) noexcept
    : constitutive_relation_(std::move(constitutive_relation)) {}

tuples::TaggedTuple<::Tags::FixedSource<Tags::Displacement<3>>>
Mirror::variables(
    const tnsr::I<DataVector, 3>& x,
    tmpl::list<::Tags::FixedSource<Tags::Displacement<3>>> /*meta*/) noexcept {
  return {make_with_value<tnsr::I<DataVector, 3>>(x, 0.)};
}

void Mirror::pup(PUP::er& p) noexcept { p | constitutive_relation_; }

bool operator==(const Mirror& lhs, const Mirror& rhs) noexcept {
  return lhs.constitutive_relation_ == rhs.constitutive_relation_;
}

bool operator!=(const Mirror& lhs, const Mirror& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace Elasticity::AnalyticData
