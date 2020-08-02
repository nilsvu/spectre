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

Mirror::Mirror(const double beam_width) noexcept : beam_width_(beam_width) {}

tuples::TaggedTuple<Tags::MinusNormalDotStress<3>> Mirror::boundary_variables(
    const tnsr::I<DataVector, 3>& x, const Direction<3>& direction,
    const tnsr::i<DataVector, 3>& /*face_normal*/,
    tmpl::list<Tags::MinusNormalDotStress<3>> /*meta*/) const noexcept {
  if (direction == Direction<3>::lower_zeta()) {
    auto minus_n_dot_stress = make_with_value<tnsr::I<DataVector, 3>>(x, 0.);
    const DataVector r = get(magnitude(x));
    // Normal is (0, 0, -1)
    get<2>(minus_n_dot_stress) =
        exp(-square(r) / square(beam_width_)) / M_PI / square(beam_width_);
    return {std::move(minus_n_dot_stress)};
  } else {
    return {make_with_value<tnsr::I<DataVector, 3>>(x, 0.)};
  }
}

tuples::TaggedTuple<Tags::Displacement<3>> Mirror::variables(
    const tnsr::I<DataVector, 3>& x,
    tmpl::list<Tags::Displacement<3>> /*meta*/) const noexcept {
  for (size_t i = 0; i < get<2>(x).size(); ++i) {
    ASSERT(get<2>(x)[i] > 0,
           "Displacement field is only available on a boundary at z > 0");
  }
  return {make_with_value<tnsr::I<DataVector, 3>>(x, 0.)};
}

tuples::TaggedTuple<::Tags::FixedSource<Tags::Displacement<3>>>
Mirror::variables(
    const tnsr::I<DataVector, 3>& x,
    tmpl::list<::Tags::FixedSource<Tags::Displacement<3>>> /*meta*/) noexcept {
  return {make_with_value<tnsr::I<DataVector, 3>>(x, 0.)};
}

void Mirror::pup(PUP::er& p) noexcept { p | beam_width_; }

bool operator==(const Mirror& lhs, const Mirror& rhs) noexcept {
  return lhs.beam_width_ == rhs.beam_width_;
}

bool operator!=(const Mirror& lhs, const Mirror& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace Elasticity::AnalyticData
