// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines tags related to domain quantities

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Tags.hpp"

namespace Tags {

template <typename SourceFrame, typename TargetFrame>
struct SurfaceJacobianDeterminant : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "SurfaceJacobianDeterminant()"; }
};

template <size_t Dim, typename SourceFrame, typename TargetFrame>
struct SurfaceJacobianDeterminantCompute
    : SurfaceJacobianDeterminant<SourceFrame, TargetFrame>,
      db::ComputeTag {
  using base = SurfaceJacobianDeterminant<SourceFrame, TargetFrame>;
  using argument_tags =
      tmpl::list<JacobianDeterminant<SourceFrame, TargetFrame>,
                 Magnitude<UnnormalizedFaceNormal<Dim, TargetFrame>>>;
  static Scalar<DataVector> function(
      const Scalar<DataVector>& jacobian_determinant,
      const Scalar<DataVector>& magnitude_of_face_normal) noexcept {
    return Scalar<DataVector>(get(magnitude_of_face_normal) *
                              get(jacobian_determinant));
  }
};

}  // namespace Tags
