// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class ComplexDataVector;
class DataVector;
/// \endcond

namespace GrSelfForce::Tags {

struct MModeRe : db::SimpleTag {
  using type = tnsr::aa<DataVector, 3>;
};

struct MModeIm : db::SimpleTag {
  using type = tnsr::aa<DataVector, 3>;
};

struct Alpha : db::SimpleTag {
  using type = Scalar<ComplexDataVector>;
};

struct Beta : db::SimpleTag {
  using type = tnsr::aaBB<ComplexDataVector, 3>;
};

struct GammaRstar : db::SimpleTag {
  using type = tnsr::aaBB<ComplexDataVector, 3>;
};

struct GammaTheta : db::SimpleTag {
  using type = tnsr::aaBB<ComplexDataVector, 3>;
};

struct FieldIsRegularized : db::SimpleTag {
  using type = bool;
};

struct SingularField : db::SimpleTag {
  using type = tnsr::aa<ComplexDataVector, 3>;
};

struct BoyerLindquistRadius : db::SimpleTag {
  using type = Scalar<DataVector>;
};

}  // namespace ScalarSelfForce::Tags
