// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class DataVector;
/// \endcond

namespace Punctures {
/// Tags related to the puncture equation
namespace Tags {

/*!
 * \brief The puncture field $u(x)$ to solve for
 *
 * \see Punctures
 */
struct Field : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/*!
 * \brief The source field $\alpha(x)$
 *
 * \see Punctures
 */
struct Alpha : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/*!
 * \brief The source field $\beta(x)$
 *
 * \see Punctures
 */
struct Beta : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/*!
 * \brief The traceless conformal extrinsic curvature $\bar{A}_{ij}$
 *
 * \see Punctures
 */
struct TracelessConformalExtrinsicCurvature : db::SimpleTag {
  using type = tnsr::II<DataVector, 3>;
};

/// @{
/*!
 * \brief The conformal factor minus one, $\psi - 1 = u + \frac{1}{\alpha}$
 *
 * \see Punctures
 */
struct ConformalFactorMinusOne : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct ConformalFactorMinusOneCompute : ConformalFactorMinusOne,
                                        db::ComputeTag {
  using base = ConformalFactorMinusOne;
  using return_type = Scalar<DataVector>;
  using argument_tags = tmpl::list<Field, Alpha>;

  static void function(const gsl::not_null<Scalar<DataVector>*> result,
                       const Scalar<DataVector>& field,
                       const Scalar<DataVector>& alpha) {
    get(*result) = get(field) + 1. / get(alpha);
  }
};
/// @}

}  // namespace Tags
}  // namespace Punctures
