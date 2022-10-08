// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>

#include "DataStructures/DataBox/Tag.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/ConstitutiveRelation.hpp"
#include "Utilities/TMPL.hpp"

namespace Elasticity {
namespace Tags {

/*!
 * \brief The elastic material's constitutive relation.
 *
 * \see `Elasticity::ConstitutiveRelations::ConstitutiveRelation`
 */
template <size_t Dim>
struct ConstitutiveRelation : db::SimpleTag {
  using type =
      std::unique_ptr<ConstitutiveRelations::ConstitutiveRelation<Dim>>;
};

}  // namespace Tags
}  // namespace Elasticity
