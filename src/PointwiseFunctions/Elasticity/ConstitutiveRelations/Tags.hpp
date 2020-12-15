// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/ConstitutiveRelation.hpp"

namespace Elasticity {
namespace Tags {

/// Base tag for the constitutive relation
struct ConstitutiveRelationBase : db::BaseTag {};

/*!
 * \brief The elastic material's constitutive relation.
 *
 * When constructing from options, copies the constitutive relation from the
 * `Metavariables::constitutive_relation_provider_option_tag` by calling its
 * constructed object's `constitutive_relation()` member function.
 *
 * The constitutive relation can be retrieved from the DataBox using its base
 * `Elasticity::Tags::ConstitutiveRelation` tag.
 *
 * \see `Elasticity::ConstitutiveRelations::ConstitutiveRelation`
 */
template <typename ConstitutiveRelationType>
struct ConstitutiveRelation : ConstitutiveRelationBase, db::SimpleTag {
  using type = ConstitutiveRelationType;
};

template <size_t Dim>
struct ConstitutiveRelationReference : ConstitutiveRelationBase, db::SimpleTag {
  using type = const ConstitutiveRelations::ConstitutiveRelation<Dim>&;
};

template <size_t Dim, typename ProviderTag>
struct ConstitutiveRelationCompute : ConstitutiveRelationReference<Dim>,
                                     db::ComputeTag {
  using base = ConstitutiveRelationReference<Dim>;
  using argument_tags = tmpl::list<ProviderTag>;

  template <typename Provider>
  static typename base::type function(const Provider& provider) noexcept {
    return provider.constitutive_relation();
  }
};

}  // namespace Tags
}  // namespace Elasticity
