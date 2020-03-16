// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/ConstitutiveRelation.hpp"
#include "Utilities/TMPL.hpp"

namespace Elasticity {
namespace Tags {

/// Base tag for the constitutive relation
template <size_t Dim>
struct ConstitutiveRelationBase : db::BaseTag {
  using type = Elasticity::ConstitutiveRelations::ConstitutiveRelation<Dim>;
};

/*!
 * \brief The elastic material's constitutive relation.
 *
 * \see `Elasticity::ConstitutiveRelations::ConstitutiveRelation`
 */
template <typename ConstitutiveRelationType>
struct ConstitutiveRelation
    : ConstitutiveRelationBase<ConstitutiveRelationType::volume_dim>,
      db::SimpleTag {
  using type = ConstitutiveRelationType;
};

/*!
 * \brief The elastic material's constitutive relation, loaded from options in
 * the `ProviderOptionTag`.
 *
 * Retrieves the constitutive relation from the object constructed from the
 * `ProviderOptionTag` by calling its `constitutive_relation()` member function.
 * Also requires the `ProviderOptionTag::type` to provide a
 * `constitutive_relation_type` type alias.
 *
 * The constitutive relation can be retrieved from the DataBox using its base
 * `Elasticity::Tags::ConstitutiveRelation` tag.
 */
template <typename ProviderOptionTag>
struct ConstitutiveRelationFrom
    : ConstitutiveRelation<typename tmpl::type_from<
          ProviderOptionTag>::constitutive_relation_type> {
  using ProviderType = tmpl::type_from<ProviderOptionTag>;
  using base =
      ConstitutiveRelation<typename ProviderType::constitutive_relation_type>;
  using type = tmpl::type_from<base>;
  using option_tags = tmpl::list<ProviderOptionTag>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const ProviderType& provider) {
    return provider.constitutive_relation();
  }
};

}  // namespace Tags
}  // namespace Elasticity
