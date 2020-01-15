// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
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
  static std::string name() noexcept { return "Material"; }
  using type =
      std::unique_ptr<ConstitutiveRelations::ConstitutiveRelation<Dim>>;
};

/*!
 * \brief The elastic material's constitutive relation, loaded from options in
 * the `ProviderOptionTag`.
 *
 * Retrieves the constitutive relation from the object constructed from the
 * `ProviderOptionTag` by calling its `constitutive_relation()` member function.
 * Also requires the `ProviderOptionTag::type` to provide a `volume_dim` static
 * variable and a `constitutive_relation_type` type alias.
 *
 * The constitutive relation can be retrieved from the DataBox using its base
 * `Elasticity::Tags::ConstitutiveRelation` tag.
 */
template <typename ProviderOptionTag,
          typename ProviderType = tmpl::type_from<ProviderOptionTag>>
struct ConstitutiveRelationFrom
    : ConstitutiveRelation<ProviderType::volume_dim> {
  using base = ConstitutiveRelation<ProviderType::volume_dim>;
  using type = tmpl::type_from<base>;
  using option_tags = tmpl::list<ProviderOptionTag>;

  template <typename Metavariables>
  static type create_from_options(const ProviderType& provider) {
    return std::make_unique<typename ProviderType::constitutive_relation_type>(
        provider.constitutive_relation());
  }
};

}  // namespace Tags
}  // namespace Elasticity
