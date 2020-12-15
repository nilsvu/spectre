// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Parallel/CreateFromOptions.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/Tags.hpp"

namespace Elasticity::ConstitutiveRelations {

namespace {
struct Provider {
  using constitutive_relation_type = IsotropicHomogeneous<3>;
  static constitutive_relation_type constitutive_relation() { return {1., 2.}; }
};

struct ProviderOptionTag {
  using type = Provider;
};

struct Metavariables {
  using constitutive_relation_provider_option_tag = ProviderOptionTag;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Elasticity.ConstitutiveRelations.Tags",
                  "[Unit][Elasticity]") {
  TestHelpers::db::test_base_tag<Elasticity::Tags::ConstitutiveRelationBase>(
      "ConstitutiveRelationBase");
  TestHelpers::db::test_simple_tag<Elasticity::Tags::ConstitutiveRelation<
      ConstitutiveRelations::IsotropicHomogeneous<3>>>("ConstitutiveRelation");
}

}  // namespace Elasticity::ConstitutiveRelations
