// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/CreateFromOptions.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/Tags.hpp"
#include "tests/Unit/DataStructures/DataBox/TestHelpers.hpp"

namespace Elasticity {
namespace ConstitutiveRelations {

namespace {
struct Provider {
  using constitutive_relation_type = IsotropicHomogeneous<3>;
  static constitutive_relation_type constitutive_relation() { return {1., 2.}; }
};

struct ProviderOptionTag {
  using type = Provider;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Elasticity.ConstitutiveRelations.Tags",
                  "[Unit][Elasticity]") {
  TestHelpers::db::test_base_tag<
      Elasticity::Tags::ConstitutiveRelationBase<1>>(
      "ConstitutiveRelationBase");
  TestHelpers::db::test_simple_tag<
      Elasticity::Tags::ConstitutiveRelation<IsotropicHomogeneous<1>>>(
      "ConstitutiveRelation");
  static_assert(
      cpp17::is_same_v<
          db::const_item_type<Elasticity::Tags::ConstitutiveRelationBase<1>>,
          Elasticity::ConstitutiveRelations::ConstitutiveRelation<1>>,
      "Failed testing ConstitutiveRelationBase");
  static_assert(
      cpp17::is_same_v<
          db::const_item_type<Elasticity::Tags::ConstitutiveRelationBase<1>,
                              tmpl::list<Elasticity::Tags::ConstitutiveRelation<
                                  IsotropicHomogeneous<1>>>>,
          Elasticity::ConstitutiveRelations::IsotropicHomogeneous<1>>,
      "Failed testing ConstitutiveRelationBase");
  {
    INFO("ConstitutiveRelationFrom");
    // Fake some output of option-parsing
    const tuples::TaggedTuple<ProviderOptionTag> options{Provider{}};
    // Dispatch the `create_from_options` function call that constructs the
    // constitutive relation from the options.
    const auto constructed_data = Parallel::create_from_options<NoSuchType>(
        options,
        tmpl::list<
            Elasticity::Tags::ConstitutiveRelationFrom<ProviderOptionTag>>{});
    // Since the result is a tagged tuple we can't use base tags to retrieve it
    const auto& constructed_constitutive_relation = tuples::get<
        Elasticity::Tags::ConstitutiveRelationFrom<ProviderOptionTag>>(
        constructed_data);
    CHECK(constructed_constitutive_relation == IsotropicHomogeneous<3>{1., 2.});
  }
}

}  // namespace ConstitutiveRelations
}  // namespace Elasticity
