// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/ConstitutiveRelation.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/Tags.hpp"

namespace Elasticity {
namespace ConstitutiveRelations {

namespace {
struct Provider {
  static constexpr size_t volume_dim = 1;
  using constitutive_relation_type = IsotropicHomogeneous<volume_dim>;
  static constitutive_relation_type constitutive_relation() { return {1., 1.}; }
};

struct ProviderOptionTag {
  using type = Provider;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Elasticity.Tags", "[Unit][Elasticity]") {
  CHECK(db::tag_name<Elasticity::Tags::ConstitutiveRelation<1>>() ==
        "Material");

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
    // Since the result is a tagged tuple we can't use base tags to retrieve it.
    // Also, `tuples::get` doesn't automatically dereference the unique_ptr.
    const auto& constructed_constitutive_relation = tuples::get<
        Elasticity::Tags::ConstitutiveRelationFrom<ProviderOptionTag>>(
        constructed_data);
    CHECK(dynamic_cast<IsotropicHomogeneous<1>&>(
              *constructed_constitutive_relation) ==
          IsotropicHomogeneous<1>{1., 1.});
  }
}

}  // namespace ConstitutiveRelations
}  // namespace Elasticity
