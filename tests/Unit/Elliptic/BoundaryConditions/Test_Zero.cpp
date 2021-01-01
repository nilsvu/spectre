// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Elliptic/BoundaryConditions/Zero.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic::BoundaryConditions {

namespace {
struct ScalarFieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.BoundaryConditions.Zero", "[Unit][Elliptic]") {
  using BoundaryConditionType =
      BoundaryCondition<1, tmpl::list<elliptic::BoundaryConditions::Registrars::
                                          Zero<1, tmpl::list<ScalarFieldTag>>>>;
  Parallel::register_derived_classes_with_charm<BoundaryConditionType>();

  {
    INFO("Test concrete derived class");
    elliptic::BoundaryConditions::Zero<1, tmpl::list<ScalarFieldTag>>
        boundary_condition{elliptic::BoundaryConditionType::Dirichlet};
    CHECK(get<elliptic::Tags::BoundaryConditionType<ScalarFieldTag>>(
              boundary_condition.boundary_condition_types()) ==
          elliptic::BoundaryConditionType::Dirichlet);
    const auto serialized_and_deserialized_boundary_condition =
        serialize_and_deserialize(boundary_condition);
    CHECK(get<elliptic::Tags::BoundaryConditionType<ScalarFieldTag>>(
              serialized_and_deserialized_boundary_condition
                  .boundary_condition_types()) ==
          elliptic::BoundaryConditionType::Dirichlet);
  }

  const auto created_dirichlet_boundary_condition =
      TestHelpers::test_factory_creation<BoundaryConditionType>(
          "Zero:\n"
          "  ScalarFieldTag: Dirichlet");
  const auto dirichlet_boundary_condition =
      serialize_and_deserialize(created_dirichlet_boundary_condition);

  const auto created_neumann_boundary_condition =
      TestHelpers::test_factory_creation<BoundaryConditionType>(
          "Zero:\n"
          "  ScalarFieldTag: Neumann");
  const auto neumann_boundary_condition =
      serialize_and_deserialize(created_neumann_boundary_condition);

  const auto box = db::create<db::AddSimpleTags<>>();
  {
    INFO("Dirichlet");
    Scalar<DataVector> dirichlet_field{size_t{1}, 1.};
    Scalar<DataVector> neumann_field{size_t{1}, 1.};
    (*dirichlet_boundary_condition)(box, Direction<1>::lower_xi(),
                                    make_not_null(&dirichlet_field),
                                    make_not_null(&neumann_field));
    CHECK_ITERABLE_APPROX(get(dirichlet_field), DataVector{0.});
    CHECK_ITERABLE_APPROX(get(neumann_field), DataVector{1.});
  }
  {
    INFO("Neumann");
    Scalar<DataVector> dirichlet_field{size_t{1}, 1.};
    Scalar<DataVector> neumann_field{size_t{1}, 1.};
    (*neumann_boundary_condition)(box, Direction<1>::lower_xi(),
                                  make_not_null(&dirichlet_field),
                                  make_not_null(&neumann_field));
    CHECK_ITERABLE_APPROX(get(dirichlet_field), DataVector{1.});
    CHECK_ITERABLE_APPROX(get(neumann_field), DataVector{0.});
  }
  {
    INFO("Linearization");
    const auto linearized_boundary_condition =
        dirichlet_boundary_condition->linearization();
    Scalar<DataVector> dirichlet_field{size_t{1}, 1.};
    Scalar<DataVector> neumann_field{size_t{1}, 1.};
    (*linearized_boundary_condition)(box, Direction<1>::lower_xi(),
                                     make_not_null(&dirichlet_field),
                                     make_not_null(&neumann_field));
    CHECK_ITERABLE_APPROX(get(dirichlet_field), DataVector{0.});
    CHECK_ITERABLE_APPROX(get(neumann_field), DataVector{1.});
  }
}

}  // namespace elliptic::BoundaryConditions
