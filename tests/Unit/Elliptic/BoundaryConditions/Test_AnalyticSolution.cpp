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
#include "Elliptic/BoundaryConditions/AnalyticSolution.hpp"
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

SPECTRE_TEST_CASE("Unit.Elliptic.BoundaryConditions.AnalyticSolution",
                  "[Unit][Elliptic]") {
  using BoundaryConditionType = BoundaryCondition<
      1, tmpl::list<elliptic::BoundaryConditions::Registrars::AnalyticSolution<
             1, tmpl::list<ScalarFieldTag>>>>;
  Parallel::register_derived_classes_with_charm<BoundaryConditionType>();

  {
    INFO("Test concrete derived class");
    elliptic::BoundaryConditions::AnalyticSolution<1,
                                                   tmpl::list<ScalarFieldTag>>
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
          "AnalyticSolution:\n"
          "  ScalarFieldTag: Dirichlet");
  const auto dirichlet_boundary_condition =
      serialize_and_deserialize(created_dirichlet_boundary_condition);

  const auto created_neumann_boundary_condition =
      TestHelpers::test_factory_creation<BoundaryConditionType>(
          "AnalyticSolution:\n"
          "  ScalarFieldTag: Neumann");
  const auto neumann_boundary_condition =
      serialize_and_deserialize(created_neumann_boundary_condition);

  Mesh<1> mesh{3, Spectral::Basis::Legendre,
               Spectral::Quadrature::GaussLobatto};
  Variables<db::wrap_tags_in<
      ::Tags::Analytic,
      tmpl::list<ScalarFieldTag, ::Tags::Flux<ScalarFieldTag, tmpl::size_t<1>,
                                              Frame::Inertial>>>>
      analytic_solutions{mesh.number_of_grid_points()};
  get(get<::Tags::Analytic<ScalarFieldTag>>(analytic_solutions)) =
      DataVector{1., 2., 3.};
  get<0>(get<::Tags::Analytic<
             ::Tags::Flux<ScalarFieldTag, tmpl::size_t<1>, Frame::Inertial>>>(
      analytic_solutions)) = DataVector{4., 5., 6.};
  auto direction = Direction<1>::upper_xi();
  tnsr::i<DataVector, 1> face_normal{DataVector{2.}};
  auto box = db::create<db::AddSimpleTags<
      domain::Tags::Mesh<1>,
      domain::Tags::Interface<domain::Tags::BoundaryDirectionsExterior<1>,
                              domain::Tags::Direction<1>>,
      domain::Tags::Interface<
          domain::Tags::BoundaryDirectionsExterior<1>,
          ::Tags::Normalized<
              domain::Tags::UnnormalizedFaceNormal<1, Frame::Inertial>>>>>(
      std::move(mesh),
      std::unordered_map<Direction<1>, Direction<1>>{
          std::make_pair(direction, direction)},
      std::unordered_map<Direction<1>, tnsr::i<DataVector, 1>>{
          std::make_pair(direction, std::move(face_normal))});

  const auto check_dirichlet_and_neumann =
      [&direction, &dirichlet_boundary_condition,
       &neumann_boundary_condition](const auto& local_box) {
        Scalar<DataVector> dirichlet_field{size_t{1}, 0.};
        Scalar<DataVector> neumann_field{size_t{1}, 0.};
        (*dirichlet_boundary_condition)(local_box, direction,
                                        make_not_null(&dirichlet_field),
                                        make_not_null(&neumann_field));
        CHECK_ITERABLE_APPROX(get(dirichlet_field), DataVector{3.});
        CHECK_ITERABLE_APPROX(get(neumann_field), DataVector{0.});
        (*neumann_boundary_condition)(local_box, direction,
                                      make_not_null(&dirichlet_field),
                                      make_not_null(&neumann_field));
        CHECK_ITERABLE_APPROX(get(dirichlet_field), DataVector{3.});
        CHECK_ITERABLE_APPROX(get(neumann_field), DataVector{12.});
      };

  SECTION("Non-optional analytic solution") {
    const auto box_with_solutions = db::create_from<
        db::RemoveTags<>,
        db::AddSimpleTags<::Tags::AnalyticSolutions<tmpl::list<
            ScalarFieldTag,
            ::Tags::Flux<ScalarFieldTag, tmpl::size_t<1>, Frame::Inertial>>>>>(
        std::move(box), std::move(analytic_solutions));
    check_dirichlet_and_neumann(box_with_solutions);
  }
  SECTION("Optional analytic solution") {
    const auto box_with_solutions = db::create_from<
        db::RemoveTags<>,
        db::AddSimpleTags<::Tags::AnalyticSolutionsOptional<tmpl::list<
            ScalarFieldTag,
            ::Tags::Flux<ScalarFieldTag, tmpl::size_t<1>, Frame::Inertial>>>>>(
        std::move(box), std::make_optional(std::move(analytic_solutions)));
    check_dirichlet_and_neumann(box_with_solutions);
  }

  {
    INFO("Linearization");
    const auto linearized_boundary_condition =
        dirichlet_boundary_condition->linearization();
    Scalar<DataVector> dirichlet_field{size_t{1}, 1.};
    Scalar<DataVector> neumann_field{size_t{1}, 1.};
    (*linearized_boundary_condition)(box, direction,
                                     make_not_null(&dirichlet_field),
                                     make_not_null(&neumann_field));
    CHECK_ITERABLE_APPROX(get(dirichlet_field), DataVector{0.});
    CHECK_ITERABLE_APPROX(get(neumann_field), DataVector{1.});
  }
}

}  // namespace elliptic::BoundaryConditions
