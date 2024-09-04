// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <type_traits>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Tags/Faces.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace domain {

namespace {

using namespace std::complex_literals;

template <typename DataType, size_t Dim>
struct FacesTestCompute
    : Tags::Faces<
          Dim, ::Tags::Variables<tmpl::list<::Tags::TempScalar<0, DataType>>>>,
      db::ComputeTag {
  using base = Tags::Faces<
      Dim, ::Tags::Variables<tmpl::list<::Tags::TempScalar<0, DataType>>>>;
  using argument_tags = tmpl::list<>;
  static void function(
      const gsl::not_null<DirectionMap<
          Dim, Variables<tmpl::list<::Tags::TempScalar<0, DataType>>>>*>
          vars_on_faces) {
    (*vars_on_faces)[Direction<Dim>::lower_xi()] =
        Variables<tmpl::list<::Tags::TempScalar<0, DataType>>>{3, 1.};
    if constexpr (std::is_same_v<DataType, ComplexDataVector>) {
      (*vars_on_faces)[Direction<Dim>::lower_xi()] += ComplexDataVector(3, 2.i);
    }
  }
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Tags.Faces", "[Unit][Domain]") {
  constexpr size_t Dim = 1;
  TestHelpers::db::test_prefix_tag<Tags::Faces<Dim, ::Tags::TempScalar<0>>>(
      "Faces(TempTensor0)");
  static_assert(
      std::is_same_v<
          make_faces_tags<
              Dim, tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>>,
              tmpl::list<::Tags::TempScalar<1>>>,
          tmpl::list<Tags::Faces<Dim, ::Tags::TempScalar<0>>,
                     ::Tags::TempScalar<1>>>);
  using vars_tag = ::Tags::Variables<tmpl::list<::Tags::TempScalar<0>>>;
  using vars_on_faces_tag = Tags::Faces<Dim, vars_tag>;
  using Vars = Variables<tmpl::list<::Tags::TempScalar<0>>>;
  using scalar_on_faces_tag = Tags::Faces<Dim, ::Tags::TempScalar<0>>;
  {
    INFO("Subitems");
    auto box = db::create<db::AddSimpleTags<vars_on_faces_tag>>(
        DirectionMap<Dim, Vars>{});
    CHECK(db::get<scalar_on_faces_tag>(box).empty());
    db::mutate<vars_on_faces_tag>(
        [](const gsl::not_null<DirectionMap<Dim, Vars>*> vars_on_faces) {
          vars_on_faces->emplace(Direction<Dim>::lower_xi(),
                                 Vars{size_t{3}, 0.});
        },
        make_not_null(&box));
    CHECK(db::get<scalar_on_faces_tag>(box).at(Direction<Dim>::lower_xi()) ==
          Scalar<DataVector>{size_t{3}, 0.});
    db::mutate<scalar_on_faces_tag>(
        [](const gsl::not_null<DirectionMap<Dim, Scalar<DataVector>>*>
               scalar_on_faces) {
          get(scalar_on_faces->at(Direction<Dim>::lower_xi())) = 1.;
        },
        make_not_null(&box));
    CHECK(db::get<vars_on_faces_tag>(box).at(Direction<Dim>::lower_xi()) ==
          Vars{size_t{3}, 1.});
  }
  {
    INFO("Compute-subitems");
    auto box =
        db::create<db::AddSimpleTags<>,
                   db::AddComputeTags<FacesTestCompute<DataVector, Dim>>>();
    CHECK(get(db::get<scalar_on_faces_tag>(box).at(
              Direction<Dim>::lower_xi())) == DataVector{size_t{3}, 1.});
  }
  {
    INFO("Compute-subitems complex");
    auto box = db::create<
        db::AddSimpleTags<>,
        db::AddComputeTags<FacesTestCompute<ComplexDataVector, Dim>>>();
    CHECK(
        get(db::get<Tags::Faces<Dim, ::Tags::TempScalar<0, ComplexDataVector>>>(
                box)
                .at(Direction<Dim>::lower_xi())) ==
        ComplexDataVector{size_t{3}, 1. + 2.i});
  }
}

}  // namespace domain
