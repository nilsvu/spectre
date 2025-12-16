// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>

#include "DataStructures/Tensor/EagerMath/GramSchmidtOrthonormalize.hpp"
#include "DataStructures/Tensor/Tensor.hpp"

SPECTRE_TEST_CASE("Unit.Tensor.EagerMath.GramSchmidtOrthonormalize",
                  "[DataStructures][Unit]") {
  tnsr::aa<double, 3> minkowski_metric{};
  get<0, 0>(minkowski_metric) = -1.0;
  get<1, 1>(minkowski_metric) = 1.0;
  get<2, 2>(minkowski_metric) = 1.0;
  get<3, 3>(minkowski_metric) = 1.0;

  {
    INFO("Already orthonormal basis");
    tnsr::A<double, 3> vec1{{{1.0, 0.0, 0.0, 0.0}}};
    tnsr::A<double, 3> vec2{{{0.0, 1.0, 0.0, 0.0}}};
    tnsr::A<double, 3> vec3{{{0.0, 0.0, 1.0, 0.0}}};

    gram_schmidt_orthonormalize(
        std::array{make_not_null(&vec1), make_not_null(&vec2),
                   make_not_null(&vec3)},
        minkowski_metric);

    CHECK(get(dot_product(vec1, vec1, minkowski_metric)) == approx(-1.0));
    CHECK(get(dot_product(vec2, vec2, minkowski_metric)) == approx(1.0));
    CHECK(get(dot_product(vec3, vec3, minkowski_metric)) == approx(1.0));
    CHECK(get(dot_product(vec1, vec2, minkowski_metric)) == approx(0.0));
    CHECK(get(dot_product(vec1, vec3, minkowski_metric)) == approx(0.0));
    CHECK(get(dot_product(vec2, vec3, minkowski_metric)) == approx(0.0));
  }

  {
    INFO("Linearly independent, not orthonormal");
    tnsr::A<double, 3> vec1{{{2.0, 1.0, 0.0, 0.0}}};
    tnsr::A<double, 3> vec2{{{1.0, 1.0, 1.0, 0.0}}};
    tnsr::A<double, 3> vec3{{{1.0, 2.0, -1.0, 1.0}}};

    gram_schmidt_orthonormalize(
        std::array{make_not_null(&vec1), make_not_null(&vec2),
                   make_not_null(&vec3)},
        minkowski_metric);

    CHECK(get(dot_product(vec1, vec1, minkowski_metric)) == approx(-1.0));
    CHECK(get(dot_product(vec2, vec2, minkowski_metric)) == approx(1.0));
    CHECK(get(dot_product(vec3, vec3, minkowski_metric)) == approx(1.0));
    CHECK(get(dot_product(vec1, vec2, minkowski_metric)) == approx(0.0));
    CHECK(get(dot_product(vec1, vec3, minkowski_metric)) == approx(0.0));
    CHECK(get(dot_product(vec2, vec3, minkowski_metric)) == approx(0.0));
  }
}
