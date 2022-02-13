// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <blaze/math/CompressedVector.h>
#include <blaze/math/DynamicVector.h>
#include <cstddef>

#include "DataStructures/CompressedMatrix/OptionCreation.hpp"
#include "DataStructures/CompressedMatrix/Serialization.hpp"
#include "DataStructures/CompressedVector/OptionCreation.hpp"
#include "DataStructures/CompressedVector/Serialization.hpp"
#include "DataStructures/DynamicMatrix/OptionCreation.hpp"
#include "DataStructures/DynamicMatrix/Serialization.hpp"
#include "DataStructures/DynamicVector/MakeWithValue.hpp"
#include "DataStructures/DynamicVector/OptionCreation.hpp"
#include "DataStructures/DynamicVector/Serialization.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/MakeWithValue.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.Blaze", "[DataStructures][Unit]") {
  {
    INFO("DynamicVector");
    test_serialization(blaze::DynamicVector<double>{0., 1., 0., 2.});
    CHECK(make_with_value<blaze::DynamicVector<double>>(
              blaze::DynamicVector<double>(3), 1.) ==
          blaze::DynamicVector<double>(3, 1.));
    CHECK(TestHelpers::test_creation<blaze::DynamicVector<double>>(
              "[0., 1., 0., 2.]") ==
          blaze::DynamicVector<double>{0., 1., 0., 2.});
  }
  {
    INFO("CompressedVector");
    test_serialization(blaze::CompressedVector<double>{0., 1., 0., 2.});
    CHECK(TestHelpers::test_creation<blaze::CompressedVector<double>>(
              "[0., 1., 0., 2.]") ==
          blaze::CompressedVector<double>{0., 1., 0., 2.});
  }
  {
    INFO("DynamicMatrix");
    test_serialization(blaze::DynamicMatrix<double, blaze::columnMajor>{
        {0., 1., 2.}, {3., 0., 4.}});
    test_serialization(blaze::DynamicMatrix<double, blaze::rowMajor>{
        {0., 1., 2.}, {3., 0., 4.}});
    CHECK(TestHelpers::test_creation<
              blaze::DynamicMatrix<double, blaze::columnMajor>>(
              "[[0., 1., 2.], [3., 0., 4.]]") ==
          blaze::DynamicMatrix<double, blaze::columnMajor>{{0., 1., 2.},
                                                           {3., 0., 4.}});
    CHECK(TestHelpers::test_creation<
              blaze::DynamicMatrix<double, blaze::rowMajor>>(
              "[[0., 1., 2.], [3., 0., 4.]]") ==
          blaze::DynamicMatrix<double, blaze::rowMajor>{{0., 1., 2.},
                                                        {3., 0., 4.}});
  }
  {
    INFO("CompressedMatrix");
    test_serialization(blaze::CompressedMatrix<double, blaze::columnMajor>{
        {0., 1., 2.}, {3., 0., 4.}});
    test_serialization(blaze::CompressedMatrix<double, blaze::rowMajor>{
        {0., 1., 2.}, {3., 0., 4.}});
    CHECK(TestHelpers::test_creation<
              blaze::CompressedMatrix<double, blaze::columnMajor>>(
              "[[0., 1., 2.], [3., 0., 4.]]") ==
          blaze::CompressedMatrix<double, blaze::columnMajor>{{0., 1., 2.},
                                                              {3., 0., 4.}});
    CHECK(TestHelpers::test_creation<
              blaze::CompressedMatrix<double, blaze::rowMajor>>(
              "[[0., 1., 2.], [3., 0., 4.]]") ==
          blaze::CompressedMatrix<double, blaze::rowMajor>{{0., 1., 2.},
                                                           {3., 0., 4.}});
  }
}
