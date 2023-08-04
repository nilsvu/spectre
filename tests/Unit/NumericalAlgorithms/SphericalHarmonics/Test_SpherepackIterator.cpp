// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Literals.hpp"

SPECTRE_TEST_CASE("Unit.SphericalHarmonics.SpherepackIterator",
                  "[NumericalAlgorithms][Unit]") {
  const std::vector<size_t> test_l = {0, 1, 1, 2, 2, 2, 3, 3, 3, 4,
                                      4, 4, 1, 2, 2, 3, 3, 4, 4};
  const std::vector<size_t> test_m = {0, 0, 1, 0, 1, 2, 0, 1, 2, 0,
                                      1, 2, 1, 1, 2, 1, 2, 1, 2};
  const std::vector<size_t> test_index = {0,   15,  20,  30,  35, 40, 45,
                                          50,  55,  60,  65,  70, 95, 110,
                                          115, 125, 130, 140, 145};

  // [spherepack_iterator_example]
  const size_t l_max = 4;
  const size_t m_max = 2;
  const size_t stride = 5;
  SpherepackIterator iter(l_max, m_max, stride);
  // Allocate space for a SPHEREPACK array
  std::vector<double> array(iter.spherepack_array_size() * stride);
  // Set each array element equal to l+m for real part
  // and l-m for imaginary part.
  size_t i = 0;
  for (iter.reset(); iter; ++iter, ++i) {
    if (iter.coefficient_array() == SpherepackIterator::CoefficientArray::a) {
      array[iter()] = iter.l() + iter.m();
    } else {
      array[iter()] = iter.l() - iter.m();
    }
    CHECK(iter.l() == test_l[i]);
    CHECK(iter.m() == test_m[i]);
    CHECK(iter() == test_index[i]);
  }
  // [spherepack_iterator_example]
  CHECK(iter.l_max() == 4);
  CHECK(iter.m_max() == 2);
  CHECK(iter.n_th() == 5);
  CHECK(iter.n_ph() == 5);
  for (i = 0; i < test_index.size(); ++i) {
    auto j = test_index[i];
    if (i > 11) {  // For specific test_index chosen above.
      // imag part
      CHECK(array[j] == test_l[i] - test_m[i]);
    } else {
      // real part
      CHECK(array[j] == test_l[i] + test_m[i]);
    }
  }

  // Check compact index
  iter.reset();
  for (size_t k = 0; k < array.size(); k++) {
    const auto compact_index = iter.compact_index(k);
    const size_t current_compact_index = iter.current_compact_index();
    if (compact_index) {
      CHECK(*compact_index == current_compact_index);
      ++iter;
    }
  }

  // Test set functions
  CHECK(iter.set(2, 1, SpherepackIterator::CoefficientArray::b)() == 110);
  // Test the set function for the case l>m_max+1
  CHECK(iter.set(4, 1, SpherepackIterator::CoefficientArray::a)() == 65);
  CHECK(iter.set(4, 1, SpherepackIterator::CoefficientArray::b)() == 140);
  CHECK(iter.reset()() == 0);
  CHECK(iter.set(2, 1)() == 35);
  CHECK(iter.set(2, -1)() == 110);
  // Test setting the current compact index
  CHECK(iter.set(0)() == test_index[0]);
  CHECK(iter.set(3)() == test_index[3]);
  CHECK(iter.set(18)() == test_index[18]);
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      ([&iter]() { iter.set(100); })(),
      Catch::Matchers::ContainsSubstring(
          "Trying to set the current compact index to 100 which is "
          "beyond the size of the offset array 19"));
#endif  // SPECTRE_DEBUG

  // Test coefficient_arrya stream operator (assumes output of last 'set').
  CHECK(get_output(iter.coefficient_array()) == "b");

  // Test inequality
  const SpherepackIterator iter2(3, 2, 5);  // Different lmax,mmax
  const SpherepackIterator iter3(4, 2, 4);  // Different stride
  const SpherepackIterator iter4(4, 2, 5);  // Different current state
  CHECK(iter2 != iter);
  CHECK(iter != iter2);
  CHECK(iter != iter3);
  CHECK(iter3 != iter);
  CHECK(iter4 != iter);
  CHECK(iter != iter4);

  const auto iter_copy = iter;
  CHECK(iter_copy == iter);
  test_move_semantics(std::move(iter), iter_copy, 3_st, 2_st, 3_st);
}
