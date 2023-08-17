// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Code to wrap or improve the Catch testing framework used for unit tests.

#pragma once

#define CATCH_CONFIG_ENABLE_ALL_STRINGMAKERS

#include <blaze/math/typetraits/IsColumnMajorMatrix.h>
#include <blaze/math/typetraits/IsDenseMatrix.h>
#include <catch2/catch_all.hpp>
#include <csignal>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/StlStreamDeclarations.hpp"
#include "Utilities/System/Abort.hpp"
#include "Utilities/System/Exit.hpp"
#include "Utilities/TypeTraits/IsA.hpp"
#include "Utilities/TypeTraits/IsIterable.hpp"
#include "Utilities/TypeTraits/IsMaplike.hpp"

/*!
 * \ingroup TestingFrameworkGroup
 * \brief A replacement for Catch's TEST_CASE that silences clang-tidy warnings
 */
#define SPECTRE_TEST_CASE(m, n) TEST_CASE(m, n)  // NOLINT

/*!
 * \ingroup TestingFrameworkGroup
 * \brief A similar to Catch's REQUIRE statement, but can be used in tests that
 * spawn several chares with possibly complex interaction between the chares.
 */
#define SPECTRE_PARALLEL_REQUIRE(expr)                                  \
  do {                                                                  \
    if (not(expr)) {                                                    \
      ERROR("\nFailed comparison: " << #expr << "\nLine: " << __LINE__  \
                                    << "\nFile: " << __FILE__ << "\n"); \
    }                                                                   \
  } while (false)

/*!
 * \ingroup TestingFrameworkGroup
 * \brief A similar to Catch's REQUIRE_FALSE statement, but can be used in tests
 * that spawn several chares with possibly complex interaction between the
 * chares.
 */
#define SPECTRE_PARALLEL_REQUIRE_FALSE(expr)                            \
  do {                                                                  \
    if ((expr)) {                                                       \
      ERROR("\nFailed comparison: " << #expr << "\nLine: " << __LINE__  \
                                    << "\nFile: " << __FILE__ << "\n"); \
    }                                                                   \
  } while (false)

using Approx = Catch::Approx;

/*!
 * \ingroup TestingFrameworkGroup
 * \brief Set a default tolerance for floating-point number comparison
 *
 * \details
 * Catch's default (relative) tolerance for comparing floating-point numbers is
 * `std::numeric_limits<float>::%epsilon() * 100`, or roughly \f$10^{-5}\f$.
 * This tolerance is too loose for checking many scientific algorithms that
 * rely on double precision floating-point accuracy, so we provide a tighter
 * tighter tolerance through the `approx` static object.
 *
 * \example
 * \snippet Test_TestingFramework.cpp approx_test
 */
// clang-tidy: static object creation may throw exception
static Approx approx =                                          // NOLINT
    Approx::custom()                                            // NOLINT
        .epsilon(std::numeric_limits<double>::epsilon() * 100)  // NOLINT
        .scale(1.0);                                            // NOLINT

/*!
 * \ingroup TestingFrameworkGroup
 * \brief A wrapper around Catch's CHECK macro that checks approximate
 * equality of the two entries in a std::complex. For efficiency, no function
 * forwarding is performed, just a pair of `CHECK`s inline
 */
#define CHECK_COMPLEX_APPROX(a, b)                                     \
  do {                                                                 \
    INFO(__FILE__ ":" + std::to_string(__LINE__) + ": " #a " == " #b); \
    CHECK(approx(real(a)) == real(b));                                 \
    CHECK(approx(imag(a)) == imag(b));                                 \
  } while (false)

/*!
 * \ingroup TestingFrameworkGroup
 * \brief Same as `CHECK_COMPLEX_APPROX` with user-defined Approx.
 *  The third argument should be of type `Approx`.
 */
#define CHECK_COMPLEX_CUSTOM_APPROX(a, b, appx)                        \
  do {                                                                 \
    INFO(__FILE__ ":" + std::to_string(__LINE__) + ": " #a " == " #b); \
    CHECK(appx(real(a)) == real(b));                                   \
    CHECK(appx(imag(a)) == imag(b));                                   \
  } while (false)

/*!
 * \ingroup TestingFrameworkGroup
 * \brief A wrapper around Catch's CHECK macro that checks approximate
 * equality of entries in iterable containers.  For maplike
 * containers, keys are checked for strict equality and values are
 * checked for approximate equality.
 */
#define CHECK_ITERABLE_APPROX(a, b)                                          \
  do {                                                                       \
    INFO(__FILE__ ":" + std::to_string(__LINE__) + ": " #a " == " #b);       \
    check_iterable_approx<std::common_type_t<                                \
        std::decay_t<decltype(a)>, std::decay_t<decltype(b)>>>::apply(a, b); \
  } while (false)

/*!
 * \ingroup TestingFrameworkGroup
 * \brief Same as `CHECK_ITERABLE_APPROX` with user-defined Approx.
 *  The third argument should be of type `Approx`.
 */
#define CHECK_ITERABLE_CUSTOM_APPROX(a, b, appx)                             \
  do {                                                                       \
    INFO(__FILE__ ":" + std::to_string(__LINE__) + ": " #a " == " #b);       \
    check_iterable_approx<std::common_type_t<                                \
        std::decay_t<decltype(a)>, std::decay_t<decltype(b)>>>::apply(a, b,  \
                                                                      appx); \
  } while (false)

/// \cond HIDDEN_SYMBOLS
template <typename T, typename = std::nullptr_t>
struct check_iterable_approx {
  // clang-tidy: non-const reference
  static void apply(const T& a, const T& b, Approx& appx = approx) {  // NOLINT
    CHECK(a == appx(b));
  }
};

template <typename T>
struct check_iterable_approx<std::complex<T>, std::nullptr_t> {
  // clang-tidy: non-const reference
  static void apply(const std::complex<T>& a, const std::complex<T>& b,
                    Approx& appx = approx) {  // NOLINT
    check_iterable_approx<T>::apply(real(a), real(b), appx);
    check_iterable_approx<T>::apply(imag(a), imag(b), appx);
  }
};

template <typename T>
struct check_iterable_approx<
    T, Requires<not tt::is_maplike_v<T> and tt::is_iterable_v<T> and
                not tt::is_a_v<std::unordered_set, T>>> {
  // clang-tidy: non-const reference
  static void apply(const T& a, const T& b, Approx& appx = approx) {  // NOLINT
    CAPTURE(a);
    CAPTURE(b);
    auto a_it = a.begin();
    auto b_it = b.begin();
    CHECK(a_it != a.end());
    CHECK(b_it != b.end());
    while (a_it != a.end() and b_it != b.end()) {
      check_iterable_approx<std::decay_t<decltype(*a_it)>>::apply(*a_it, *b_it,
                                                                  appx);
      ++a_it;
      ++b_it;
    }
    {
      INFO("Iterable is longer in first argument than in second argument");
      CHECK(a_it == a.end());
    }
    {
      INFO("Iterable is shorter in first argument than in second argument");
      CHECK(b_it == b.end());
    }
  }
};

template <typename T>
struct check_iterable_approx<T, Requires<tt::is_a_v<std::unordered_set, T>>> {
  // clang-tidy: non-const reference
  static void apply(const T& a, const T& b,
                    Approx& /*appx*/ = approx) {  // NOLINT
    // Approximate comparison of unordered sets is difficult
    CHECK(a == b);
  }
};

template <typename T>
struct check_iterable_approx<
    T, Requires<tt::is_maplike_v<T> and tt::is_iterable_v<T>>> {
  // clang-tidy: non-const reference
  static void apply(const T& a, const T& b, Approx& appx = approx) {  // NOLINT
    CAPTURE(a);
    CAPTURE(b);
    for (const auto& kv : a) {
      const auto& key = kv.first;
      try {
        const auto& a_value = kv.second;
        const auto& b_value = b.at(key);
        CAPTURE(key);
        check_iterable_approx<std::decay_t<decltype(a_value)>>::apply(
            a_value, b_value, appx);
      } catch (const std::out_of_range&) {
        INFO("Missing key in second container: " << key);
        CHECK(false);
      }
    }

    for (const auto& kv : b) {
      const auto& key = kv.first;
      try {
        a.at(key);
        // We've checked that the values match above.
      } catch (const std::out_of_range&) {
        INFO("Missing key in first container: " << key);
        CHECK(false);
      }
    }
  }
};

#define CHECK_MATRIX_APPROX(a, b)                                            \
  do {                                                                       \
    INFO(__FILE__ ":" + std::to_string(__LINE__) + ": " #a " == " #b);       \
    check_matrix_approx<std::common_type_t<                                  \
        std::decay_t<decltype(a)>, std::decay_t<decltype(b)>>>::apply(a, b); \
  } while (false)

#define CHECK_MATRIX_CUSTOM_APPROX(a, b, appx)                               \
  do {                                                                       \
    INFO(__FILE__ ":" + std::to_string(__LINE__) + ": " #a " == " #b);       \
    check_matrix_approx<std::common_type_t<                                  \
        std::decay_t<decltype(a)>, std::decay_t<decltype(b)>>>::apply(a, b,  \
                                                                      appx); \
  } while (false)

template <typename M>
struct check_matrix_approx {
  // clang-tidy: non-const reference
  static void apply(const M& a, const M& b,
                    Approx& appx = approx) {  // NOLINT
    // This implementation is for a column-major matrix. It does not trivially
    // generalize to a row-major matrix because the iterator
    // `blaze::DynamicMatrix<T, SO>::cbegin(i)` traverses either a row or a
    // column, and takes its index as argument.
    static_assert(blaze::IsColumnMajorMatrix_v<M> and
                  blaze::IsDenseMatrix_v<M>);
    CHECK(a.columns() == b.columns());
    for (size_t j = 0; j < a.columns(); j++) {
      CAPTURE(a);
      CAPTURE(b);
      auto a_it = a.cbegin(j);
      auto b_it = b.cbegin(j);
      CHECK(a_it != a.cend(j));
      CHECK(b_it != b.cend(j));
      while (a_it != a.cend(j) and b_it != b.cend(j)) {
        check_iterable_approx<std::decay_t<decltype(*a_it)>>::apply(
            *a_it, *b_it, appx);
        ++a_it;
        ++b_it;
      }
      {
        INFO("Column " << j
                       << " of the first matrix is longer than that of the "
                          "second matrix.");
        CHECK(a_it == a.end(j));
      }
      {
        INFO("Column " << j
                       << " of the first matrix is shorter than that of the "
                          "second matrix.");
        CHECK(b_it == b.end(j));
      }
    }
  }
};
/// \endcond

/// \cond HIDDEN_SYMBOLS
[[noreturn]] inline void spectre_testing_signal_handler(int /*signal*/) {
  sys::exit();
}
/// \endcond

/*!
 * \ingroup TestingFrameworkGroup
 * \brief Mark a test as checking the output with a regular expression
 *
 * \details
 * The OUTPUT_TEST() macro should be the first line in the SPECTRE_TEST_CASE.
 * Catch requires at least one CHECK in each test to pass, so we add one in
 * case nothing but the output is checked.
 *
 * \example
 * \snippet Test_Parallel.cpp output_test_example
 */
#define OUTPUT_TEST() \
  do {                \
    CHECK(true);      \
  } while (false)
