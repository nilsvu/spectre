// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <complex>
#include <tuple>

#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/ModalVector.hpp"
#include "Helpers/DataStructures/VectorImplTestHelper.hpp"
#include "Utilities/DereferenceWrapper.hpp"   // IWYU pragma: keep
#include "Utilities/ErrorHandling/Error.hpp"  // IWYU pragma: keep
#include "Utilities/Functional.hpp"
#include "Utilities/StdHelpers.hpp"  // IWYU pragma: keep
#include "Utilities/TypeTraits.hpp"  // IWYU pragma: keep

namespace {
void test_complex_modal_vector_math() {
  const TestHelpers::VectorImpl::Bound generic{{-100.0, 100.0}};

  const auto unary_ops = std::make_tuple(
      std::make_tuple(funcl::Conj<>{}, std::make_tuple(generic)),
      std::make_tuple(funcl::Imag<>{}, std::make_tuple(generic)),
      std::make_tuple(funcl::Real<>{}, std::make_tuple(generic)));

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::Normal, ComplexModalVector>(unary_ops);

  const auto real_unary_ops = std::make_tuple(
      std::make_tuple(funcl::Imag<>{}, std::make_tuple(generic)),
      std::make_tuple(funcl::Real<>{}, std::make_tuple(generic)));

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::Normal, ModalVector>(real_unary_ops);

  const auto binary_ops = std::make_tuple(
      std::make_tuple(funcl::Minus<>{}, std::make_tuple(generic, generic)),
      std::make_tuple(funcl::Plus<>{}, std::make_tuple(generic, generic)));

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::Normal, ComplexModalVector,
      ModalVector>(binary_ops);

  const auto cascaded_ops = std::make_tuple(
      std::make_tuple(funcl::Minus<funcl::Plus<>, funcl::Identity>{},
                      std::make_tuple(generic, generic, generic)));

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::Strict, ComplexModalVector,
      ModalVector>(cascaded_ops);

  const auto array_binary_ops = std::make_tuple(
      std::make_tuple(funcl::Minus<>{}, std::make_tuple(generic, generic)),
      std::make_tuple(funcl::Plus<>{}, std::make_tuple(generic, generic)));

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::Strict,
      std::array<ComplexModalVector, 2>>(array_binary_ops);

  // Note that the binary operations that involve a complex modal vector and
  // various scalar types have been moved to
  // `Test_ComplexModalVectorInhomogeneousOperations.cpp` in an effort to better
  // parallelize the build.
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.ComplexModalVector",
                  "[DataStructures][Unit]") {
  {
    INFO("test construct and assign");
    TestHelpers::VectorImpl::vector_test_construct_and_assign<
        ComplexModalVector, std::complex<double>>();
  }
  {
    INFO("test serialize and deserialize");
    TestHelpers::VectorImpl::vector_test_serialize<ComplexModalVector,
                                                   std::complex<double>>();
  }
  {
    INFO("test set_data_ref functionality");
    TestHelpers::VectorImpl::vector_test_ref<ComplexModalVector,
                                             std::complex<double>>();
  }
  {
    INFO("test math after move");
    TestHelpers::VectorImpl::vector_test_math_after_move<
        ComplexModalVector, std::complex<double>>();
  }
  {
    INFO("test ComplexModalVector math operations");
    test_complex_modal_vector_math();
  }

#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      TestHelpers::VectorImpl::vector_ref_test_size_error<ComplexModalVector>(
          TestHelpers::VectorImpl::RefSizeErrorTestKind::ExpressionAssign),
      Catch::Matchers::Contains("Must assign into same size"));
  CHECK_THROWS_WITH(
      TestHelpers::VectorImpl::vector_ref_test_size_error<ComplexModalVector>(
          TestHelpers::VectorImpl::RefSizeErrorTestKind::Copy),
      Catch::Matchers::Contains("Must copy into same size"));
  CHECK_THROWS_WITH(
      TestHelpers::VectorImpl::vector_ref_test_size_error<ComplexModalVector>(
          TestHelpers::VectorImpl::RefSizeErrorTestKind::Move),
      Catch::Matchers::Contains("Must copy into same size"));
#endif
}
