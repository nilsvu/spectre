// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/ModalVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/DataStructures/MathWrapper.hpp"
#include "Helpers/DataStructures/TestTags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"  // IWYU pragma: keep
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"  // IWYU pragma: keep
#include "Utilities/MakeString.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits/GetFundamentalType.hpp"

// IWYU pragma: no_include <boost/tuple/tuple.hpp>

// IWYU pragma: no_include "DataStructures/VariablesForwardDecl.hpp"

// IWYU pragma: no_forward_declare Variables

static_assert(
    std::is_move_constructible<
        Variables<tmpl::list<TestHelpers::Tags::Scalar<DataVector>,
                             TestHelpers::Tags::Vector<DataVector>>>>::value);

static_assert(not std::is_convertible_v<
              DataVector,
              Variables<tmpl::list<TestHelpers::Tags::Scalar<DataVector>>>>);

namespace {
std::string repeat_string_with_commas(const std::string& str,
                                      const size_t repeats) {
  MakeString t;
  for (size_t i = 0; i < repeats - 1; ++i) {
    t << str << ",";
  }
  t << str;
  return t;
}

// An empty Variables is a separate implementation, so we put tests
// for it in a separate function here.
void test_empty_variables() {
  Variables<tmpl::list<>> empty_vars;
  auto serialized_empty_vars = serialize_and_deserialize(empty_vars);
  CHECK(serialized_empty_vars == empty_vars);

  // The following test with a std::tuple<Variables<tmpl::list<>>>
  // revealed a gcc8 bug that passed the above test (the one that
  // serializes/deserializes a Variables<tmpl::list<>>).
  std::tuple<int, std::string, Variables<tmpl::list<>>> tuple_with_empty_vars{
      3, "hello", {}};
  auto serialized_tuple_with_empty_vars =
      serialize_and_deserialize(tuple_with_empty_vars);
  CHECK(serialized_tuple_with_empty_vars == tuple_with_empty_vars);

  CHECK(get_output(tuple_with_empty_vars) == "(3,hello,{})");

  CHECK(not contains_allocations(empty_vars));
}

template <typename VectorType>
void test_variables_construction_and_access() {
  using value_type = typename VectorType::value_type;
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<tt::get_fundamental_type_t<value_type>> dist{-100.0,
                                                                         100.0};
  UniformCustomDistribution<size_t> sdist{1, 5};

  const size_t number_of_grid_points = sdist(gen);
  // Test constructed and member function initialized Variables are identical
  const auto initial_fill_value = make_with_random_values<value_type>(
      make_not_null(&gen), make_not_null(&dist));

  using VariablesType =
      Variables<tmpl::list<TestHelpers::Tags::Vector<VectorType>,
                           TestHelpers::Tags::Scalar<VectorType>,
                           TestHelpers::Tags::Scalar2<VectorType>>>;

  VariablesType filled_variables{number_of_grid_points, initial_fill_value};

  VariablesType initialized_variables;
  initialized_variables.initialize(number_of_grid_points, initial_fill_value);
  CHECK(filled_variables == initialized_variables);

  CHECK(filled_variables.size() ==
        filled_variables.number_of_grid_points() *
            filled_variables.number_of_independent_components);
  CHECK(number_of_grid_points == filled_variables.number_of_grid_points());
  CHECK(number_of_grid_points * 5 == filled_variables.size());
  for (size_t i = 0; i < filled_variables.size(); ++i) {
    // clang-tidy: do not use pointer arithmetic
    CHECK(initial_fill_value == filled_variables.data()[i]);  // NOLINT
  }
  // Test that assigning to a tensor pointing into a Variables correctly sets
  // the Variables
  auto& tensor_in_filled_variables =
      get<TestHelpers::Tags::Vector<VectorType>>(filled_variables);
  CHECK(VectorType{number_of_grid_points, initial_fill_value} ==
        get<0>(tensor_in_filled_variables));
  CHECK(VectorType{number_of_grid_points, initial_fill_value} ==
        get<1>(tensor_in_filled_variables));
  CHECK(VectorType{number_of_grid_points, initial_fill_value} ==
        get<2>(tensor_in_filled_variables));

  {
    // Test structured bindings
    const auto& [vector, scalar1, scalar2] = filled_variables;
    CHECK(get(scalar1).data() ==
          get(get<TestHelpers::Tags::Scalar<VectorType>>(filled_variables))
              .data());
    CHECK(get(scalar2).data() ==
          get(get<TestHelpers::Tags::Scalar2<VectorType>>(filled_variables))
              .data());
    for (size_t i = 0; i < 3; ++i) {
      CAPTURE(i);
      CHECK(vector.get(i).data() ==
            get<TestHelpers::Tags::Vector<VectorType>>(filled_variables)
                .get(i)
                .data());
    }
  }

  const auto value_for_reference_assignment =
      make_with_random_values<value_type>(make_not_null(&gen),
                                          make_not_null(&dist));

  tnsr::I<VectorType, 3> tensor_for_reference_assignment{
      number_of_grid_points, value_for_reference_assignment};
  tensor_in_filled_variables = tensor_for_reference_assignment;

  // clang-tidy: do not use pointer arithmetic
  for (size_t i = 0; i < number_of_grid_points * 3; ++i) {
    CHECK(value_for_reference_assignment ==
          filled_variables.data()[i]);  // NOLINT
  }
  CHECK(VectorType{number_of_grid_points, value_for_reference_assignment} ==
        get<0>(tensor_in_filled_variables));
  CHECK(VectorType{number_of_grid_points, value_for_reference_assignment} ==
        get<1>(tensor_in_filled_variables));
  CHECK(VectorType{number_of_grid_points, value_for_reference_assignment} ==
        get<2>(tensor_in_filled_variables));

  const auto value_for_tensor_constructor = make_with_random_values<value_type>(
      make_not_null(&gen), make_not_null(&dist));

  tensor_in_filled_variables = tnsr::I<VectorType, 3>{
      number_of_grid_points, value_for_tensor_constructor};

  // clang-tidy: do not use pointer arithmetic
  for (size_t i = 0; i < number_of_grid_points * 3; ++i) {
    CHECK(value_for_tensor_constructor ==
          filled_variables.data()[i]);  // NOLINT
  }
  CHECK(VectorType{number_of_grid_points, value_for_tensor_constructor} ==
        get<0>(tensor_in_filled_variables));
  CHECK(VectorType{number_of_grid_points, value_for_tensor_constructor} ==
        get<1>(tensor_in_filled_variables));
  CHECK(VectorType{number_of_grid_points, value_for_tensor_constructor} ==
        get<2>(tensor_in_filled_variables));

  // Test const vector in variables points to right values
  const auto& const_tensor_in_filled_variables =
      get<TestHelpers::Tags::Vector<VectorType>>(filled_variables);
  CHECK(get<0>(const_tensor_in_filled_variables) ==
        VectorType{number_of_grid_points, value_for_tensor_constructor});
  CHECK(get<1>(const_tensor_in_filled_variables) ==
        VectorType{number_of_grid_points, value_for_tensor_constructor});
  CHECK(get<2>(const_tensor_in_filled_variables) ==
        VectorType{number_of_grid_points, value_for_tensor_constructor});

  // Test equivalence and inequivalence operators
  VariablesType another_filled_variables{number_of_grid_points,
                                         initial_fill_value};
  // handle case of same value rolled twice on the random number generators
  CHECK((filled_variables == another_filled_variables) ==
        (initial_fill_value == value_for_tensor_constructor));
  another_filled_variables = filled_variables;
  CHECK(another_filled_variables == filled_variables);

  // Test default constructed variables is allowed
  VariablesType default_variables;

  CHECK(default_variables.size() == 0);
  CHECK(default_variables.number_of_grid_points() == 0);
  default_variables = another_filled_variables;
  CHECK(another_filled_variables == default_variables);

  std::string expected_output_initial = repeat_string_with_commas(
      get_output(initial_fill_value), number_of_grid_points);

  std::string expected_output_tensor = repeat_string_with_commas(
      get_output(value_for_tensor_constructor), number_of_grid_points);

  // Test stream operator
  const std::string expected_output =
      "Vector:\n"
      "T(0)=(" +
      expected_output_tensor +
      ")\n"
      "T(1)=(" +
      expected_output_tensor +
      ")\n"
      "T(2)=(" +
      expected_output_tensor +
      ")\n\n"
      "Scalar:\n"
      "T()=(" +
      expected_output_initial +
      ")\n\n"
      "Scalar2:\n"
      "T()=(" +
      expected_output_initial + ")";
  CHECK(get_output(filled_variables) == expected_output);

  // Check self-assignment
#if defined(__clang__) && __clang_major__ > 6
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wself-assign-overloaded"
#endif  // defined(__clang__) && __clang_major__ > 6
  filled_variables = filled_variables;  // NOLINT
#if defined(__clang__) && __clang_major__ > 6
#pragma GCC diagnostic pop
#endif  // defined(__clang__) && __clang_major__ > 6
  CHECK(filled_variables == another_filled_variables);

  TestHelpers::db::test_simple_tag<
      Tags::Variables<tmpl::list<TestHelpers::Tags::Vector<VectorType>,
                                 TestHelpers::Tags::Scalar<VectorType>,
                                 TestHelpers::Tags::Scalar2<VectorType>>>>(
      "Variables(Vector,Scalar,Scalar2)");
}

template <typename VectorType>
void test_variables_move() {
  using value_type = typename VectorType::value_type;
  using Vars = Variables<tmpl::list<TestHelpers::Tags::Vector<VectorType>>>;
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<tt::get_fundamental_type_t<value_type>> dist{-100.0,
                                                                         100.0};
  UniformCustomDistribution<size_t> sdist{1, 5};

  const size_t number_of_grid_points1 = sdist(gen);
  const size_t number_of_grid_points2 = sdist(gen);
  const auto initial_fill_values =
      make_with_random_values<std::array<value_type, 2>>(make_not_null(&gen),
                                                         make_not_null(&dist));

  // Test moving with assignment operator
  {
    // non-owning semantics
    std::vector<value_type> non_owning_data_to(
        number_of_grid_points1 * Vars::number_of_independent_components,
        initial_fill_values[0]);
    Vars non_owning_move_to(non_owning_data_to.data(),
                            non_owning_data_to.size());
    std::vector<value_type> non_owning_data_from(
        number_of_grid_points2 * Vars::number_of_independent_components,
        initial_fill_values[1]);
    Vars non_owning_move_from(non_owning_data_from.data(),
                              non_owning_data_from.size());

    // Test that copying a non-owning provides an owning
    Vars copied_non_owning{non_owning_move_to};
    CHECK(copied_non_owning.is_owning());
    CHECK(not non_owning_move_to.is_owning());
    CHECK(copied_non_owning.size() == non_owning_move_to.size());
    CHECK(copied_non_owning.number_of_grid_points() ==
          non_owning_move_to.number_of_grid_points());
    CHECK(&get<TestHelpers::Tags::Vector<VectorType>>(
              copied_non_owning)[0][0] == copied_non_owning.data());
    CHECK(&get<TestHelpers::Tags::Vector<VectorType>>(
              copied_non_owning)[0][0] != non_owning_move_from.data());
    CHECK(&get<TestHelpers::Tags::Vector<VectorType>>(
              copied_non_owning)[0][0] != non_owning_move_to.data());

    copied_non_owning = non_owning_move_from;
    CHECK(copied_non_owning.is_owning());
    CHECK(not non_owning_move_from.is_owning());
    CHECK(copied_non_owning.size() == non_owning_move_from.size());
    CHECK(copied_non_owning.number_of_grid_points() ==
          non_owning_move_from.number_of_grid_points());
    CHECK(&get<TestHelpers::Tags::Vector<VectorType>>(
              copied_non_owning)[0][0] == copied_non_owning.data());
    CHECK(&get<TestHelpers::Tags::Vector<VectorType>>(
              copied_non_owning)[0][0] != non_owning_move_from.data());
    CHECK(&get<TestHelpers::Tags::Vector<VectorType>>(
              copied_non_owning)[0][0] != non_owning_move_to.data());

    Vars initializer_move_to = std::move(non_owning_move_to);
    CHECK(not initializer_move_to.is_owning());
    CHECK(initializer_move_to ==
          Vars{number_of_grid_points1, initial_fill_values[0]});
    CHECK(initializer_move_to.data() == non_owning_data_to.data());
    CHECK(&get<TestHelpers::Tags::Vector<VectorType>>(
              initializer_move_to)[0][0] == non_owning_data_to.data());
    CHECK(non_owning_move_to.is_owning());
    CHECK(non_owning_move_to.size() == 0);

    non_owning_move_to = std::move(non_owning_move_from);
    CHECK(non_owning_move_from.is_owning());
    CHECK(non_owning_move_from.size() == 0);
    CHECK(not non_owning_move_to.is_owning());
    CHECK(non_owning_move_to ==
          Vars{number_of_grid_points2, initial_fill_values[1]});
    CHECK(&get<TestHelpers::Tags::Vector<VectorType>>(
              non_owning_move_to)[0][0] == non_owning_data_from.data());

    // Intentionally testing self-move
#if defined(__clang__) or __GNUC__ >= 13
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wself-move"
#endif
    non_owning_move_to = std::move(non_owning_move_to);
#if defined(__clang__) or __GNUC__ >= 13
#pragma GCC diagnostic pop
#endif
    // clang-tidy: false positive 'move_to' used after it was moved
    CHECK(non_owning_move_to ==  // NOLINT
          Vars{number_of_grid_points2, initial_fill_values[1]});
    CHECK(&get<TestHelpers::Tags::Vector<VectorType>>(
              non_owning_move_to)[0][0] == non_owning_data_from.data());
    CHECK(not non_owning_move_to.is_owning());

    // Test set_data_ref case that wasn't implicitly tested above.
    Vars non_owning_set_data{};
    non_owning_set_data.set_data_ref(make_not_null(&non_owning_move_to));
    CHECK(non_owning_set_data.data() == non_owning_data_from.data());
    CHECK(&get<TestHelpers::Tags::Vector<VectorType>>(
              non_owning_set_data)[0][0] == non_owning_data_from.data());
    CHECK(not non_owning_set_data.is_owning());
  }

  Vars move_to{number_of_grid_points1, initial_fill_values[0]};
  Vars move_from{number_of_grid_points2, initial_fill_values[1]};

  Vars initializer_move_to = std::move(move_to);
  CHECK(initializer_move_to ==
        Vars{number_of_grid_points1, initial_fill_values[0]});
  CHECK(&get<TestHelpers::Tags::Vector<VectorType>>(
            initializer_move_to)[0][0] == initializer_move_to.data());

  move_to = std::move(move_from);
  CHECK(move_to == Vars{number_of_grid_points2, initial_fill_values[1]});
  CHECK(&get<TestHelpers::Tags::Vector<VectorType>>(move_to)[0][0] ==
        move_to.data());

  // Intentionally testing self-move
#if defined(__clang__) or __GNUC__ >= 13
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wself-move"
#endif
  move_to = std::move(move_to);
#if defined(__clang__) or __GNUC__ >= 13
#pragma GCC diagnostic pop
#endif
  // clang-tidy: false positive 'move_to' used after it was moved
  CHECK(move_to ==  // NOLINT
        Vars{number_of_grid_points2, initial_fill_values[1]});
  CHECK(&get<TestHelpers::Tags::Vector<VectorType>>(move_to)[0][0] ==
        move_to.data());

  move_to = Vars{number_of_grid_points2, initial_fill_values[0]};
  CHECK(move_to == Vars{number_of_grid_points2, initial_fill_values[0]});
  CHECK(&get<TestHelpers::Tags::Vector<VectorType>>(move_to)[0][0] ==
        move_to.data());

  move_to = Vars{};
  CHECK(move_to.size() == 0);
  Vars default_for_construction{};
  const auto constructed_from_default = std::move(default_for_construction);
  CHECK(constructed_from_default.size() == 0);

  {
    Vars reuse_after_move{number_of_grid_points1, initial_fill_values[0]};
    { Vars constructed{std::move(reuse_after_move)}; }
    // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
    reuse_after_move.initialize(number_of_grid_points1, initial_fill_values[1]);
    CHECK(reuse_after_move ==
          Vars{number_of_grid_points1, initial_fill_values[1]});
    CHECK(get<0>(get<TestHelpers::Tags::Vector<VectorType>>(
              reuse_after_move))[0] == initial_fill_values[1]);

    {
      Vars assigned{};
      assigned = std::move(reuse_after_move);
    }
    // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
    reuse_after_move.initialize(number_of_grid_points1, initial_fill_values[0]);
    CHECK(reuse_after_move ==
          Vars{number_of_grid_points1, initial_fill_values[0]});
    CHECK(get<0>(get<TestHelpers::Tags::Vector<VectorType>>(
              reuse_after_move))[0] == initial_fill_values[0]);
  }

  {
    Vars size_one{1, initial_fill_values[0]};
    Vars size_two{2, initial_fill_values[0]};
    const value_type* size_one_data = size_one.data();
    const value_type* size_two_data = size_two.data();
    Vars size_one_moved(std::move(size_one));
    Vars size_two_moved(std::move(size_two));
    CHECK(size_one_moved.data() != size_one_data);
    CHECK(size_two_moved.data() == size_two_data);
  }
}

template <typename VectorType>
void test_variables_math() {
  using value_type = typename VectorType::value_type;
  using TestVariablesType =
      Variables<tmpl::list<TestHelpers::Tags::Vector<VectorType>,
                           TestHelpers::Tags::Scalar<VectorType>,
                           TestHelpers::Tags::Scalar2<VectorType>>>;

  MAKE_GENERATOR(gen);
  // keep distribution somewhat near 1.0 to avoid accumulation of errors in
  // cascaded math
  UniformCustomDistribution<tt::get_fundamental_type_t<value_type>> dist{1.0,
                                                                         10.0};
  UniformCustomDistribution<size_t> sdist{1, 5};

  const size_t num_points = sdist(gen);  // number of grid points
  const auto value_in_variables = make_with_random_values<value_type>(
      make_not_null(&gen), make_not_null(&dist));
  const auto rand_vals = make_with_random_values<std::array<value_type, 3>>(
      make_not_null(&gen), make_not_null(&dist));

  // Test math +, -, *, /
  const auto test_close = [&num_points, &value_in_variables](
                              const TestVariablesType& vars,
                              const TestVariablesType& close) {
    CHECK(close != vars);
    CHECK_FALSE(close == vars);
    CHECK_VARIABLES_APPROX(close, vars);
    const TestVariablesType equivalent{num_points, value_in_variables};
    CHECK(equivalent == vars);
    CHECK_FALSE(equivalent != vars);
  };

  std::vector<value_type> owning_data(
      num_points * TestVariablesType::number_of_independent_components,
      value_in_variables);
  const TestVariablesType owning_vars{num_points, value_in_variables};
  const TestVariablesType non_owning_vars{owning_data.data(),
                                          owning_data.size()};

  {
    const TestVariablesType close{num_points,
                                  value_in_variables * (1.0 + 1.0e-14)};
    const TestVariablesType non_owning_close{
        const_cast<value_type*>(close.data()), close.size()};
    test_close(owning_vars, close);
    test_close(non_owning_vars, close);
    test_close(owning_vars, non_owning_close);
    test_close(non_owning_vars, non_owning_close);
  }

  TestVariablesType expected{num_points, rand_vals.at(0) * value_in_variables};

  const auto test_math = [&expected, &num_points, &rand_vals,
                          &value_in_variables](const TestVariablesType& vars) {
    expected =
        TestVariablesType{num_points, rand_vals.at(0) * value_in_variables};
    CHECK_VARIABLES_APPROX(expected, rand_vals[0] * vars);
    expected = TestVariablesType{num_points, value_in_variables * rand_vals[0]};
    CHECK_VARIABLES_APPROX(expected, vars * rand_vals[0]);
    expected = TestVariablesType{num_points, value_in_variables / rand_vals[0]};
    CHECK_VARIABLES_APPROX(expected, vars / rand_vals[0]);

    expected = TestVariablesType{num_points, 4. * value_in_variables};
    CHECK_VARIABLES_APPROX(expected, vars + vars + vars + vars);
    expected = TestVariablesType{num_points, 3. * value_in_variables};
    CHECK_VARIABLES_APPROX(expected, vars + (vars + vars));
    // clang-tidy: both sides of overloaded operator are equivalent
    expected = TestVariablesType{num_points, -2. * value_in_variables};
    CHECK_VARIABLES_APPROX(expected, vars - vars - vars - vars);  // NOLINT
    expected = TestVariablesType{num_points, value_in_variables};
    CHECK_VARIABLES_APPROX(expected,
                           vars - (vars - vars));  // NOLINT
    expected = TestVariablesType{num_points, -value_in_variables};
    CHECK_VARIABLES_APPROX(expected, vars - (vars + vars));
    expected = TestVariablesType{num_points, value_in_variables};
    CHECK_VARIABLES_APPROX(expected,
                           vars - vars + vars);  // NOLINT
    expected = TestVariablesType{num_points, value_in_variables};
    CHECK_VARIABLES_APPROX(expected, vars + vars - vars);
    expected = TestVariablesType{num_points, -value_in_variables};
    CHECK_VARIABLES_APPROX(expected, -vars);
    expected = TestVariablesType{num_points, value_in_variables};
    CHECK_VARIABLES_APPROX(expected, +vars);
  };

  test_math(owning_vars);
  test_math(non_owning_vars);

  // Test math_assignment operators +=, -=, *=, /= with values
  const auto assignment_test =
      [&expected, &num_points, &rand_vals,
       &value_in_variables](TestVariablesType& test_assignment) {
        auto expected_val = value_in_variables * rand_vals[0];
        test_assignment += TestVariablesType{num_points, value_in_variables};
        expected_val += value_in_variables;
        expected = TestVariablesType{num_points, expected_val};
        CHECK_VARIABLES_APPROX(expected, test_assignment);
        test_assignment -= TestVariablesType{num_points, rand_vals[1]};
        expected_val -= rand_vals[1];
        expected = TestVariablesType{num_points, expected_val};
        CHECK_VARIABLES_APPROX(expected, test_assignment);
        test_assignment *= rand_vals[2];
        expected_val *= rand_vals[2];
        expected = TestVariablesType{num_points, expected_val};
        CHECK_VARIABLES_APPROX(expected, test_assignment);
        test_assignment /= rand_vals[0];
        expected_val /= rand_vals[0];
        expected = TestVariablesType{num_points, expected_val};
        CHECK_VARIABLES_APPROX(expected, test_assignment);

        test_assignment +=
            TestVariablesType{num_points, value_in_variables} * rand_vals[1];
        expected_val += value_in_variables * rand_vals[1];
        expected = TestVariablesType{num_points, expected_val};
        CHECK_VARIABLES_APPROX(expected, test_assignment);
        test_assignment -=
            TestVariablesType{num_points, rand_vals[2]} * rand_vals[0];
        expected_val -= rand_vals[2] * rand_vals[0];
        expected = TestVariablesType{num_points, expected_val};
        CHECK_VARIABLES_APPROX(expected, test_assignment);

        TestVariablesType test_assignment2{num_points};
        test_assignment2 = test_assignment * 1.0;
        CHECK(test_assignment2 == test_assignment);
        test_assignment2 = test_assignment2 * 1.0;
        CHECK(test_assignment2 == test_assignment);
      };

  TestVariablesType owning_test_assignment(owning_vars * rand_vals[0]);
  std::vector<value_type> owning_data_assignment(
      num_points * TestVariablesType::number_of_independent_components,
      value_in_variables * rand_vals[0]);
  TestVariablesType non_owning_test_assignment(owning_data_assignment.data(),
                                               owning_data_assignment.size());
  assignment_test(owning_test_assignment);
  assignment_test(non_owning_test_assignment);
  CHECK(non_owning_test_assignment.data() == owning_data_assignment.data());

  const auto check_components = [](const auto& variables,
                                   const VectorType& tensor) {
    tmpl::for_each<typename std::decay_t<decltype(variables)>::tags_list>(
        [&variables, &tensor](auto tag) {
          using Tag = tmpl::type_from<decltype(tag)>;
          for (const auto& component : get<Tag>(variables)) {
            CHECK_ITERABLE_APPROX(component, tensor);
          }
        });
  };

  const auto test_vector = make_with_random_values<VectorType>(
      make_not_null(&gen), make_not_null(&dist), VectorType{4});

  // Test math assignment operators +=, -=, *=, /= with vectors
  const auto vector_math_test = [&check_components, &test_vector, &rand_vals](
                                    TestVariablesType& test_vector_math) {
    test_vector_math *= test_vector;
    VectorType vec_expect = rand_vals[0] * test_vector;
    check_components(test_vector_math, vec_expect);
    vec_expect *= test_vector;
    check_components(test_vector_math * test_vector, vec_expect);
    check_components(test_vector * test_vector_math, vec_expect);
    test_vector_math *= test_vector;
    check_components(test_vector_math, vec_expect);
    vec_expect /= test_vector;
    check_components(test_vector_math / test_vector, vec_expect);
    test_vector_math /= test_vector;
    check_components(test_vector_math, vec_expect);
  };

  TestVariablesType test_vector_math(4, rand_vals[0]);
  std::vector<value_type> data_test_vector_math(
      4 * TestVariablesType::number_of_independent_components, rand_vals[0]);
  TestVariablesType non_owning_test_vector_math(data_test_vector_math.data(),
                                                data_test_vector_math.size());
  vector_math_test(test_vector_math);
  vector_math_test(non_owning_test_vector_math);
}

template <typename VectorType>
void test_variables_prefix_semantics() {
  using TagList = tmpl::list<TestHelpers::Tags::Vector<VectorType>,
                             TestHelpers::Tags::Scalar<VectorType>,
                             TestHelpers::Tags::Scalar2<VectorType>>;
  using VariablesType = Variables<TagList>;
  using PrefixVariablesType =
      Variables<db::wrap_tags_in<TestHelpers::Tags::Prefix0, TagList>>;
  using value_type = typename VectorType::value_type;

  MAKE_GENERATOR(gen);
  UniformCustomDistribution<tt::get_fundamental_type_t<value_type>> dist{-100.0,
                                                                         100.0};
  UniformCustomDistribution<size_t> sdist{1, 5};

  const size_t number_of_grid_points = sdist(gen);
  const auto variables_vals =
      make_with_random_values<std::array<value_type, 4>>(make_not_null(&gen),
                                                         make_not_null(&dist));

  // Check move and copy from prefix variables to non-prefix variables via
  // constructor
  PrefixVariablesType prefix_vars_copy_construct_from{number_of_grid_points,
                                                      variables_vals[0]};
  PrefixVariablesType prefix_vars_move_construct_from{number_of_grid_points,
                                                      variables_vals[1]};
  VariablesType vars_to{prefix_vars_copy_construct_from};
  VariablesType move_constructed_vars{
      std::move(prefix_vars_move_construct_from)};

  VariablesType expected{number_of_grid_points, variables_vals[0]};
  CHECK_VARIABLES_APPROX(vars_to, expected);
  expected = VariablesType(number_of_grid_points, variables_vals[1]);
  CHECK_VARIABLES_APPROX(move_constructed_vars, expected);

  VariablesType constructed_from_default{PrefixVariablesType{}};
  CHECK(constructed_from_default.size() == 0);
  PrefixVariablesType assigned_from_default{number_of_grid_points};
  assigned_from_default = std::move(constructed_from_default);
  CHECK(assigned_from_default.size() == 0);

  {
    VariablesType non_owning_vars_to{prefix_vars_copy_construct_from};
    // Check move and copy from prefix variables to non-prefix variables via
    // assignment operator
    std::vector<value_type> non_owning_data_copy_from(
        number_of_grid_points *
            PrefixVariablesType::number_of_independent_components,
        variables_vals[2]);
    std::vector<value_type> non_owning_data_move_from(
        number_of_grid_points *
            PrefixVariablesType::number_of_independent_components,
        variables_vals[3]);
    PrefixVariablesType prefix_vars_copy_assign_from{
        non_owning_data_copy_from.data(), non_owning_data_copy_from.size()};
    PrefixVariablesType prefix_vars_move_assign_from{
        non_owning_data_move_from.data(), non_owning_data_move_from.size()};
    non_owning_vars_to = prefix_vars_copy_assign_from;
    CHECK(not prefix_vars_copy_assign_from.is_owning());
    CHECK(non_owning_vars_to.is_owning());
    CHECK_VARIABLES_APPROX(
        non_owning_vars_to,
        VariablesType(number_of_grid_points, variables_vals[2]));
    CHECK_VARIABLES_APPROX(
        prefix_vars_copy_assign_from,
        PrefixVariablesType(number_of_grid_points, variables_vals[2]));

    non_owning_vars_to = std::move(prefix_vars_move_assign_from);
    CHECK(prefix_vars_move_assign_from.is_owning());
    CHECK(prefix_vars_move_assign_from.size() == 0);
    CHECK(not non_owning_vars_to.is_owning());
    CHECK_VARIABLES_APPROX(
        non_owning_vars_to,
        VariablesType(number_of_grid_points, variables_vals[3]));

    // Test construction
    PrefixVariablesType non_owning_prefix_vars_copy_construct_from{
        non_owning_data_copy_from.data(), non_owning_data_copy_from.size()};
    PrefixVariablesType non_owning_prefix_vars_move_construct_from{
        non_owning_data_move_from.data(), non_owning_data_copy_from.size()};
    const VariablesType copy_constructed{
        non_owning_prefix_vars_copy_construct_from};
    CHECK(not non_owning_prefix_vars_copy_construct_from.is_owning());
    CHECK(copy_constructed.is_owning());
    CHECK_VARIABLES_APPROX(
        copy_constructed,
        VariablesType(number_of_grid_points, variables_vals[2]));
    CHECK_VARIABLES_APPROX(
        non_owning_prefix_vars_copy_construct_from,
        PrefixVariablesType(number_of_grid_points, variables_vals[2]));

    const VariablesType move_constructed{
        std::move(non_owning_prefix_vars_move_construct_from)};
    CHECK(non_owning_prefix_vars_move_construct_from.is_owning());
    CHECK(non_owning_prefix_vars_move_construct_from.size() == 0);
    CHECK(non_owning_prefix_vars_move_construct_from.number_of_grid_points() ==
          0);
    CHECK(not move_constructed.is_owning());
    CHECK_VARIABLES_APPROX(
        move_constructed,
        VariablesType(number_of_grid_points, variables_vals[3]));
  }

  // Check move and copy from prefix variables to non-prefix variables via
  // assignment operator
  PrefixVariablesType prefix_vars_copy_assign_from{number_of_grid_points,
                                                   variables_vals[2]};
  PrefixVariablesType prefix_vars_move_assign_from{number_of_grid_points,
                                                   variables_vals[3]};
  vars_to = prefix_vars_copy_assign_from;
  CHECK_VARIABLES_APPROX(
      vars_to, VariablesType(number_of_grid_points, variables_vals[2]));
  vars_to = std::move(prefix_vars_move_assign_from);
  CHECK_VARIABLES_APPROX(
      vars_to, VariablesType(number_of_grid_points, variables_vals[3]));

  // Test swap with prefix
  PrefixVariablesType prefix_vars_swap{number_of_grid_points};
  VariablesType vars_swap{2 * number_of_grid_points};
  fill_with_random_values(make_not_null(&prefix_vars_swap), make_not_null(&gen),
                          make_not_null(&dist));
  fill_with_random_values(make_not_null(&vars_swap), make_not_null(&gen),
                          make_not_null(&dist));
  const PrefixVariablesType expected_vars_swap(prefix_vars_swap);
  const VariablesType expected_prefix_vars_swap(vars_swap);
  swap(vars_swap, prefix_vars_swap);
  CHECK(
      gsl::span<const value_type>(expected_vars_swap.data(),
                                  2 * number_of_grid_points) ==
      gsl::span<const value_type>(vars_swap.data(), 2 * number_of_grid_points));
  CHECK(gsl::span<const value_type>(expected_prefix_vars_swap.data(),
                                    number_of_grid_points) ==
        gsl::span<const value_type>(prefix_vars_swap.data(),
                                    number_of_grid_points));

  {
    VariablesType reuse_after_move{number_of_grid_points, 2.0};
    { PrefixVariablesType constructed{std::move(reuse_after_move)}; }
    // NOLINTNEXTLINE(bugprone-use-after-move)
    reuse_after_move.initialize(number_of_grid_points, 3.0);
    CHECK(reuse_after_move == VariablesType{number_of_grid_points, 3.0});
    CHECK(get<0>(get<TestHelpers::Tags::Vector<VectorType>>(
              reuse_after_move))[0] == 3.0);

    {
      PrefixVariablesType assigned{};
      assigned = std::move(reuse_after_move);
    }
    // NOLINTNEXTLINE(bugprone-use-after-move)
    reuse_after_move.initialize(number_of_grid_points, 4.0);
    CHECK(reuse_after_move == VariablesType{number_of_grid_points, 4.0});
    CHECK(get<0>(get<TestHelpers::Tags::Vector<VectorType>>(
              reuse_after_move))[0] == 4.0);
  }
}

template <typename VectorType>
void test_variables_prefix_math() {
  using TagList = tmpl::list<TestHelpers::Tags::Vector<VectorType>,
                             TestHelpers::Tags::Scalar<VectorType>,
                             TestHelpers::Tags::Scalar2<VectorType>>;
  using Prefix0VariablesType =
      Variables<db::wrap_tags_in<TestHelpers::Tags::Prefix0, TagList>>;
  using Prefix1VariablesType =
      Variables<db::wrap_tags_in<TestHelpers::Tags::Prefix1, TagList>>;
  using Prefix2VariablesType =
      Variables<db::wrap_tags_in<TestHelpers::Tags::Prefix2, TagList>>;
  using Prefix3VariablesType =
      Variables<db::wrap_tags_in<TestHelpers::Tags::Prefix3, TagList>>;
  using value_type = typename VectorType::value_type;

  MAKE_GENERATOR(gen);
  // keep distribution somewhat near 1.0 to avoid accumulation of errors in
  // cascaded math
  UniformCustomDistribution<tt::get_fundamental_type_t<value_type>> dist{1.0,
                                                                         10.0};
  UniformCustomDistribution<size_t> sdist{1, 5};

  const size_t number_of_grid_points = sdist(gen);
  const auto value_in_variables = make_with_random_values<value_type>(
      make_not_null(&gen), make_not_null(&dist));
  const auto random_vals = make_with_random_values<std::array<value_type, 3>>(
      make_not_null(&gen), make_not_null(&dist));

  // Test arithmetic operators +, -, *, / for prefix Variables.

  // We wish to verify that operations among multiple Variables with distinct
  // prefixes give a Variables populated with the desired data. This is
  // important for multiplying, for instance a variables populated with time
  // derivatives with a variables populated with time intervals, which would
  // have the same types, but have distinct prefixes.
  Prefix0VariablesType prefix_vars0(number_of_grid_points, value_in_variables);
  Prefix1VariablesType prefix_vars1(number_of_grid_points, value_in_variables);
  Prefix2VariablesType prefix_vars2(number_of_grid_points, value_in_variables);
  Prefix3VariablesType prefix_vars3(number_of_grid_points, value_in_variables);

  Prefix3VariablesType expected{number_of_grid_points,
                                value_in_variables * random_vals[0]};
  CHECK_VARIABLES_APPROX(expected, random_vals[0] * prefix_vars0);
  expected = Prefix3VariablesType{number_of_grid_points,
                                  value_in_variables * random_vals[0]};
  CHECK_VARIABLES_APPROX(expected, prefix_vars0 * random_vals[0]);
  expected = Prefix3VariablesType{number_of_grid_points,
                                  value_in_variables / random_vals[0]};
  CHECK_VARIABLES_APPROX(expected, prefix_vars0 / random_vals[0]);
  expected =
      Prefix3VariablesType{number_of_grid_points, 4.0 * value_in_variables};
  CHECK_VARIABLES_APPROX(
      expected, prefix_vars0 + prefix_vars1 + prefix_vars2 + prefix_vars3);
  expected =
      Prefix3VariablesType{number_of_grid_points, 4.0 * value_in_variables};
  CHECK_VARIABLES_APPROX(
      expected, (prefix_vars0 + prefix_vars1) + (prefix_vars2 + prefix_vars3));
  expected =
      Prefix3VariablesType{number_of_grid_points, 3.0 * value_in_variables};

  CHECK_VARIABLES_APPROX(expected,
                         prefix_vars0 + (prefix_vars1 + prefix_vars2));

  // clang-tidy: both sides of overloaded operator are equivalent
  expected =
      Prefix3VariablesType{number_of_grid_points, -2.0 * value_in_variables};
  CHECK_VARIABLES_APPROX(
      expected,
      prefix_vars0 - prefix_vars1 - prefix_vars2 - prefix_vars3);  // NOLINT
  expected =
      Prefix3VariablesType{number_of_grid_points, 2.0 * value_in_variables};
  CHECK_VARIABLES_APPROX(
      expected,
      prefix_vars0 - (prefix_vars1 - prefix_vars2 - prefix_vars3));  // NOLINT
  expected =
      Prefix3VariablesType{number_of_grid_points, 0.0 * value_in_variables};
  CHECK_VARIABLES_APPROX(
      expected,
      (prefix_vars0 - prefix_vars1) - (prefix_vars2 - prefix_vars3));  // NOLINT
  expected =
      Prefix3VariablesType{number_of_grid_points, 1.0 * value_in_variables};
  CHECK_VARIABLES_APPROX(
      expected, prefix_vars0 - (prefix_vars1 - prefix_vars2));  // NOLINT
  expected =
      Prefix3VariablesType{number_of_grid_points, 1.0 * value_in_variables};
  CHECK_VARIABLES_APPROX(expected,
                         prefix_vars0 - prefix_vars1 + prefix_vars2);  // NOLINT
  expected =
      Prefix3VariablesType{number_of_grid_points, 1.0 * value_in_variables};
  CHECK_VARIABLES_APPROX(expected, prefix_vars0 + prefix_vars1 - prefix_vars2);
  expected = Prefix3VariablesType{number_of_grid_points, -value_in_variables};
  CHECK_VARIABLES_APPROX(expected, -prefix_vars0);
  expected = Prefix3VariablesType{number_of_grid_points, value_in_variables};
  CHECK_VARIABLES_APPROX(expected, +prefix_vars0);

  // Test assignment arithmetic operators +=, -=, *=, /= with values
  Prefix0VariablesType test_assignment(prefix_vars0 * 1.0);
  test_assignment +=
      Prefix1VariablesType{number_of_grid_points, value_in_variables};
  auto expected_val = 2.0 * value_in_variables;
  Prefix0VariablesType expected_prefix0{number_of_grid_points, expected_val};
  CHECK_VARIABLES_APPROX(expected_prefix0, test_assignment);
  test_assignment -=
      Prefix1VariablesType{number_of_grid_points, random_vals[0]};
  expected_val -= random_vals[0];
  expected_prefix0 = Prefix0VariablesType{number_of_grid_points, expected_val};
  CHECK_VARIABLES_APPROX(expected_prefix0, test_assignment);
  test_assignment *= random_vals[1];
  expected_val *= random_vals[1];
  expected_prefix0 = Prefix0VariablesType{number_of_grid_points, expected_val};
  CHECK_VARIABLES_APPROX(expected_prefix0, test_assignment);
  test_assignment /= random_vals[2];
  expected_val /= random_vals[2];
  expected_prefix0 = Prefix0VariablesType{number_of_grid_points, expected_val};
  CHECK_VARIABLES_APPROX(expected_prefix0, test_assignment);

  test_assignment +=
      Prefix1VariablesType{number_of_grid_points, random_vals[0]} *
      random_vals[1];
  expected_val += random_vals[0] * random_vals[1];
  expected_prefix0 = Prefix0VariablesType{number_of_grid_points, expected_val};
  CHECK_VARIABLES_APPROX(expected_prefix0, test_assignment);
  test_assignment -=
      Prefix1VariablesType{number_of_grid_points, random_vals[0]} *
      random_vals[2];
  expected_val -= random_vals[0] * random_vals[2];
  expected_prefix0 = Prefix0VariablesType{number_of_grid_points, expected_val};
  CHECK_VARIABLES_APPROX(expected_prefix0, test_assignment);
}

template <typename VectorType>
void test_variables_serialization() {
  using value_type = typename VectorType::value_type;
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<tt::get_fundamental_type_t<value_type>> dist{-100.0,
                                                                         100.0};
  UniformCustomDistribution<size_t> sdist{1, 5};

  const size_t number_of_grid_points = sdist(gen);
  const auto values_in_variables =
      make_with_random_values<std::array<value_type, 2>>(make_not_null(&gen),
                                                         make_not_null(&dist));

  // Test serialization of a Variables and a tuple of Variables
  Variables<tmpl::list<TestHelpers::Tags::Vector<VectorType>>> test_variables(
      number_of_grid_points, values_in_variables[0]);
  test_serialization(test_variables);
  auto tuple_of_test_variables = std::make_tuple(
      Variables<tmpl::list<TestHelpers::Tags::Vector<VectorType>>>{
          number_of_grid_points, values_in_variables[1]});
  test_serialization(tuple_of_test_variables);
}

template <typename VectorType>
void test_variables_assign_subset() {
  using value_type = typename VectorType::value_type;
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<tt::get_fundamental_type_t<value_type>> dist{-100.0,
                                                                         100.0};
  UniformCustomDistribution<size_t> sdist{1, 5};

  const size_t number_of_grid_points = sdist(gen);
  const auto values_in_variables =
      make_with_random_values<std::array<value_type, 4>>(make_not_null(&gen),
                                                         make_not_null(&dist));

  // Test assigning a single tag to a Variables with two tags, either from a
  // Variables or a TaggedTuple
  const auto test_assign_to_vars_with_two_tags = [&number_of_grid_points,
                                                  &values_in_variables](
                                                     const auto& vars_subset0,
                                                     const auto& vars_subset1,
                                                     const value_type&
                                                         vars_subset0_val,
                                                     const value_type&
                                                         vars_subset1_val) {
    Variables<tmpl::list<TestHelpers::Tags::Vector<VectorType>,
                         TestHelpers::Tags::Scalar<VectorType>>>
        vars_set{number_of_grid_points, values_in_variables[0]};
    CHECK(
        get<TestHelpers::Tags::Vector<VectorType>>(vars_set) ==
        tnsr::I<VectorType, 3>(number_of_grid_points, values_in_variables[0]));
    CHECK(get<TestHelpers::Tags::Scalar<VectorType>>(vars_set) ==
          ::Scalar<VectorType>(number_of_grid_points, values_in_variables[0]));
    vars_set.assign_subset(vars_subset0);
    CHECK(get<TestHelpers::Tags::Vector<VectorType>>(vars_set) ==
          tnsr::I<VectorType, 3>(number_of_grid_points, vars_subset0_val));
    CHECK(get<TestHelpers::Tags::Scalar<VectorType>>(vars_set) ==
          ::Scalar<VectorType>(number_of_grid_points, values_in_variables[0]));
    vars_set.assign_subset(vars_subset1);
    CHECK(get<TestHelpers::Tags::Vector<VectorType>>(vars_set) ==
          tnsr::I<VectorType, 3>(number_of_grid_points, vars_subset0_val));
    CHECK(get<TestHelpers::Tags::Scalar<VectorType>>(vars_set) ==
          ::Scalar<VectorType>(number_of_grid_points, vars_subset1_val));
  };

  test_assign_to_vars_with_two_tags(
      Variables<tmpl::list<TestHelpers::Tags::Vector<VectorType>>>(
          number_of_grid_points, values_in_variables[1]),
      Variables<tmpl::list<TestHelpers::Tags::Scalar<VectorType>>>(
          number_of_grid_points, values_in_variables[2]),
      values_in_variables[1], values_in_variables[2]);
  test_assign_to_vars_with_two_tags(
      tuples::TaggedTuple<TestHelpers::Tags::Vector<VectorType>>(
          typename TestHelpers::Tags::Vector<VectorType>::type{
              number_of_grid_points, values_in_variables[1]}),
      tuples::TaggedTuple<TestHelpers::Tags::Scalar<VectorType>>(
          typename TestHelpers::Tags::Scalar<VectorType>::type{
              number_of_grid_points, values_in_variables[2]}),
      values_in_variables[1], values_in_variables[2]);

  test_assign_to_vars_with_two_tags(
      Variables<tmpl::list<TestHelpers::Tags::Vector<VectorType>,
                           TestHelpers::Tags::OneForm<VectorType>>>(
          number_of_grid_points, values_in_variables[1]),
      Variables<tmpl::list<TestHelpers::Tags::Scalar<VectorType>,
                           TestHelpers::Tags::OneForm<VectorType>>>(
          number_of_grid_points, values_in_variables[2]),
      values_in_variables[1], values_in_variables[2]);
  test_assign_to_vars_with_two_tags(
      tuples::TaggedTuple<TestHelpers::Tags::Vector<VectorType>,
                          TestHelpers::Tags::OneForm<VectorType>>(
          typename TestHelpers::Tags::Vector<VectorType>::type{
              number_of_grid_points, values_in_variables[1]},
          typename TestHelpers::Tags::OneForm<VectorType>::type{
              number_of_grid_points, values_in_variables[0]}),
      tuples::TaggedTuple<TestHelpers::Tags::Scalar<VectorType>,
                          TestHelpers::Tags::OneForm<VectorType>>(
          typename TestHelpers::Tags::Scalar<VectorType>::type{
              number_of_grid_points, values_in_variables[2]},
          typename TestHelpers::Tags::OneForm<VectorType>::type{
              number_of_grid_points, values_in_variables[0]}),
      values_in_variables[1], values_in_variables[2]);

  // Test assigning a single tag to a Variables with three tags, either from a
  // Variables or a TaggedTuple
  const auto test_assign_to_vars_with_three_tags = [&number_of_grid_points,
                                                    &values_in_variables](
                                                       const auto& vars_subset0,
                                                       const auto& vars_subset1,
                                                       const value_type&
                                                           vars_subset0_val,
                                                       const value_type&
                                                           vars_subset1_val) {
    Variables<tmpl::list<TestHelpers::Tags::Vector<VectorType>,
                         TestHelpers::Tags::Scalar<VectorType>,
                         TestHelpers::Tags::Scalar2<VectorType>>>
        vars_set(number_of_grid_points, values_in_variables[0]);
    CHECK(
        get<TestHelpers::Tags::Vector<VectorType>>(vars_set) ==
        tnsr::I<VectorType, 3>(number_of_grid_points, values_in_variables[0]));
    CHECK(get<TestHelpers::Tags::Scalar<VectorType>>(vars_set) ==
          ::Scalar<VectorType>(number_of_grid_points, values_in_variables[0]));
    CHECK(get<TestHelpers::Tags::Scalar2<VectorType>>(vars_set) ==
          ::Scalar<VectorType>(number_of_grid_points, values_in_variables[0]));
    vars_set.assign_subset(vars_subset0);
    CHECK(get<TestHelpers::Tags::Vector<VectorType>>(vars_set) ==
          tnsr::I<VectorType, 3>(number_of_grid_points, vars_subset0_val));
    CHECK(get<TestHelpers::Tags::Scalar<VectorType>>(vars_set) ==
          ::Scalar<VectorType>(number_of_grid_points, values_in_variables[0]));
    CHECK(get<TestHelpers::Tags::Scalar2<VectorType>>(vars_set) ==
          ::Scalar<VectorType>(number_of_grid_points, values_in_variables[0]));
    vars_set.assign_subset(vars_subset1);
    CHECK(get<TestHelpers::Tags::Vector<VectorType>>(vars_set) ==
          tnsr::I<VectorType, 3>(number_of_grid_points, vars_subset0_val));
    CHECK(get<TestHelpers::Tags::Scalar<VectorType>>(vars_set) ==
          ::Scalar<VectorType>(number_of_grid_points, vars_subset1_val));
    CHECK(get<TestHelpers::Tags::Scalar2<VectorType>>(vars_set) ==
          ::Scalar<VectorType>(number_of_grid_points, values_in_variables[0]));
  };

  test_assign_to_vars_with_three_tags(
      Variables<tmpl::list<TestHelpers::Tags::Vector<VectorType>>>(
          number_of_grid_points, values_in_variables[1]),
      Variables<tmpl::list<TestHelpers::Tags::Scalar<VectorType>>>(
          number_of_grid_points, values_in_variables[2]),
      values_in_variables[1], values_in_variables[2]);
  test_assign_to_vars_with_three_tags(
      tuples::TaggedTuple<TestHelpers::Tags::Vector<VectorType>>(
          typename TestHelpers::Tags::Vector<VectorType>::type{
              number_of_grid_points, values_in_variables[1]}),
      tuples::TaggedTuple<TestHelpers::Tags::Scalar<VectorType>>(
          typename TestHelpers::Tags::Scalar<VectorType>::type{
              number_of_grid_points, values_in_variables[2]}),
      values_in_variables[1], values_in_variables[2]);

  test_assign_to_vars_with_three_tags(
      Variables<tmpl::list<TestHelpers::Tags::Vector<VectorType>,
                           TestHelpers::Tags::OneForm<VectorType>>>(
          number_of_grid_points, values_in_variables[1]),
      Variables<tmpl::list<TestHelpers::Tags::Scalar<VectorType>,
                           TestHelpers::Tags::OneForm<VectorType>>>(
          number_of_grid_points, values_in_variables[2]),
      values_in_variables[1], values_in_variables[2]);
  test_assign_to_vars_with_three_tags(
      tuples::TaggedTuple<TestHelpers::Tags::Vector<VectorType>,
                          TestHelpers::Tags::OneForm<VectorType>>(
          typename TestHelpers::Tags::Vector<VectorType>::type{
              number_of_grid_points, values_in_variables[1]},
          typename TestHelpers::Tags::OneForm<VectorType>::type{
              number_of_grid_points, values_in_variables[0]}),
      tuples::TaggedTuple<TestHelpers::Tags::Scalar<VectorType>,
                          TestHelpers::Tags::OneForm<VectorType>>(
          typename TestHelpers::Tags::Scalar<VectorType>::type{
              number_of_grid_points, values_in_variables[2]},
          typename TestHelpers::Tags::OneForm<VectorType>::type{
              number_of_grid_points, values_in_variables[0]}),
      values_in_variables[1], values_in_variables[2]);

  // Test assignment to a Variables with a single tag either from a Variables or
  // a TaggedTuple
  const auto test_assign_to_vars_with_one_tag =
      [&number_of_grid_points, &values_in_variables](
          const auto& vars_subset0, const value_type& vars_subset0_val) {
        Variables<tmpl::list<TestHelpers::Tags::Vector<VectorType>>> vars_set(
            number_of_grid_points, values_in_variables[0]);
        CHECK(get<TestHelpers::Tags::Vector<VectorType>>(vars_set) ==
              tnsr::I<VectorType, 3>(number_of_grid_points,
                                     values_in_variables[0]));
        vars_set.assign_subset(vars_subset0);
        CHECK(get<TestHelpers::Tags::Vector<VectorType>>(vars_set) ==
              tnsr::I<VectorType, 3>(number_of_grid_points, vars_subset0_val));
      };

  test_assign_to_vars_with_one_tag(
      Variables<tmpl::list<TestHelpers::Tags::Vector<VectorType>>>(
          number_of_grid_points, values_in_variables[1]),
      values_in_variables[1]);
  test_assign_to_vars_with_one_tag(
      tuples::TaggedTuple<TestHelpers::Tags::Vector<VectorType>>(
          typename TestHelpers::Tags::Vector<VectorType>::type{
              number_of_grid_points, values_in_variables[1]}),
      values_in_variables[1]);

  test_assign_to_vars_with_one_tag(
      Variables<tmpl::list<TestHelpers::Tags::Vector<VectorType>,
                           TestHelpers::Tags::OneForm<VectorType>>>(
          number_of_grid_points, values_in_variables[1]),
      values_in_variables[1]);
  test_assign_to_vars_with_one_tag(
      tuples::TaggedTuple<TestHelpers::Tags::Vector<VectorType>,
                          TestHelpers::Tags::OneForm<VectorType>>(
          typename TestHelpers::Tags::Vector<VectorType>::type{
              number_of_grid_points, values_in_variables[1]},
          typename TestHelpers::Tags::OneForm<VectorType>::type{
              number_of_grid_points, values_in_variables[0]}),
      values_in_variables[1]);
}

template <typename VectorType>
void test_variables_extract_subset() {
  using value_type = typename VectorType::value_type;
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<tt::get_fundamental_type_t<value_type>> dist{-100.0,
                                                                         100.0};
  UniformCustomDistribution<size_t> sdist{1, 5};

  const size_t number_of_grid_points = sdist(gen);

  const auto vars_subset0 = make_with_random_values<
      Variables<tmpl::list<TestHelpers::Tags::Vector<VectorType>>>>(
      make_not_null(&gen), make_not_null(&dist),
      VectorType{number_of_grid_points});
  const auto vars_subset1 = make_with_random_values<
      Variables<tmpl::list<TestHelpers::Tags::Scalar<VectorType>>>>(
      make_not_null(&gen), make_not_null(&dist),
      VectorType{number_of_grid_points});

  Variables<tmpl::list<TestHelpers::Tags::Vector<VectorType>,
                       TestHelpers::Tags::Scalar<VectorType>>>
      vars(number_of_grid_points);
  get<TestHelpers::Tags::Vector<VectorType>>(vars) =
      get<TestHelpers::Tags::Vector<VectorType>>(vars_subset0);
  get<TestHelpers::Tags::Scalar<VectorType>>(vars) =
      get<TestHelpers::Tags::Scalar<VectorType>>(vars_subset1);
  CHECK(vars.template extract_subset<
            tmpl::list<TestHelpers::Tags::Vector<VectorType>>>() ==
        vars_subset0);
  CHECK(vars.template extract_subset<
            tmpl::list<TestHelpers::Tags::Scalar<VectorType>>>() ==
        vars_subset1);
  CHECK(vars.template extract_subset<
            tmpl::list<TestHelpers::Tags::Vector<VectorType>,
                       TestHelpers::Tags::Scalar<VectorType>>>() == vars);
}

template <typename VectorType>
void test_variables_reference_subset() {
  using value_type = typename VectorType::value_type;
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<tt::get_fundamental_type_t<value_type>> dist{-100.0,
                                                                         100.0};
  UniformCustomDistribution<size_t> sdist{1, 5};

  using FullVars = Variables<tmpl::list<TestHelpers::Tags::Vector<VectorType>,
                                        TestHelpers::Tags::Scalar<VectorType>>>;
  const size_t number_of_grid_points = sdist(gen);

  auto full_vars = make_with_random_values<FullVars>(
      make_not_null(&gen), make_not_null(&dist), number_of_grid_points);
  const auto check_reference = [&](auto tags) {
    const auto reference_vars =
        full_vars.template reference_subset<decltype(tags)>();
    CHECK(not reference_vars.is_owning());
    CHECK(reference_vars ==
          full_vars.template extract_subset<decltype(tags)>());
  };
  check_reference(typename FullVars::tags_list{});
  check_reference(tmpl::list<TestHelpers::Tags::Vector<VectorType>>{});
  check_reference(tmpl::list<TestHelpers::Tags::Scalar<VectorType>>{});
  {
    auto empty_reference = full_vars.template reference_subset<tmpl::list<>>();
    static_assert(
        std::is_same_v<decltype(empty_reference), Variables<tmpl::list<>>>);
  }
}

template <typename VectorType>
void test_variables_reference_with_different_prefixes() {
  using value_type = typename VectorType::value_type;
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<tt::get_fundamental_type_t<value_type>> dist{-100.0,
                                                                         100.0};
  UniformCustomDistribution<size_t> sdist{1, 5};

  const size_t number_of_grid_points = sdist(gen);

  using Variables1 = Variables<tmpl::list<
      TestHelpers::Tags::Vector<VectorType>,
      TestHelpers::Tags::Prefix0<TestHelpers::Tags::Scalar<VectorType>>>>;
  using Variables2 = Variables<tmpl::list<
      TestHelpers::Tags::Prefix0<TestHelpers::Tags::Vector<VectorType>>,
      TestHelpers::Tags::Scalar<VectorType>>>;

  auto owning = make_with_random_values<Variables1>(
      make_not_null(&gen), make_not_null(&dist), number_of_grid_points);
  auto non_owning =
      owning.template reference_with_different_prefixes<Variables2>();
  CHECK(owning.size() == non_owning.size());
  CHECK(owning.data() == non_owning.data());
}

template <typename VectorType>
void test_variables_from_tagged_tuple() {
  using value_type = typename VectorType::value_type;
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<tt::get_fundamental_type_t<value_type>> dist{-100.0,
                                                                         100.0};
  UniformCustomDistribution<size_t> sdist{1, 5};

  const size_t number_of_grid_points = sdist(gen);
  tuples::TaggedTuple<TestHelpers::Tags::Vector<VectorType>,
                      TestHelpers::Tags::Scalar<VectorType>>
      source;
  get<TestHelpers::Tags::Vector<VectorType>>(source) = make_with_random_values<
      typename TestHelpers::Tags::Vector<VectorType>::type>(
      make_not_null(&gen), make_not_null(&dist),
      VectorType{number_of_grid_points});
  get<TestHelpers::Tags::Scalar<VectorType>>(source) = make_with_random_values<
      typename TestHelpers::Tags::Scalar<VectorType>::type>(
      make_not_null(&gen), make_not_null(&dist),
      VectorType{number_of_grid_points});

  Variables<tmpl::list<TestHelpers::Tags::Vector<VectorType>,
                       TestHelpers::Tags::Scalar<VectorType>>>
      assigned(number_of_grid_points);
  assigned.assign_subset(source);
  const auto created = variables_from_tagged_tuple(source);
  CHECK(assigned == created);

  std::vector<value_type> owning_data(
      number_of_grid_points * assigned.number_of_independent_components);
  Variables<tmpl::list<TestHelpers::Tags::Vector<VectorType>,
                       TestHelpers::Tags::Scalar<VectorType>>>
      non_owning_assigned(owning_data.data(), owning_data.size());
  non_owning_assigned.assign_subset(source);
  CHECK(non_owning_assigned == created);
  CHECK(non_owning_assigned.data() == owning_data.data());
}

template <typename VectorType>
void test_variables_equal_within_roundoff() {
  INFO(pretty_type::name<VectorType>());
  MAKE_GENERATOR(gen);
  using value_type = typename VectorType::value_type;
  UniformCustomDistribution<tt::get_fundamental_type_t<value_type>>
      roundoff_dist{-1.e-12, 1.e-12};
  UniformCustomDistribution<size_t> sdist{1, 5};
  const size_t num_points = sdist(gen);
  using Vars = Variables<tmpl::list<TestHelpers::Tags::Vector<VectorType>,
                                    TestHelpers::Tags::Scalar<VectorType>>>;

  const auto do_checks = [&num_points, &gen, &roundoff_dist](Vars& vars) {
    CHECK(equal_within_roundoff(vars, 0.));
    CHECK(equal_within_roundoff(vars, 0., 0.));
    CHECK(equal_within_roundoff(0., vars));
    CHECK(equal_within_roundoff(vars, Vars{num_points, 0.}));
    get(get<TestHelpers::Tags::Scalar<VectorType>>(vars)) += 1.;
    CHECK_FALSE(equal_within_roundoff(vars, 0.));
    CHECK_FALSE(equal_within_roundoff(vars, Vars{num_points, 0.}));
    vars = make_with_random_values<Vars>(
        make_not_null(&gen), make_not_null(&roundoff_dist), num_points);
    CHECK_FALSE(equal_within_roundoff(vars, 0., 0.));
    CHECK(equal_within_roundoff(vars, 0., 1.e-12));
  };

  Vars owning_vars{num_points, 0.};
  do_checks(owning_vars);

  std::vector<value_type> owning_data(
      num_points * owning_vars.number_of_independent_components, 0.);
  Vars non_owning_vars{owning_data.data(), owning_data.size()};
  do_checks(non_owning_vars);
}

template <typename VectorType>
void test_math_wrapper() {
  INFO(pretty_type::short_name<VectorType>());
  MAKE_GENERATOR(gen);
  using value_type = typename VectorType::value_type;
  UniformCustomDistribution<tt::get_fundamental_type_t<value_type>> value_dist{
      -10.0, 10.0};
  UniformCustomDistribution<size_t> sdist{1, 5};
  const size_t num_points = sdist(gen);
  using Vars = Variables<tmpl::list<TestHelpers::Tags::Vector<VectorType>,
                                    TestHelpers::Tags::Scalar<VectorType>>>;
  const auto vars1 = make_with_random_values<Vars>(
      make_not_null(&gen), make_not_null(&value_dist), num_points);
  const auto vars2 = make_with_random_values<Vars>(
      make_not_null(&gen), make_not_null(&value_dist), num_points);
  const auto scalar = make_with_random_values<typename Vars::value_type>(
      make_not_null(&gen), make_not_null(&value_dist), num_points);
  TestHelpers::MathWrapper::test_type(vars1, vars2, scalar);
}

template <typename VectorType>
void test_contains_allocations() {
  INFO(pretty_type::short_name<VectorType>());
  using Vars = Variables<tmpl::list<TestHelpers::Tags::Vector<VectorType>,
                                    TestHelpers::Tags::Scalar<VectorType>>>;
  CHECK(not contains_allocations(Vars{}));
  CHECK(not contains_allocations(Vars{0}));
  for (size_t size = 1; size < 5; ++size) {
    CAPTURE(size);
    Vars source{size};
    const auto* const source_data = source.data();
    const auto dest = std::move(source);
    CHECK(contains_allocations(dest) == (source_data == dest.data()));
  }
}

template <typename VectorType>
void test_make_with_value() {
  INFO(pretty_type::short_name<VectorType>());
  using Vars = Variables<tmpl::list<TestHelpers::Tags::Vector<VectorType>,
                                    TestHelpers::Tags::Scalar<VectorType>>>;
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<size_t> sdist{1, 5};
  const size_t num_points = sdist(gen);
  CHECK(make_with_value<Vars>(DataVector(num_points, 4.5), -2.3) ==
        Vars(num_points, -2.3));
  CHECK(make_with_value<DataVector>(Vars(num_points, 4.5), -2.3) ==
        DataVector(num_points, -2.3));
}

template <typename VectorType>
void test_set_number_of_grid_points() {
  INFO(pretty_type::short_name<VectorType>());
  using Vars = Variables<tmpl::list<TestHelpers::Tags::Vector<VectorType>,
                                    TestHelpers::Tags::Scalar<VectorType>>>;
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<size_t> sdist{1, 5};
  const size_t num_points = sdist(gen);

  Vars resized{};
  set_number_of_grid_points(make_not_null(&resized), num_points);
  CHECK(resized.number_of_grid_points() == num_points);
  DataVector resized_vector{};
  set_number_of_grid_points(make_not_null(&resized_vector), Vars(num_points));
  CHECK(resized_vector.size() == num_points);
}

void test_asserts() {
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      ([]() {
        Variables<tmpl::list<TestHelpers::Tags::Vector<DataVector>,
                             TestHelpers::Tags::Scalar<DataVector>,
                             TestHelpers::Tags::Scalar2<DataVector>>>
            vars(1, -3.0);
        auto& tensor_in_vars = get<TestHelpers::Tags::Vector<DataVector>>(vars);
        tensor_in_vars = tnsr::I<DataVector, 3>{10_st, -4.0};
      }()),
      Catch::Matchers::ContainsSubstring(
          "Must move into same size, not 10 into 1"));
  CHECK_THROWS_WITH(
      ([]() {
        Variables<tmpl::list<TestHelpers::Tags::Scalar<DataVector>>> vars;
        get<TestHelpers::Tags::Scalar<DataVector>>(vars) =
            Scalar<DataVector>{{{{0.}}}};
      }()),
      Catch::Matchers::ContainsSubstring(
          "Must move into same size, not 1 into 0"));
  CHECK_THROWS_WITH(
      ([]() {
        Variables<tmpl::list<TestHelpers::Tags::Scalar<DataVector>,
                             TestHelpers::Tags::Scalar2<DataVector>>>
            source(2, -3.0);
        // Multiply by one to convert to an expression template.
        (void)static_cast<
            Variables<tmpl::list<TestHelpers::Tags::Vector<DataVector>>>>(
            1.0 * source);
      }()),
      Catch::Matchers::ContainsSubstring(
          "Invalid size 4 for a Variables with 3 components."));
  CHECK_THROWS_WITH(
      ([]() {
        Variables<tmpl::list<TestHelpers::Tags::Scalar<DataVector>,
                             TestHelpers::Tags::Scalar2<DataVector>>>
            source(2, -3.0);
        Variables<tmpl::list<TestHelpers::Tags::Vector<DataVector>>>
            destination;
        // Multiply by one to convert to an expression template.
        destination = 1.0 * source;
      }()),
      Catch::Matchers::ContainsSubstring(
          "Invalid size 4 for a Variables with 3 components."));
#endif // defined(SPECTRE_DEBUG)
}

SPECTRE_TEST_CASE("Unit.DataStructures.Variables", "[DataStructures][Unit]") {
  {
    INFO("Test Variables construction, access, and assignment");
    test_variables_construction_and_access<ComplexDataVector>();
    test_variables_construction_and_access<ComplexModalVector>();
    test_variables_construction_and_access<DataVector>();
    test_variables_construction_and_access<ModalVector>();
  }

  {
    INFO("Test Variables move operations");
    test_variables_move<ComplexDataVector>();
    test_variables_move<ComplexModalVector>();
    test_variables_move<DataVector>();
    test_variables_move<ModalVector>();
  }

  {
    INFO("Test Variables arithmetic operations");
    test_variables_math<DataVector>();
    test_variables_math<ComplexDataVector>();
    // tests for ModalVector and ComplexModalVector omitted due to limited
    // arithmetic operation support for ModalVectors
  }

  {
    INFO("Test Prefix Variables move and copy semantics");
    test_variables_prefix_semantics<ComplexDataVector>();
    test_variables_prefix_semantics<ComplexModalVector>();
    test_variables_prefix_semantics<DataVector>();
    test_variables_prefix_semantics<ModalVector>();
  }

  {
    INFO("Test Prefix Variables arithmetic operations");
    test_variables_prefix_math<ComplexDataVector>();
    test_variables_prefix_math<ComplexModalVector>();
    test_variables_prefix_math<DataVector>();
    test_variables_prefix_math<ModalVector>();
  }

  {
    INFO("Test Variables serialization");
    test_variables_serialization<ComplexDataVector>();
    test_variables_serialization<ComplexModalVector>();
    test_variables_serialization<DataVector>();
    test_variables_serialization<ModalVector>();
  }

  {
    INFO("Test Variables assign subset");
    test_variables_assign_subset<ComplexDataVector>();
    test_variables_assign_subset<ComplexModalVector>();
    test_variables_assign_subset<DataVector>();
    test_variables_assign_subset<ModalVector>();
  }

  {
    INFO("Test Variables extract subset");
    test_variables_extract_subset<ComplexDataVector>();
    test_variables_extract_subset<ComplexModalVector>();
    test_variables_extract_subset<DataVector>();
    test_variables_extract_subset<ModalVector>();
  }

  {
    INFO("Test Variables reference subset");
    test_variables_reference_subset<ComplexDataVector>();
    test_variables_reference_subset<ComplexModalVector>();
    test_variables_reference_subset<DataVector>();
    test_variables_reference_subset<ModalVector>();
  }

  {
    INFO("Test Variables reference_with_different_prefixes");
    test_variables_reference_with_different_prefixes<ComplexDataVector>();
    test_variables_reference_with_different_prefixes<ComplexModalVector>();
    test_variables_reference_with_different_prefixes<DataVector>();
    test_variables_reference_with_different_prefixes<ModalVector>();
  }

  {
    INFO("Test variables_from_tagged_tuple");
    test_variables_from_tagged_tuple<ComplexDataVector>();
    test_variables_from_tagged_tuple<ComplexModalVector>();
    test_variables_from_tagged_tuple<DataVector>();
    test_variables_from_tagged_tuple<ModalVector>();
  }

  {
    INFO("Test equal_within_roundoff with Variables");
    test_variables_equal_within_roundoff<ComplexDataVector>();
    test_variables_equal_within_roundoff<ComplexModalVector>();
    test_variables_equal_within_roundoff<DataVector>();
    test_variables_equal_within_roundoff<ModalVector>();
  }

  {
    INFO("Test MathWrapper");
    test_math_wrapper<ComplexDataVector>();
    test_math_wrapper<ComplexModalVector>();
    test_math_wrapper<DataVector>();
    test_math_wrapper<ModalVector>();
  }

  {
    INFO("Test contains_allocations");
    test_contains_allocations<ComplexDataVector>();
    test_contains_allocations<ComplexModalVector>();
    test_contains_allocations<DataVector>();
    test_contains_allocations<ModalVector>();
  }

  {
    INFO("Test make_with_value");
    test_make_with_value<ComplexDataVector>();
    test_make_with_value<ComplexModalVector>();
    test_make_with_value<DataVector>();
    test_make_with_value<ModalVector>();
  }

  {
    INFO("Test set_number_of_grid_points");
    test_set_number_of_grid_points<ComplexDataVector>();
    test_set_number_of_grid_points<ComplexModalVector>();
    test_set_number_of_grid_points<DataVector>();
    test_set_number_of_grid_points<ModalVector>();
  }

  TestHelpers::db::test_simple_tag<Tags::TempScalar<1>>("TempTensor1");

  SECTION("Test empty variables") { test_empty_variables(); }

  test_asserts();
}
}  // namespace
