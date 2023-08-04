// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <deque>

#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTimeHelpers.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"

namespace domain::FunctionsOfTime {
SPECTRE_TEST_CASE("Unit.Domain.FunctionsOfTime.FunctionOfTimeHelpers",
                  "[Domain][Unit]") {
  {
    INFO("StoredInfo Construction");
    // mdp1 = MaxDerivPlusOne
    constexpr size_t mdp1 = 3;
    constexpr double time = 5.4;
    DataVector init_func{{1.1, 1.2, 1.3}};
    DataVector init_deriv{{2.4, 2.5, 2.6}};
    DataVector init_2deriv{{3.7, 3.8, 3.9}};
    std::array<DataVector, mdp1> init_arr =
        std::array<DataVector, mdp1>{{init_func, init_deriv, init_2deriv}};

    FunctionOfTimeHelpers::StoredInfo<mdp1> si1{time, init_arr};
    FunctionOfTimeHelpers::StoredInfo<mdp1, true> si2{time, init_arr};
    CHECK(si1 == si2);
    CHECK(si1.time == time);
    CHECK(si2.time == time);
    CHECK(gsl::at(si1.stored_quantities, 0) == init_func);
    CHECK(gsl::at(si2.stored_quantities, 0) == init_func);
    CHECK(gsl::at(si1.stored_quantities, 1) == init_deriv);
    CHECK(gsl::at(si2.stored_quantities, 1) == init_deriv);
    CHECK(gsl::at(si1.stored_quantities, 2) == init_2deriv / 2.0);
    CHECK(gsl::at(si2.stored_quantities, 2) == init_2deriv / 2.0);

    const auto si1_serialize_and_deserialize = serialize_and_deserialize(si1);
    const auto si2_serialize_and_deserialize = serialize_and_deserialize(si2);
    CHECK(si1_serialize_and_deserialize == si2_serialize_and_deserialize);

    FunctionOfTimeHelpers::StoredInfo<mdp1, false> si3{time, init_arr};
    CHECK(si3.time == time);
    CHECK(gsl::at(si3.stored_quantities, 0) == init_func);
    CHECK(gsl::at(si3.stored_quantities, 1) == init_deriv);
    CHECK(gsl::at(si3.stored_quantities, 2) == init_2deriv);

    const auto si3_serialize_and_deserialize = serialize_and_deserialize(si3);
    CHECK(si3_serialize_and_deserialize.time == time);
    CHECK(gsl::at(si3_serialize_and_deserialize.stored_quantities, 0) ==
          init_func);
    CHECK(gsl::at(si3_serialize_and_deserialize.stored_quantities, 1) ==
          init_deriv);
    CHECK(gsl::at(si3_serialize_and_deserialize.stored_quantities, 2) ==
          init_2deriv);

    CHECK(get_output(si1) ==
          "t=5.4: (1.1,1.2,1.3) (2.4,2.5,2.6) (1.85,1.9,1.95)");
  }

  {
    INFO("StoredInfo From Upper Bound");
    constexpr size_t mdp1 = 1;
    std::deque<FunctionOfTimeHelpers::StoredInfo<mdp1>> all_stored_info{
        FunctionOfTimeHelpers::StoredInfo<mdp1>{
            0.0, std::array<DataVector, mdp1>{DataVector{1, 0.0}}}};
    for (int t = 1; t < 10; t++) {
      all_stored_info.emplace_back(
          static_cast<double>(t),
          std::array<DataVector, mdp1>{DataVector{3, 0.0}});
    }

    const size_t size = all_stored_info.size();

    const auto test = [&all_stored_info, &size]() {
      // Test between times
      const auto& upper_bound_stored_info1 =
          FunctionOfTimeHelpers::stored_info_from_upper_bound(
              4.5, all_stored_info, size);
      CHECK(upper_bound_stored_info1 == gsl::at(all_stored_info, 4));

      // Test at a time
      const auto& upper_bound_stored_info2 =
          FunctionOfTimeHelpers::stored_info_from_upper_bound(
              4.0, all_stored_info, size);
      CHECK(upper_bound_stored_info2 == gsl::at(all_stored_info, 3));

      // Test slightly before first time
      const auto& upper_bound_stored_info3 =
          FunctionOfTimeHelpers::stored_info_from_upper_bound(
              -1.e-15, all_stored_info, size);
      CHECK(upper_bound_stored_info3 == gsl::at(all_stored_info, 0));

      // Test after all times
      const auto& upper_bound_stored_info4 =
          FunctionOfTimeHelpers::stored_info_from_upper_bound(
              12.5, all_stored_info, size);
      CHECK(upper_bound_stored_info4 == gsl::at(all_stored_info, size - 1));
    };

    test();
    all_stored_info.emplace_back(
        11.0, std::array<DataVector, mdp1>{DataVector{3, 0.0}});
    // Test when our stored size has increased, but the size we are allowed to
    // check with remains the same that we get the same results
    test();
  }

  CHECK_THROWS_WITH(
      []() {
        double expr_time = 4.0;
        double next_expr_time = 3.5;
        FunctionOfTimeHelpers::reset_expiration_time(make_not_null(&expr_time),
                                                     next_expr_time);
      }(),
      Catch::Matchers::ContainsSubstring(
          "Attempted to change expiration time to 3.5") and
          Catch::Matchers::ContainsSubstring(
              ", which precedes the previous expiration time of 4"));

  CHECK_THROWS_WITH(
      ([]() {
        constexpr size_t mdp1 = 1;
        std::deque<FunctionOfTimeHelpers::StoredInfo<mdp1>> all_stored_info{
            FunctionOfTimeHelpers::StoredInfo<mdp1>{
                0.0, std::array<DataVector, mdp1>{DataVector{1, 0.0}}}};
        for (int t = 1; t < 10; t++) {
          all_stored_info.emplace_back(
              static_cast<double>(t),
              std::array<DataVector, mdp1>{DataVector{1, 0.0}});
        }

        (void)FunctionOfTimeHelpers::stored_info_from_upper_bound(
            -1.0, all_stored_info, all_stored_info.size());
      }()),
      Catch::Matchers::ContainsSubstring("requested time -1") and
          Catch::Matchers::ContainsSubstring(" precedes earliest time 0"));

#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      []() {
        constexpr size_t mdp1 = 3;
        std::deque<FunctionOfTimeHelpers::StoredInfo<mdp1>> all_stored_info;

        (void)FunctionOfTimeHelpers::stored_info_from_upper_bound(
            1.0, all_stored_info, 0);
      }(),
      Catch::Matchers::ContainsSubstring(
          "Deque of StoredInfos you are trying to access is empty. Was "
          "it constructed properly?"));
#endif
}
}  // namespace domain::FunctionsOfTime
