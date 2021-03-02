// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Helper functions to test the linearization of elliptic boundary conditions

#pragma once

#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/TestingFramework.hpp"
#include "Utilities/TMPL.hpp"

namespace TestHelpers::elliptic::BoundaryConditions {
namespace detail {

template <typename BoundaryCondition, typename... PrimalFields,
          typename... ArgsTags, typename... ArgsLinearizedTags>
void test_linearization_impl(const BoundaryCondition& boundary_condition,
                             const double correction_magnitude,
                             const DataVector& used_for_size,
                             tmpl::list<PrimalFields...> /*meta*/,
                             tmpl::list<ArgsTags...> /*meta*/,
                             tmpl::list<ArgsLinearizedTags...> /*meta*/) {
  CAPTURE(correction_magnitude);
  using FieldsType = Variables<tmpl::list<PrimalFields...>>;
  using NDotFluxesType =
      Variables<tmpl::list<::Tags::NormalDotFlux<PrimalFields>...>>;

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(0.5, 1.);
  std::uniform_real_distribution<> dist_eps(-correction_magnitude / 2.,
                                            correction_magnitude / 2.);
  Approx custom_approx =
      Approx::custom().epsilon(correction_magnitude).scale(1.);

  // Generate two sets of random fields that differ by O(correction_magnitude)
  auto fields = make_with_random_values<FieldsType>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  auto n_dot_fluxes = make_with_random_values<NDotFluxesType>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  auto fields_correction = make_with_random_values<FieldsType>(
      make_not_null(&generator), make_not_null(&dist_eps), used_for_size);
  auto n_dot_fluxes_correction = make_with_random_values<NDotFluxesType>(
      make_not_null(&generator), make_not_null(&dist_eps), used_for_size);
  FieldsType fields_corrected = fields + fields_correction;
  NDotFluxesType n_dot_fluxes_corrected = n_dot_fluxes + n_dot_fluxes_correction;

  // Generate random background fields
  auto args = make_with_random_values<Variables<tmpl::append<
      typename BoundaryCondition::argument_tags,
      tmpl::list<PrimalFields..., ::Tags::NormalDotFlux<PrimalFields>...>>>>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  // The linearized boundary conditions may use the nonlinear fields and fluxes
  // as background
  args.assign_subset(fields);
  args.assign_subset(n_dot_fluxes);

  // Apply the nonlinear boundary conditions to both sets of fields and take the
  // difference
  boundary_condition.apply(
      make_not_null(&get<PrimalFields>(fields))...,
      make_not_null(&get<::Tags::NormalDotFlux<PrimalFields>>(n_dot_fluxes))...,
      get<ArgsTags>(args)...);
  boundary_condition.apply(
      make_not_null(&get<PrimalFields>(fields_corrected))...,
      make_not_null(
          &get<::Tags::NormalDotFlux<PrimalFields>>(n_dot_fluxes_corrected))...,
      get<ArgsTags>(args)...);
  const FieldsType fields_diff = fields_corrected - fields;
  const NDotFluxesType n_dot_fluxes_diff =
      n_dot_fluxes_corrected - n_dot_fluxes;

  // Apply the linearized boundary conditions to the difference in the fields.
  // This should be the same as the difference of the nonlinear boundary
  // conditions, to the order of the correction_magnitude.
  boundary_condition.apply_linearized(
      make_not_null(&get<PrimalFields>(fields_correction))...,
      make_not_null(&get<::Tags::NormalDotFlux<PrimalFields>>(
          n_dot_fluxes_correction))...,
      get<ArgsLinearizedTags>(args)...);
  CHECK_VARIABLES_CUSTOM_APPROX(fields_diff, fields_correction, custom_approx);
  CHECK_VARIABLES_CUSTOM_APPROX(n_dot_fluxes_diff, n_dot_fluxes_correction,
                                custom_approx);
}

}  // namespace detail

/// Test the `boundary_condition` has a linearization and that it is actually
/// the linearization of the nonlinear boundary condition to the order given by
/// the `correction_magnitude`
template <typename System, typename BoundaryCondition>
void test_linearization(const BoundaryCondition& boundary_condition,
                        const double correction_magnitude,
                        const DataVector& used_for_size) {
  detail::test_linearization_impl(
      boundary_condition, correction_magnitude, used_for_size,
      typename System::primal_fields{},
      typename BoundaryCondition::argument_tags{},
      typename BoundaryCondition::argument_tags_linearized{});
}
}  // namespace TestHelpers::elliptic::BoundaryConditions
