// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/Elasticity/HalfSpaceMirror.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <gsl/gsl_sf_bessel.h>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "ErrorHandling/Error.hpp"
#include "NumericalAlgorithms/Integration/GslQuadAdaptive.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Elasticity::Solutions {

HalfSpaceMirror::HalfSpaceMirror(
    const double beam_width, constitutive_relation_type constitutive_relation,
    const size_t integration_intervals, const double absolute_tolerance,
    const double relative_tolerance) noexcept
    : beam_width_(beam_width),
      constitutive_relation_(std::move(constitutive_relation)),
      integration_intervals_(integration_intervals),
      absolute_tolerance_(absolute_tolerance),
      relative_tolerance_(relative_tolerance) {}

tuples::TaggedTuple<Tags::Displacement<3>> HalfSpaceMirror::variables(
    const tnsr::I<DataVector, 3>& x,
    tmpl::list<Tags::Displacement<3>> /*meta*/) const noexcept {
  const double shear_modulus = constitutive_relation_.shear_modulus();
  const double lame_parameter = constitutive_relation_.lame_parameter();
  auto result = make_with_value<tnsr::I<DataVector, 3>>(x, 0.);
  const auto radius = sqrt(square(get<0>(x)) + square(get<1>(x)));

  const integration::GslQuadAdaptive<
      integration::GslIntegralType::UpperBoundaryInfinite>
      integration{integration_intervals_};
  const double lower_boundary = 0.;
  const size_t num_points = get<0>(x).size();
  for (size_t i = 0; i < num_points; i++) {
    const double z = get<2>(x)[i];
    const double r = radius[i];
    if (r >= 100 * std::numeric_limits<double>::epsilon()) {
      const double surface_term_r = 1. - (lame_parameter + 2. * shear_modulus) /
                                             (lame_parameter + shear_modulus);
      const double result_r =
          0.25 / (M_PI * shear_modulus * r) *
          integration(
              [&r, &z, &surface_term_r, this](const double k) noexcept {
                return gsl_sf_bessel_J1(k * r) *
                       exp(-k * z - square(k * beam_width_ / 2.)) *
                       (surface_term_r + k * z);
              },
              lower_boundary, absolute_tolerance_, relative_tolerance_);
      get<0>(result)[i] = get<0>(x)[i] * result_r;
      get<1>(result)[i] = get<1>(x)[i] * result_r;
    }  // else x and y component vanish for r = 0
    const double surface_term_z =
        1. + shear_modulus / (lame_parameter + shear_modulus);
    const double result_z =
        0.25 / (shear_modulus * M_PI) *
        integration(
            [&r, &z, &surface_term_z, this](const double k) noexcept {
              return gsl_sf_bessel_J0(k * r) *
                     exp(-k * z - square(k * beam_width_ / 2.)) *
                     (surface_term_z + k * z);
            },
            lower_boundary, absolute_tolerance_, relative_tolerance_);
    get<2>(result)[i] = result_z;
  }
  return {std::move(result)};
}

tuples::TaggedTuple<Tags::Strain<3>> HalfSpaceMirror::variables(
    const tnsr::I<DataVector, 3>& x, tmpl::list<Tags::Strain<3>> /*meta*/) const
    noexcept {
  const double shear_modulus = constitutive_relation_.shear_modulus();
  const double lame_parameter = constitutive_relation_.lame_parameter();
  auto strain = make_with_value<tnsr::ii<DataVector, 3>>(x, 0.);
  const auto radius = sqrt(square(get<0>(x)) + square(get<1>(x)));
  const integration::GslQuadAdaptive<
      integration::GslIntegralType::UpperBoundaryInfinite>
      integration{integration_intervals_};
  const double lower_boundary = 0.;
  const size_t num_points = get<0>(x).size();
  for (size_t i = 0; i < num_points; i++) {
    const double r = radius[i];
    const double z = get<2>(x)[i];

    const double surface_term_trace =
        -2. * shear_modulus / (lame_parameter + shear_modulus);
    const double trace_term =
        0.25 / (shear_modulus * M_PI) *
        integration(
            [&r, &z, &surface_term_trace, this](const double k) noexcept {
              return k * gsl_sf_bessel_J0(k * r) *
                     exp(-k * z - square(k * beam_width_ / 2.)) *
                     surface_term_trace;
            },
            lower_boundary, absolute_tolerance_, relative_tolerance_);

    const double surface_term_zz =
        shear_modulus / (lame_parameter + shear_modulus);
    const double strain_zz =
        -0.25 / (shear_modulus * M_PI) *
        integration(
            [&r, &z, &surface_term_zz, this](const double k) noexcept {
              return k * gsl_sf_bessel_J0(k * r) *
                     exp(-k * z - square(k * beam_width_ / 2.)) *
                     (surface_term_zz + k * z);
            },
            lower_boundary, absolute_tolerance_, relative_tolerance_);

    if (r >= 100 * std::numeric_limits<double>::epsilon()) {
      const double strain_rz =
          -0.25 / (shear_modulus * M_PI) *
          integration(
              [&r, &z, this](const double k) noexcept {
                return k * gsl_sf_bessel_J1(k * r) *
                       exp(-k * z - square(k * beam_width_ / 2.)) * (k * z);
              },
              lower_boundary, absolute_tolerance_, relative_tolerance_);

      const double surface_term_r = 1. - (lame_parameter + 2. * shear_modulus) /
                                             (lame_parameter + shear_modulus);
      const double displacement_r =
          0.25 / (shear_modulus * M_PI) *
          integration(
              [&r, &z, &surface_term_r, this](const double k) noexcept {
                return gsl_sf_bessel_J1(k * r) *
                       exp(-k * z - square(k * beam_width_ / 2.)) *
                       (surface_term_r + k * z);
              },
              lower_boundary, absolute_tolerance_, relative_tolerance_);
      const double cos_phi = get<0>(x)[i] / r;
      const double sin_phi = get<1>(x)[i] / r;
      const double strain_pp = displacement_r / r;
      const double strain_rr = trace_term - strain_pp - strain_zz;
      get<0, 0>(strain)[i] =
          strain_pp + square(cos_phi) * (strain_rr - strain_pp);
      get<0, 1>(strain)[i] = cos_phi * sin_phi * (strain_rr - strain_pp);
      get<0, 2>(strain)[i] = cos_phi * strain_rz;
      get<1, 1>(strain)[i] =
          strain_pp + square(sin_phi) * (strain_rr - strain_pp);
      get<1, 2>(strain)[i] = sin_phi * strain_rz;
      get<2, 2>(strain)[i] = strain_zz;

    } else {
      get<0, 0>(strain)[i] = 0.5 * (trace_term - strain_zz);
      get<1, 1>(strain)[i] = get<0, 0>(strain)[i];
      get<2, 2>(strain)[i] = strain_zz;
      // off-diagonal components vanish for r = 0
    }
  }
  return {std::move(strain)};
}

tuples::TaggedTuple<::Tags::FixedSource<Tags::Displacement<3>>>
HalfSpaceMirror::variables(
    const tnsr::I<DataVector, 3>& x,
    tmpl::list<::Tags::FixedSource<Tags::Displacement<3>>> /*meta*/) noexcept {
  return {make_with_value<tnsr::I<DataVector, 3>>(x, 0.)};
}

void HalfSpaceMirror::pup(PUP::er& p) noexcept {
  p | beam_width_;
  p | constitutive_relation_;
  p | integration_intervals_;
  p | absolute_tolerance_;
  p | relative_tolerance_;
}

bool operator==(const HalfSpaceMirror& lhs,
                const HalfSpaceMirror& rhs) noexcept {
  return lhs.beam_width_ == rhs.beam_width_ and
         lhs.constitutive_relation_ == rhs.constitutive_relation_ and
         lhs.integration_intervals_ == rhs.integration_intervals_ and
         lhs.absolute_tolerance_ == rhs.absolute_tolerance_ and
         lhs.relative_tolerance_ == rhs.relative_tolerance_;
}

bool operator!=(const HalfSpaceMirror& lhs,
                const HalfSpaceMirror& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace Elasticity::Solutions
