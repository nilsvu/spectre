// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/SelfForce/Scalar/CircularOrbit.hpp"

#include <complex>
#include <cstddef>
#include <effsource.hpp>
#include <gsl/gsl_errno.h>
#include <korb.hpp>
#include <utility>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TortoiseCoordinates.hpp"
#include "Utilities/Gsl.hpp"

namespace ScalarSelfForce::AnalyticData {

CircularOrbit::CircularOrbit(const double black_hole_mass,
                             const double black_hole_spin,
                             const double orbital_radius,
                             const int m_mode_number)
    : black_hole_mass_(black_hole_mass),
      black_hole_spin_(black_hole_spin),
      orbital_radius_(orbital_radius),
      m_mode_number_(m_mode_number) {}

tnsr::I<double, 2> CircularOrbit::puncture_position() const {
  const double a = black_hole_spin_ * black_hole_mass_;
  const double M = black_hole_mass_;
  const double r_plus = M * (1. + sqrt(1. - square(black_hole_spin_)));
  const double r_0 = orbital_radius_;
  const double r_star = korb_rsfromrsubtrplus(r_0 - r_plus, a);
  return tnsr::I<double, 2>{{{r_star, 0.}}};
}

// Background
tuples::TaggedTuple<Tags::Alpha, Tags::Beta, Tags::Gamma>
CircularOrbit::variables(
    const tnsr::I<DataVector, 2>& x,
    tmpl::list<Tags::Alpha, Tags::Beta, Tags::Gamma> /*meta*/) const {
  const double a = black_hole_spin_ * black_hole_mass_;
  const double M = black_hole_mass_;
  const double r_plus = M * (1. + sqrt(1. - square(black_hole_spin_)));
  const double r_minus = M * (1. - sqrt(1. - square(black_hole_spin_)));
  const double r_0 = orbital_radius_;
  const double omega = 1. / (a + sqrt(cube(r_0) / M));
  const auto& r_star = get<0>(x);
  const auto& cos_theta = get<1>(x);
  const DataVector r_minus_r_plus =
      gr::boyer_lindquist_radius_minus_r_plus_from_tortoise(r_star, M,
                                                            black_hole_spin_);
  const DataVector r = r_minus_r_plus + r_plus;
  const DataVector delta = r_minus_r_plus * (r - r_minus);
  const DataVector r_sq_plus_a_sq = square(r) + square(a);
  const DataVector r_sq_plus_a_sq_sq = square(r_sq_plus_a_sq);
  const DataVector sin_theta_squared = 1. - square(cos_theta);
  const DataVector sigma_squared =
      r_sq_plus_a_sq_sq - square(a) * delta * sin_theta_squared;
  tuples::TaggedTuple<Tags::Alpha, Tags::Beta, Tags::Gamma> result{};
  auto& alpha = get<Tags::Alpha>(result);
  auto& beta = get<Tags::Beta>(result);
  auto& gamma = get<Tags::Gamma>(result);
  get(alpha) = delta / r_sq_plus_a_sq_sq;
  const ComplexDataVector temp1 =
      1. / r * std::complex<double>(0., 2. * a * m_mode_number_);
  get(beta) = (-square(m_mode_number_ * omega) * sigma_squared +
               4. * a * square(m_mode_number_) * omega * M * r +
               delta * (square(m_mode_number_) / sin_theta_squared +
                        2. * M / r * (1. - square(a) / M / r) + temp1)) /
              r_sq_plus_a_sq_sq;
  get<0>(gamma) =
      -1. / r_sq_plus_a_sq * std::complex<double>(0., 2. * a * m_mode_number_) +
      2. * square(a) * get(alpha) / r;
  get<1>(gamma) = ComplexDataVector{cos_theta.size(), 0.};
  get(alpha) *= sin_theta_squared;
  return result;
}

// Initial guess
tuples::TaggedTuple<Tags::MMode> CircularOrbit::variables(
    const tnsr::I<DataVector, 2>& x, tmpl::list<Tags::MMode> /*meta*/) const {
  tuples::TaggedTuple<Tags::MMode> result{};
  auto& field = get<Tags::MMode>(result);
  get(field) = ComplexDataVector{get<0>(x).size(), 0.};
  return result;
}

// Fixed sources
tuples::TaggedTuple<
    ::Tags::FixedSource<Tags::MMode>, Tags::SingularField,
    ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>,
    Tags::BoyerLindquistRadius>
CircularOrbit::variables(
    const tnsr::I<DataVector, 2>& x,
    tmpl::list<
        ::Tags::FixedSource<Tags::MMode>, Tags::SingularField,
        ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>,
        Tags::BoyerLindquistRadius> /*meta*/) const {
  const double a = black_hole_spin_ * black_hole_mass_;
  const double M = black_hole_mass_;
  const double r_0 = orbital_radius_;
  const double r_plus = M * (1. + sqrt(1. - square(black_hole_spin_)));
  const double r_minus = M * (1. - sqrt(1. - square(black_hole_spin_)));
  {
    // Initialize effsource
    effsource_init(M, a);
    struct coordinate xp;
    xp.t = 0;
    xp.r = r_0;
    xp.theta = M_PI_2;
    xp.phi = 0;
    // Circular equatorial orbit, as given in the EffectiveSource example
    const double e = ((r_0 - 2.0 * M) * sqrt(M * r_0) + a * M) /
                     (sqrt(M * r_0) * sqrt(r_0 * r_0 - 3.0 * M * r_0 +
                                           2.0 * a * sqrt(M * r_0)));
    const double l = (M * (a * a + r_0 * r_0 - 2.0 * a * sqrt(M * r_0))) /
                     (sqrt(M * r_0) * sqrt(r_0 * r_0 - 3.0 * M * r_0 +
                                           2.0 * a * sqrt(M * r_0)));
    effsource_set_particle(&xp, e, l, 0.);
  }
  const auto& r_star = get<0>(x);
  const auto& cos_theta = get<1>(x);
  const DataVector r_minus_r_plus =
      gr::boyer_lindquist_radius_minus_r_plus_from_tortoise(r_star, M,
                                                            black_hole_spin_);
  const DataVector r = r_minus_r_plus + r_plus;
  const DataVector delta = r_minus_r_plus * (r - r_minus);
  const DataVector r_sq_plus_a_sq = square(r) + square(a);
  const DataVector r_sq_plus_a_sq_sq = square(r_sq_plus_a_sq);
  const DataVector delta_phi = m_mode_number_ * a / (r_plus - r_minus) *
                               log((r - r_plus) / (r - r_minus));
  const ComplexDataVector rotation =
      cos(delta_phi) - std::complex<double>(0., 1.) * sin(delta_phi);
  tuples::TaggedTuple<
      ::Tags::FixedSource<Tags::MMode>, Tags::SingularField,
      ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>,
      Tags::BoyerLindquistRadius>
      result{};
  get(get<Tags::BoyerLindquistRadius>(result)) = r;
  const size_t num_points = get<0>(x).size();
  Scalar<ComplexDataVector>& effective_source =
      get<::Tags::FixedSource<Tags::MMode>>(result);
  get(effective_source).destructive_resize(num_points);
  Scalar<ComplexDataVector>& singular_field = get<Tags::SingularField>(result);
  get(singular_field).destructive_resize(num_points);
  tnsr::i<ComplexDataVector, 2>& deriv_singular_field =
      get<::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>>(
          result);
  get<0>(deriv_singular_field).destructive_resize(num_points);
  get<1>(deriv_singular_field).destructive_resize(num_points);
  struct coordinate x_i;
  double PhiS[2], dPhiS_dx[8], d2PhiS_dx2[20], src[2];
  for (size_t i = 0; i < num_points; ++i) {
    x_i.t = 0;
    x_i.r = r[i];
    x_i.theta = acos(cos_theta[i]);
    x_i.phi = 0;
    effsource_calc_m(m_mode_number_, &x_i, PhiS, dPhiS_dx, d2PhiS_dx2, src);
    get(effective_source)[i] = src[0] + std::complex<double>(0., 1.) * src[1];
    get(singular_field)[i] = PhiS[0] + std::complex<double>(0., 1.) * PhiS[1];
    get<0>(deriv_singular_field)[i] =
        dPhiS_dx[2] + std::complex<double>(0., 1.) * dPhiS_dx[3];
    get<1>(deriv_singular_field)[i] =
        dPhiS_dx[4] + std::complex<double>(0., 1.) * dPhiS_dx[5];
  }
  // Rotate the source by delta_phi and multiply by r / 2 pi
  get(effective_source) *= rotation * 0.5 * r / M_PI;
  // Factor Delta * (r^2 + a^2 cos^2(theta)) / Sigma^2
  // Factor Sigma^2 / (r^2 + a^2)^2 from first-order formulation
  get(effective_source) *=
      delta * (square(r) + square(a * cos_theta)) / r_sq_plus_a_sq_sq;
  get(singular_field) *= rotation * 0.5 * r / M_PI;
  get<0>(deriv_singular_field) *= rotation * 0.5 * r / M_PI;
  get<0>(deriv_singular_field) +=
      get(singular_field) / r - std::complex<double>(0., a * m_mode_number_) /
                                    delta * get(singular_field);
  get<0>(deriv_singular_field) *= delta / r_sq_plus_a_sq;
  get<1>(deriv_singular_field) *= rotation * 0.5 * r / M_PI;
  get<1>(deriv_singular_field) /= -sqrt(1. - square(cos_theta));
  return result;
}

void CircularOrbit::pup(PUP::er& p) {
  elliptic::analytic_data::Background::pup(p);
  elliptic::analytic_data::InitialGuess::pup(p);
  p | black_hole_mass_;
  p | black_hole_spin_;
  p | orbital_radius_;
  p | m_mode_number_;
}

bool operator==(const CircularOrbit& lhs, const CircularOrbit& rhs) {
  return lhs.black_hole_mass_ == rhs.black_hole_mass_ and
         lhs.black_hole_spin_ == rhs.black_hole_spin_ and
         lhs.orbital_radius_ == rhs.orbital_radius_ and
         lhs.m_mode_number_ == rhs.m_mode_number_;
}

bool operator!=(const CircularOrbit& lhs, const CircularOrbit& rhs) {
  return not(lhs == rhs);
}

PUP::able::PUP_ID CircularOrbit::my_PUP_ID = 0;  // NOLINT

}  // namespace ScalarSelfForce::AnalyticData
