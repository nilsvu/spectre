// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/SelfForce/GeneralRelativity/CircularOrbit.hpp"

#include <complex>
#include <cstddef>
#include <effsource_gr.hpp>
#include <korb.hpp>
#include <utility>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/Tags.hpp"
#include "PointwiseFunctions/AnalyticData/SelfForce/GeneralRelativity/ABC.hpp"
#include "PointwiseFunctions/AnalyticData/SelfForce/GeneralRelativity/source_convert.hpp"
#include "PointwiseFunctions/GeneralRelativity/TortoiseCoordinates.hpp"
#include "Utilities/Gsl.hpp"

namespace GrSelfForce::AnalyticData {

CircularOrbit::CircularOrbit(const double black_hole_mass,
                             const double black_hole_spin,
                             const double orbital_radius,
                             const int m_mode_number)
    : black_hole_mass_(black_hole_mass),
      black_hole_spin_(black_hole_spin),
      orbital_radius_(orbital_radius),
      m_mode_number_(m_mode_number) {}

tnsr::I<double, 3> CircularOrbit::puncture_position() const {
  const double a = black_hole_spin_ * black_hole_mass_;
  const double M = black_hole_mass_;
  const double r_plus = M * (1. + sqrt(1. - square(black_hole_spin_)));
  const double r_0 = orbital_radius_;
  const double r_star = korb_rsfromrsubtrplus(r_0 - r_plus, a);
  return tnsr::I<double, 3>{{{r_star, M_PI_2, 0.}}};
}

// Background
tuples::TaggedTuple<Tags::Alpha, Tags::Beta, Tags::GammaRstar, Tags::GammaTheta>
CircularOrbit::variables(const tnsr::I<DataVector, 3>& x,
                         tmpl::list<Tags::Alpha, Tags::Beta, Tags::GammaRstar,
                                    Tags::GammaTheta> /*meta*/) const {
  const double a = black_hole_spin_ * black_hole_mass_;
  const double M = black_hole_mass_;
  const double r_plus = M * (1. + sqrt(1. - square(black_hole_spin_)));
  const double r_0 = orbital_radius_;
  const double omega = 1. / (a + sqrt(cube(r_0) / M));
  const auto& r_star = get<0>(x);
  const auto& theta = get<1>(x);
  const DataVector r = gr::boyer_lindquist_radius_minus_r_plus_from_tortoise(
                           r_star, M, black_hole_spin_) +
                       r_plus;
  const DataVector delta = square(r) - 2.0 * M * r + square(a);
  const DataVector r_sq_plus_a_sq = square(r) + square(a);
  const DataVector r_sq_plus_a_sq_sq = square(r_sq_plus_a_sq);
  const DataVector sin_theta = sin(theta);
  const DataVector cos_theta = cos(theta);
  const DataVector sin_theta_squared = square(sin_theta);
  const DataVector sigma_squared =
      r_sq_plus_a_sq_sq - square(a) * delta * sin_theta_squared;
  tuples::TaggedTuple<Tags::Alpha, Tags::Beta, Tags::GammaRstar,
                      Tags::GammaTheta>
      result{};
  auto& alpha = get<Tags::Alpha>(result);
  auto& beta = get<Tags::Beta>(result);
  auto& gamma_rstar = get<Tags::GammaRstar>(result);
  auto& gamma_theta = get<Tags::GammaTheta>(result);
  const size_t num_points = r.size();
  get(alpha) = delta / r_sq_plus_a_sq_sq;
  for (size_t i = 0; i < beta.size(); ++i) {
    beta[i] = ComplexDataVector{num_points, 0.};
    gamma_rstar[i] = ComplexDataVector{num_points, 0.};
    gamma_theta[i] = ComplexDataVector{num_points, 0.};
  }
  const ComplexDataVector temp1 =
      1. / r * std::complex<double>(0., 2. * a * m_mode_number_);
  // tt, tr, ttheta, tphi, rr, rtheta, rphi, theta theta, theta phi, phi phi
  double Areal[10][10], Aimag[10][10], Breal[10][10], Bimag[10][10],
      Creal[10][10], Cimag[10][10];
  for (size_t i = 0; i < r.size(); i++) {
    getAreal(m_mode_number_, a, m_mode_number_ * omega, r[i], theta[i], Areal);
    getAimag(m_mode_number_, a, m_mode_number_ * omega, r[i], theta[i], Aimag);
    getBreal(m_mode_number_, a, m_mode_number_ * omega, r[i], theta[i], Breal);
    getBimag(m_mode_number_, a, m_mode_number_ * omega, r[i], theta[i], Bimag);
    getCreal(m_mode_number_, a, m_mode_number_ * omega, r[i], theta[i], Creal);
    getCimag(m_mode_number_, a, m_mode_number_ * omega, r[i], theta[i], Cimag);
    for (size_t a1 = 0; a1 < 4; ++a1) {
      for (size_t b = 0; b <= a1; ++b) {
        const size_t matrix_i =
            tnsr::aa<ComplexDataVector, 3>::get_storage_index(
                std::array<size_t, 2>{{a1, b}});
        for (size_t c = 0; c < 4; ++c) {
          for (size_t d = 0; d <= c; ++d) {
            const size_t matrix_j =
                tnsr::aa<ComplexDataVector, 3>::get_storage_index(
                    std::array<size_t, 2>{{c, d}});
            gamma_rstar.get(a1, b, c, d)[i] =
                Areal[matrix_i][matrix_j] +
                std::complex<double>(0., 1.) * Aimag[matrix_i][matrix_j];
            gamma_theta.get(a1, b, c, d)[i] =
                Breal[matrix_i][matrix_j] +
                std::complex<double>(0., 1.) * Bimag[matrix_i][matrix_j];
            beta.get(a1, b, c, d)[i] =
                Creal[matrix_i][matrix_j] +
                std::complex<double>(0., 1.) * Cimag[matrix_i][matrix_j];
          }
        }
      }
    }
  }
  for (size_t i = 0; i < beta.size(); ++i) {
    beta[i] *= -1.;
    gamma_rstar[i] *= -1.;
    gamma_theta[i] *= -1. / get(alpha);
  }
  return result;
}

// Initial guess
tuples::TaggedTuple<Tags::MModeRe, Tags::MModeIm> CircularOrbit::variables(
    const tnsr::I<DataVector, 3>& x,
    tmpl::list<Tags::MModeRe, Tags::MModeIm> /*meta*/) const {
  tuples::TaggedTuple<Tags::MModeRe, Tags::MModeIm> result{};
  auto& field_re = get<Tags::MModeRe>(result);
  auto& field_im = get<Tags::MModeIm>(result);
  for (size_t i = 0; i < field_re.size(); ++i) {
    field_re[i] = DataVector{get<0>(x).size(), 0.};
    field_im[i] = DataVector{get<0>(x).size(), 0.};
  }
  return result;
}

// Fixed sources
tuples::TaggedTuple<
    ::Tags::FixedSource<Tags::MModeRe>, ::Tags::FixedSource<Tags::MModeIm>,
    Tags::SingularField,
    ::Tags::deriv<Tags::SingularField, tmpl::size_t<3>, Frame::Inertial>,
    Tags::BoyerLindquistRadius>
CircularOrbit::variables(
    const tnsr::I<DataVector, 3>& x,
    tmpl::list<
        ::Tags::FixedSource<Tags::MModeRe>, ::Tags::FixedSource<Tags::MModeIm>,
        Tags::SingularField,
        ::Tags::deriv<Tags::SingularField, tmpl::size_t<3>, Frame::Inertial>,
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
    const double e = ((r_0 - 2.0 * M) * sqrt(M * r_0) + a * M) /
                     (sqrt(M * r_0) * sqrt(r_0 * r_0 - 3.0 * M * r_0 +
                                           2.0 * a * sqrt(M * r_0)));
    const double l = (M * (a * a + r_0 * r_0 - 2.0 * a * sqrt(M * r_0))) /
                     (sqrt(M * r_0) * sqrt(r_0 * r_0 - 3.0 * M * r_0 +
                                           2.0 * a * sqrt(M * r_0)));
    effsource_set_particle(&xp, e, l, 0.);
  }
  const auto& r_star = get<0>(x);
  const auto& theta = get<1>(x);
  const DataVector r = gr::boyer_lindquist_radius_minus_r_plus_from_tortoise(
                           r_star, M, black_hole_spin_) +
                       r_plus;
  const DataVector delta = square(r) - 2.0 * M * r + square(a);
  const DataVector r_sq_plus_a_sq = square(r) + square(a);
  const DataVector r_sq_plus_a_sq_sq = square(r_sq_plus_a_sq);
  const DataVector delta_phi = m_mode_number_ * a / (r_plus - r_minus) *
                               log((r - r_plus) / (r - r_minus));
  const ComplexDataVector rotation =
      cos(delta_phi) - std::complex<double>(0., 1.) * sin(delta_phi);
  tuples::TaggedTuple<
      ::Tags::FixedSource<Tags::MModeRe>, ::Tags::FixedSource<Tags::MModeIm>,
      Tags::SingularField,
      ::Tags::deriv<Tags::SingularField, tmpl::size_t<3>, Frame::Inertial>,
      Tags::BoyerLindquistRadius>
      result{};
  get(get<Tags::BoyerLindquistRadius>(result)) = r;
  const size_t num_points = get<0>(x).size();
  tnsr::aa<ComplexDataVector, 3> effective_source{num_points};
  tnsr::aa<ComplexDataVector, 3>& singular_field =
      get<Tags::SingularField>(result);
  for (size_t i = 0; i < singular_field.size(); i++) {
    singular_field[i].destructive_resize(num_points);
  }
  tnsr::iaa<ComplexDataVector, 3>& deriv_singular_field =
      get<::Tags::deriv<Tags::SingularField, tmpl::size_t<3>, Frame::Inertial>>(
          result);
  for (size_t i = 0; i < deriv_singular_field.size(); i++) {
    deriv_singular_field[i].destructive_resize(num_points);
  }
  struct coordinate x_i;
  double hS_re[10], hS_im[10];
  double hS_tommy_re[10], hS_tommy_im[10];
  double dhS_dr_re[10], dhS_dr_im[10], dhS_dth_re[10], dhS_dth_im[10],
      dhS_dph_re[10], dhS_dph_im[10], dhS_dt_re[10], dhS_dt_im[10];
  double dhS_drstar_re_tommy[10], dhS_drstar_im_tommy[10], dhS_dth_re_tommy[10],
      dhS_dth_im_tommy[10];
  double src_re[10], src_im[10];
  double src_tommy_re[10], src_tommy_im[10];
  for (size_t i = 0; i < get<0>(x).size(); ++i) {
    x_i.t = 0;
    x_i.r = r[i];
    x_i.theta = theta[i];
    x_i.phi = 0;
    effsource_calc_m(m_mode_number_, &x_i, hS_re, hS_im, dhS_dr_re, dhS_dr_im,
                     dhS_dth_re, dhS_dth_im, dhS_dph_re, dhS_dph_im, dhS_dt_re,
                     dhS_dt_im, src_re, src_im);
    psiTommy(m_mode_number_, a, r[i], theta[i], hS_re, hS_im, hS_tommy_re,
             hS_tommy_im);
    dpsidthetaTommy(m_mode_number_, a, r[i], theta[i], hS_re, hS_im, dhS_dth_re,
                    dhS_dth_im, dhS_dth_re_tommy, dhS_dth_im_tommy);
    dpsidrstarTommy(m_mode_number_, a, r[i], theta[i], hS_re, hS_im, dhS_dr_re,
                    dhS_dr_im, dhS_drstar_re_tommy, dhS_drstar_im_tommy);
    SeffTommy(m_mode_number_, a, r[i], theta[i], src_re, src_im, src_tommy_re,
              src_tommy_im);
    for (size_t a1 = 0; a1 < 4; ++a1) {
      for (size_t b = 0; b <= a1; ++b) {
        const size_t comp = tnsr::aa<ComplexDataVector, 3>::get_storage_index(
            std::array<size_t, 2>{{a1, b}});
        // TODO: check this minus sign
        effective_source.get(a1, b)[i] =
            -src_tommy_re[comp] -
            std::complex<double>(0., 1.) * src_tommy_im[comp];
        singular_field.get(a1, b)[i] =
            hS_tommy_re[comp] +
            std::complex<double>(0., 1.) * hS_tommy_im[comp];
        // deriv_singular_field.get(0, a1, b)[i] =
        //     dhS_dt_re[comp] + std::complex<double>(0., 1.) * dhS_dt_im[comp];
        deriv_singular_field.get(0, a1, b)[i] =
            dhS_drstar_re_tommy[comp] +
            std::complex<double>(0., 1.) * dhS_drstar_im_tommy[comp];
        deriv_singular_field.get(1, a1, b)[i] =
            dhS_dth_re_tommy[comp] +
            std::complex<double>(0., 1.) * dhS_dth_im_tommy[comp];
        deriv_singular_field.get(2, a1, b)[i] = 0.;
      }
    }
  }
  auto& fixed_source_re = get<::Tags::FixedSource<Tags::MModeRe>>(result);
  auto& fixed_source_im = get<::Tags::FixedSource<Tags::MModeIm>>(result);
  for (size_t i = 0; i < fixed_source_re.size(); i++) {
    fixed_source_re[i] = real(effective_source[i]);
    fixed_source_im[i] = imag(effective_source[i]);
  }
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

}  // namespace GrSelfForce::AnalyticData
