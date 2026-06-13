// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/SelfForce/GeneralRelativity/AnalyticData/NumericData.hpp"

#include <algorithm>
#include <complex>
#include <cstddef>
// #include <effsource_gr.hpp>
#include <effsource_comoving.hpp>
#include <utility>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/AnalyticData/CircularOrbitConvertEffsource.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/Tags.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/Helpers.hpp"
#include "NumericalAlgorithms/Interpolation/MultiLinearSpanInterpolation.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "PointwiseFunctions/GeneralRelativity/TortoiseCoordinates.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/CaptureForError.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace GrSelfForce::AnalyticData {

namespace {

Interpolator load_data_from_file(const std::string& filename,
                                 const std::string& subfile_name) {
  // Open file
  CAPTURE_FOR_ERROR(filename);
  CAPTURE_FOR_ERROR(subfile_name);
  const h5::H5File<h5::AccessType::ReadOnly> h5file(filename);
  const auto& datfile = h5file.get<h5::Dat>(subfile_name);

  // Read metadata
  const auto num_radial_points =
      h5::read_value_attribute<size_t>(datfile.dataset_id(), "rNum");
  const auto num_angular_points =
      h5::read_value_attribute<size_t>(datfile.dataset_id(), "thetaNum");
  const auto delta_r =
      h5::read_value_attribute<double>(datfile.dataset_id(), "dr");
  const auto r_min =
      h5::read_value_attribute<double>(datfile.dataset_id(), "rMin");
  const auto delta_theta =
      h5::read_value_attribute<double>(datfile.dataset_id(), "dtheta");
  const auto theta_min =
      h5::read_value_attribute<double>(datfile.dataset_id(), "thetaMin");

  // Construct coordinates
  std::vector<double> r(num_radial_points);
  for (size_t i = 0; i < num_radial_points; ++i) {
    r[i] = r_min + static_cast<double>(i) * delta_r;
  }
  std::vector<double> theta(num_angular_points);
  for (size_t j = 0; j < num_angular_points; ++j) {
    theta[j] = theta_min + static_cast<double>(j) * delta_theta;
  }

  // Load data
  static constexpr size_t NumberOfVars = 20;
  const auto matrix_data = datfile.get_data();
  if (matrix_data.rows() != num_radial_points * num_angular_points) {
    const size_t rows = matrix_data.rows();
    CAPTURE_FOR_ERROR(rows);
    CAPTURE_FOR_ERROR(num_radial_points);
    CAPTURE_FOR_ERROR(num_angular_points);
    ERROR("Number of points in data file does not match header information");
  }
  if (matrix_data.columns() != NumberOfVars) {
    const size_t columns = matrix_data.columns();
    CAPTURE_FOR_ERROR(columns);
    CAPTURE_FOR_ERROR(NumberOfVars);
    ERROR("Unexpected number of variables in data file");
  }
  // Data is stored in (var, r, theta) order with var varying fastest
  std::vector<double> flat_data(NumberOfVars * num_radial_points *
                                num_angular_points);
  for (size_t i = 0; i < num_radial_points * num_angular_points; i++) {
    for (size_t k = 0; k < NumberOfVars; ++k) {
      flat_data[i * NumberOfVars + k] = matrix_data(i, k);
    }
  }

  // Close file
  h5file.close();

  // Construct interpolator
  return {std::move(r), std::move(theta), std::move(flat_data)};
}

Interpolator1D load_data_from_file_1D(const std::string& filename,
                                      const std::string& subfile_name,
                                      const std::string& num_points_attr,
                                      const std::string& coord_min_attr,
                                      const std::string& delta_coord_attr) {
  CAPTURE_FOR_ERROR(filename);
  CAPTURE_FOR_ERROR(subfile_name);
  const h5::H5File<h5::AccessType::ReadOnly> h5file(filename);
  const auto& datfile = h5file.get<h5::Dat>(subfile_name);

  const auto num_points =
      h5::read_value_attribute<size_t>(datfile.dataset_id(), num_points_attr);
  const auto coord_min =
      h5::read_value_attribute<double>(datfile.dataset_id(), coord_min_attr);
  const auto delta_coord =
      h5::read_value_attribute<double>(datfile.dataset_id(), delta_coord_attr);

  std::vector<double> coord(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    coord[i] = coord_min + static_cast<double>(i) * delta_coord;
  }

  static constexpr size_t NumberOfVars = 40;
  const auto matrix_data = datfile.get_data();
  if (matrix_data.rows() != num_points) {
    const size_t rows = matrix_data.rows();
    CAPTURE_FOR_ERROR(rows);
    CAPTURE_FOR_ERROR(num_points);
    ERROR("Number of points in data file does not match header information");
  }
  if (matrix_data.columns() != NumberOfVars) {
    const size_t columns = matrix_data.columns();
    CAPTURE_FOR_ERROR(columns);
    CAPTURE_FOR_ERROR(NumberOfVars);
    ERROR("Unexpected number of variables in data file");
  }
  std::vector<double> flat_data(NumberOfVars * num_points);
  for (size_t i = 0; i < num_points; ++i) {
    for (size_t k = 0; k < NumberOfVars; ++k) {
      flat_data[i * NumberOfVars + k] = matrix_data(i, k);
    }
  }

  h5file.close();
  return {std::move(coord), std::move(flat_data)};
}

std::array<Interpolator, 4> load_all_data(const std::string& filename) {
  return {{load_data_from_file(filename, "RetRetV"),
           load_data_from_file(filename, "RetRetT"),
           load_data_from_file(filename, "RetRetU"),
           load_data_from_file(filename, "Seff")}};
}

std::array<Interpolator1D, 4> load_all_boundary_data(
    const std::string& filename) {
  return {{load_data_from_file_1D(filename, "Left", "thetaNum", "thetaMin",
                                  "dtheta"),
           load_data_from_file_1D(filename, "Right", "thetaNum", "thetaMin",
                                  "dtheta"),
           load_data_from_file_1D(filename, "Bottom", "rNum", "rMin", "dr"),
           load_data_from_file_1D(filename, "Top", "rNum", "rMin", "dr")}};
}

}  // namespace

NumericData::NumericData(
    std::string filename, const double black_hole_mass,
    const double black_hole_spin, const double orbital_radius,
    const int m_mode_number,
    const std::array<double, 4> hyperboloidal_slicing_transitions,
    const bool penetrating_horizon, const bool pi_2_rotation)
    : filename_(std::move(filename)),
      circular_orbit_(black_hole_mass, black_hole_spin, orbital_radius,
                      m_mode_number,
                      {{{hyperboloidal_slicing_transitions[0],
                         hyperboloidal_slicing_transitions[1],
                         hyperboloidal_slicing_transitions[2],
                         hyperboloidal_slicing_transitions[3]}}},
                      penetrating_horizon),
      pi_2_rotation_(pi_2_rotation) {
  interpolators_ = load_all_data(filename_);
  boundary_interpolators_ = load_all_boundary_data(filename_);
}

NumericData::NumericData(CkMigrateMessage* m)
    : elliptic::analytic_data::Background(m),
      elliptic::analytic_data::InitialGuess(m) {}

tnsr::I<double, 2> NumericData::puncture_position() const {
  return circular_orbit_.puncture_position();
}

// Background
tuples::TaggedTuple<Tags::Alpha, Tags::Beta, Tags::GammaRstar, Tags::GammaTheta>
NumericData::variables(
    const tnsr::I<DataVector, 2>& x,
    tmpl::list<Tags::Alpha, Tags::Beta, Tags::GammaRstar, Tags::GammaTheta>
        meta) const {
  return circular_orbit_.variables(x, meta);
}

// Initial guess
tuples::TaggedTuple<Tags::MMode> NumericData::variables(
    const tnsr::I<DataVector, 2>& x, tmpl::list<Tags::MMode> meta) const {
  return circular_orbit_.variables(x, meta);
}

// Fixed sources
tuples::TaggedTuple<
    ::Tags::FixedSource<Tags::MMode>, Tags::SingularField,
    ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>,
    Tags::BoyerLindquistRadius>
NumericData::variables(
    const tnsr::I<DataVector, 2>& x,
    tmpl::list<
        ::Tags::FixedSource<Tags::MMode>, Tags::SingularField,
        ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>,
        Tags::BoyerLindquistRadius> /*meta*/,
    const bool field_is_regularized) const {
  const double black_hole_spin_ = circular_orbit_.black_hole_spin();
  const double black_hole_mass_ = circular_orbit_.black_hole_mass();
  // const double orbital_radius_ = circular_orbit_.orbital_radius();
  const int m_mode_number_ = circular_orbit_.m_mode_number();
  const double a = black_hole_spin_ * black_hole_mass_;
  const double M = black_hole_mass_;
  // const double r_0 = orbital_radius_;
  const double r_plus = M * (1. + sqrt(1. - square(black_hole_spin_)));
  const double r_minus = M * (1. - sqrt(1. - square(black_hole_spin_)));
  const auto& r = get<0>(x);
  const DataVector theta =
      acos(get<1>(x));  // get<1>(x) is cos_theta when penetrating_horizon
  const DataVector r_minus_r_plus = r - r_plus;
  const DataVector delta = r_minus_r_plus * (r - r_minus);
  const DataVector delta_phi = m_mode_number_ * a / (r_plus - r_minus) *
                               log((r - r_plus) / (r - r_minus));
  const ComplexDataVector rotation =
      cos(delta_phi) + std::complex<double>(0., 1.) * sin(delta_phi);
  tuples::TaggedTuple<
      ::Tags::FixedSource<Tags::MMode>, Tags::SingularField,
      ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>,
      Tags::BoyerLindquistRadius>
      result{};
  get(get<Tags::BoyerLindquistRadius>(result)) = r;
  const size_t num_points = get<0>(x).size();
  tnsr::aa<ComplexDataVector, 3>& effective_source =
      get<::Tags::FixedSource<Tags::MMode>>(result);
  tnsr::aa<ComplexDataVector, 3>& singular_field =
      get<Tags::SingularField>(result);
  for (size_t i = 0; i < singular_field.size(); i++) {
    effective_source[i].destructive_resize(num_points);
    singular_field[i].destructive_resize(num_points);
  }
  auto& deriv_singular_field =
      get<::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>>(
          result);
  for (size_t i = 0; i < deriv_singular_field.size(); i++) {
    deriv_singular_field[i].destructive_resize(num_points);
  }
  // Decide which interpolator to use based on position
  const auto& hyperboloidal_slicing_transitions =
      circular_orbit_.hyperboloidal_slicing_transitions().value();
  const auto& interpolator = [&]() {
    if (field_is_regularized) {
      return interpolators_[3].interpolator;
    }
    const double any_r = r[0];
    if (any_r < hyperboloidal_slicing_transitions[0]) {
      return interpolators_[0].interpolator;
    } else if (any_r < hyperboloidal_slicing_transitions[2]) {
      return interpolators_[1].interpolator;
    } else {
      return interpolators_[2].interpolator;
    }
  }();
  const std::array<std::array<double, 2>, 2> interpolator_bounds{
      {{{interpolator.lower_bound(0), interpolator.upper_bound(0)}},
       {{interpolator.lower_bound(1), interpolator.upper_bound(1)}}}};
  // Interpolate data
  // Ordering of components:
  // tt, tr, ttheta, tphi, rr, rtheta, rphi, theta theta, theta phi, phi phi
  for (size_t i = 0; i < num_points; ++i) {
    const double r_clamped =
        std::clamp(r[i], interpolator_bounds[0][0], interpolator_bounds[0][1]);
    if (not equal_within_roundoff(r[i], r_clamped)) {
      ERROR("Requested r = " << r[i] << " outside of interpolation bounds ["
                             << interpolator_bounds[0][0] << ", "
                             << interpolator_bounds[0][1] << "]");
    }
    const double theta_clamped = std::clamp(theta[i], interpolator_bounds[1][0],
                                            interpolator_bounds[1][1]);
    if ((theta[i] < interpolator_bounds[1][0] or
         theta[i] > interpolator_bounds[1][1]) and
        // Allow extrapolation to the poles
        not(equal_within_roundoff(theta[i], 0., 0.15) or
            equal_within_roundoff(theta[i], M_PI, 0.15) or
            equal_within_roundoff(theta[i], interpolator_bounds[1][0]) or
            equal_within_roundoff(theta[i], interpolator_bounds[1][1]))) {
      ERROR("Requested theta = " << theta[i]
                                 << " outside of interpolation bounds ["
                                 << interpolator_bounds[1][0] << ", "
                                 << interpolator_bounds[1][1] << "]");
    }
    const auto weights = interpolator.get_weights(r_clamped, theta_clamped);
    // Load raw BL-frame source from h5 (upper-triangular ordering: k=0..9)
    std::array<double, 10> src_re_arr{};
    std::array<double, 10> src_im_arr{};
    std::array<double, 10> src_conv_re{};
    std::array<double, 10> src_conv_im{};
    for (size_t k = 0; k < 10; ++k) {
      gsl::at(src_re_arr, k) = interpolator.interpolate(weights, 2 * k);
      gsl::at(src_im_arr, k) = interpolator.interpolate(weights, 2 * k + 1);
      if (pi_2_rotation_) {
        const std::complex<double> rotated =
            (gsl::at(src_re_arr, k) +
             std::complex<double>(0., 1.) * gsl::at(src_im_arr, k)) *
            (2. * M_PI * rotation[i]);
        gsl::at(src_re_arr, k) = rotated.real();
        gsl::at(src_im_arr, k) = rotated.imag();
      }
    }
    // Convert from BL frame to VR (comoving ingoing EF) frame
    detail::convert_effsource_Seff_vr(m_mode_number_, a, r[i], get<1>(x)[i],
                                      src_re_arr, src_im_arr, src_conv_re,
                                      src_conv_im);
    // Store into SpECTRE lower-triangular ordering
    for (size_t a1 = 0; a1 < 4; ++a1) {
      for (size_t b = 0; b <= a1; ++b) {
        const size_t comp = tnsr::aa<ComplexDataVector, 3>::get_storage_index(
            std::array<size_t, 2>{{a1, b}});
        effective_source.get(a1, b)[i] =
            gsl::at(src_conv_re, comp) +
            std::complex<double>(0., 1.) * gsl::at(src_conv_im, comp);
      }
    }
  }

  if (field_is_regularized) {
    const double r_wt_left = interpolators_[3].interpolator.lower_bound(0);
    const double r_wt_right = interpolators_[3].interpolator.upper_bound(0);
    const double theta_wt_bot = interpolators_[3].interpolator.lower_bound(1);
    const double theta_wt_top = interpolators_[3].interpolator.upper_bound(1);

    {
      for (auto& component : singular_field) {
        component = ComplexDataVector(num_points, 0.0);
      }
      for (auto& component : deriv_singular_field) {
        component = ComplexDataVector(num_points, 0.0);
      }
    }

    // Override face/mortar points with h5 boundary data for consistency with
    // the numerical Seff.
    for (size_t i = 0; i < num_points; ++i) {
      const bool on_left = equal_within_roundoff(r[i], r_wt_left);
      const bool on_right = equal_within_roundoff(r[i], r_wt_right);
      const bool on_bottom = equal_within_roundoff(theta[i], theta_wt_bot);
      const bool on_top = equal_within_roundoff(theta[i], theta_wt_top);

      if (not(on_left or on_right or on_bottom or on_top)) {
        continue;
      }

      const auto& binterp = on_left    ? boundary_interpolators_[0].interpolator
                            : on_right ? boundary_interpolators_[1].interpolator
                            : on_bottom
                                ? boundary_interpolators_[2].interpolator
                                : boundary_interpolators_[3].interpolator;
      const double coord_i = (on_left or on_right) ? theta[i] : r[i];
      const auto weights = binterp.get_weights(coord_i);

      // Load raw BL-frame hS and its normal derivative from h5
      // (upper-triangular ordering k=0..9) and convert to VR frame.
      // NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
      std::array<double, 10> hS_re_arr{};
      std::array<double, 10> hS_im_arr{};
      std::array<double, 10> dhS_re_arr{};
      std::array<double, 10> dhS_im_arr{};
      std::array<double, 10> hS_conv_re{};
      std::array<double, 10> hS_conv_im{};
      std::array<double, 10> dhS_conv_re{};
      std::array<double, 10> dhS_conv_im{};
      for (size_t k = 0; k < 10; ++k) {
        hS_re_arr[k] = binterp.interpolate(weights, 2 * k);
        hS_im_arr[k] = binterp.interpolate(weights, 2 * k + 1);
        dhS_re_arr[k] = binterp.interpolate(weights, 20 + 2 * k);
        dhS_im_arr[k] = binterp.interpolate(weights, 20 + 2 * k + 1);
        if (pi_2_rotation_) {
          const std::complex<double> rotated_hS =
              (hS_re_arr[k] + std::complex<double>(0., 1.) * hS_im_arr[k]) *
              (2. * M_PI * rotation[i]);
          hS_re_arr[k] = rotated_hS.real();
          hS_im_arr[k] = rotated_hS.imag();
          std::complex<double> rotated_dhS;
          const std::complex<double> dhS =
              (dhS_re_arr[k] + std::complex<double>(0., 1.) * dhS_im_arr[k]);
          rotated_dhS = 2 * M_PI * rotation[i] * dhS;
          dhS_re_arr[k] = rotated_dhS.real();
          dhS_im_arr[k] = rotated_dhS.imag();
        }
      }
      detail::convert_effsource_psi_vr(m_mode_number_, a, r[i], get<1>(x)[i],
                                       hS_re_arr, hS_im_arr, hS_conv_re,
                                       hS_conv_im);
      if (on_left or on_right) {
        // Left/Right: normal is r, columns 20-39 are dr derivative
        detail::convert_effsource_dpsidr_vr(
            m_mode_number_, a, r[i], get<1>(x)[i], hS_re_arr, hS_im_arr,
            dhS_re_arr, dhS_im_arr, dhS_conv_re, dhS_conv_im);
      } else {
        // Bottom/Top: normal is theta, columns 20-39 are dtheta derivative;
        // conversion also maps d/dtheta -> d/d(cos_theta)
        detail::convert_effsource_dpsidz_vr(
            m_mode_number_, a, r[i], get<1>(x)[i], hS_re_arr, hS_im_arr,
            dhS_re_arr, dhS_im_arr, dhS_conv_re, dhS_conv_im);
      }
      for (size_t a1 = 0; a1 < 4; ++a1) {
        for (size_t b = 0; b <= a1; ++b) {
          const size_t comp = tnsr::aa<ComplexDataVector, 3>::get_storage_index(
              std::array<size_t, 2>{{a1, b}});
          singular_field.get(a1, b)[i] =
              hS_conv_re[comp] +
              std::complex<double>(0., 1.) * hS_conv_im[comp];
          if (on_left or on_right) {
            deriv_singular_field.get(0, a1, b)[i] =
                dhS_conv_re[comp] +
                std::complex<double>(0., 1.) * dhS_conv_im[comp];
            deriv_singular_field.get(1, a1, b)[i] = 0.;
          } else {
            deriv_singular_field.get(0, a1, b)[i] = 0.;
            deriv_singular_field.get(1, a1, b)[i] =
                dhS_conv_re[comp] +
                std::complex<double>(0., 1.) * dhS_conv_im[comp];
          }
        }
      }
      // NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
    }
  } else {
    for (size_t i = 0; i < singular_field.size(); i++) {
      singular_field[i] = 0.;
    }
    for (size_t i = 0; i < deriv_singular_field.size(); i++) {
      deriv_singular_field[i] = 0.;
    }
  }
  return result;
}

void NumericData::pup(PUP::er& p) {
  elliptic::analytic_data::Background::pup(p);
  elliptic::analytic_data::InitialGuess::pup(p);
  p | filename_;
  p | circular_orbit_;
  p | pi_2_rotation_;
  if (p.isUnpacking()) {
    interpolators_ = load_all_data(filename_);
    boundary_interpolators_ = load_all_boundary_data(filename_);
  }
}

bool operator==(const NumericData& lhs, const NumericData& rhs) {
  return lhs.filename_ == rhs.filename_ and
         lhs.circular_orbit_ == rhs.circular_orbit_ and
         lhs.pi_2_rotation_ == rhs.pi_2_rotation_;
}

bool operator!=(const NumericData& lhs, const NumericData& rhs) {
  return not(lhs == rhs);
}

PUP::able::PUP_ID NumericData::my_PUP_ID = 0;  // NOLINT

}  // namespace GrSelfForce::AnalyticData
