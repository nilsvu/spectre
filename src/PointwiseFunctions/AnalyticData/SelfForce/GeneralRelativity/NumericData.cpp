// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/SelfForce/GeneralRelativity/NumericData.hpp"

#include <complex>
#include <cstddef>
#include <effsource_gr.hpp>
#include <utility>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/Tags.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/Helpers.hpp"
#include "NumericalAlgorithms/Interpolation/MultiLinearSpanInterpolation.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "PointwiseFunctions/AnalyticData/SelfForce/GeneralRelativity/CircularOrbitCoeffs.hpp"
#include "PointwiseFunctions/AnalyticData/SelfForce/GeneralRelativity/CircularOrbitConvertEffsource.hpp"
#include "PointwiseFunctions/GeneralRelativity/TortoiseCoordinates.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/CaptureForError.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace GrSelfForce::AnalyticData {

Interpolator load_data_from_file(const std::string& filename,
                                 const std::string& subfile_name) {
  // Open file
  CAPTURE_FOR_ERROR(filename);
  CAPTURE_FOR_ERROR(subfile_name);
  h5::H5File<h5::AccessType::ReadOnly> h5file(filename);
  const auto& datfile = h5file.get<h5::Dat>(subfile_name);

  // Read metadata
  const size_t num_radial_points =
      h5::read_value_attribute<size_t>(datfile.dataset_id(), "rstarNum");
  const size_t num_angular_points =
      h5::read_value_attribute<size_t>(datfile.dataset_id(), "thetaNum");
  const double delta_r_star =
      h5::read_value_attribute<double>(datfile.dataset_id(), "drstar");
  const double r_star_min =
      h5::read_value_attribute<double>(datfile.dataset_id(), "rstarMin");
  const double delta_theta =
      h5::read_value_attribute<double>(datfile.dataset_id(), "dtheta");
  const double theta_min =
      h5::read_value_attribute<double>(datfile.dataset_id(), "thetaMin");

  // Construct coordinates
  std::vector<double> r_star(num_radial_points);
  for (size_t i = 0; i < num_radial_points; ++i) {
    r_star[i] = r_star_min + i * delta_r_star;
  }
  std::vector<double> theta(num_angular_points);
  for (size_t j = 0; j < num_angular_points; ++j) {
    theta[j] = theta_min + j * delta_theta;
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
  // Data is stored in (var, r_star, theta) order with var varying fastest
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
  return {std::move(r_star), std::move(theta), std::move(flat_data)};
}

std::array<Interpolator, 4> load_all_data(const std::string& filename) {
  return {{load_data_from_file(filename, "RetRetV"),
           load_data_from_file(filename, "RetRetT"),
           load_data_from_file(filename, "RetRetU"),
           load_data_from_file(filename, "Seff")}};
}

NumericData::NumericData(
    std::string filename, const double black_hole_mass,
    const double black_hole_spin, const double orbital_radius,
    const int m_mode_number,
    const std::array<double, 2> hyperboloidal_slicing_transitions)
    : filename_(std::move(filename)),
      circular_orbit_(black_hole_mass, black_hole_spin, orbital_radius,
                      m_mode_number,
                      {{{hyperboloidal_slicing_transitions[0],
                         hyperboloidal_slicing_transitions[0],
                         hyperboloidal_slicing_transitions[1],
                         hyperboloidal_slicing_transitions[1]}}}) {
  interpolators_ = load_all_data(filename_);
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
  const double orbital_radius_ = circular_orbit_.orbital_radius();
  const int m_mode_number_ = circular_orbit_.m_mode_number();
  const double a = black_hole_spin_ * black_hole_mass_;
  const double M = black_hole_mass_;
  const double r_0 = orbital_radius_;
  const double r_plus = M * (1. + sqrt(1. - square(black_hole_spin_)));
  const double r_minus = M * (1. - sqrt(1. - square(black_hole_spin_)));
  const auto& r_star = get<0>(x);
  const auto& theta = get<1>(x);
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
    const double any_r_star = r_star[0];
    if (any_r_star < hyperboloidal_slicing_transitions[0]) {
      return interpolators_[0].interpolator;
    } else if (any_r_star < hyperboloidal_slicing_transitions[2]) {
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
    if (r_star[i] < interpolator_bounds[0][0] or
        r_star[i] > interpolator_bounds[0][1]) {
      ERROR("Requested r* = " << r_star[i]
                              << " outside of interpolation bounds ["
                              << interpolator_bounds[0][0] << ", "
                              << interpolator_bounds[0][1] << "]");
    }
    if ((theta[i] < interpolator_bounds[1][0] or
         theta[i] > interpolator_bounds[1][1]) and
        // Allow extrapolation to the poles
        // TODO: check the error that this extrapolation incurs
        not(equal_within_roundoff(theta[i], 0., 0.15) or
            equal_within_roundoff(theta[i], M_PI, 0.15))) {
      ERROR("Requested theta = " << theta[i]
                                 << " outside of interpolation bounds ["
                                 << interpolator_bounds[1][0] << ", "
                                 << interpolator_bounds[1][1] << "]");
    }
    const auto weights = interpolator.get_weights(r_star[i], theta[i]);
    for (size_t k = 0; k < 10; ++k) {
      effective_source[k][i] = interpolator.interpolate(weights, 2 * k) +
                               std::complex<double>(0., 1.) *
                                   interpolator.interpolate(weights, 2 * k + 1);
    }
  }
  // Fill singular field with zeros for now
  for (size_t i = 0; i < singular_field.size(); i++) {
    singular_field[i] = 0.;
  }
  for (size_t i = 0; i < deriv_singular_field.size(); i++) {
    deriv_singular_field[i] = 0.;
  }
  return result;
}

void NumericData::pup(PUP::er& p) {
  elliptic::analytic_data::Background::pup(p);
  elliptic::analytic_data::InitialGuess::pup(p);
  p | filename_;
  p | circular_orbit_;
  if (p.isUnpacking()) {
    interpolators_ = load_all_data(filename_);
  }
}

bool operator==(const NumericData& lhs, const NumericData& rhs) {
  return lhs.filename_ == rhs.filename_ and
         lhs.circular_orbit_ == rhs.circular_orbit_;
}

bool operator!=(const NumericData& lhs, const NumericData& rhs) {
  return not(lhs == rhs);
}

PUP::able::PUP_ID NumericData::my_PUP_ID = 0;  // NOLINT

}  // namespace GrSelfForce::AnalyticData
