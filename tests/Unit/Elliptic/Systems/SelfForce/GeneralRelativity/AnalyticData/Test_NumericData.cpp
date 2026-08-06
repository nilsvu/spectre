// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <complex>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/AnalyticData/CircularOrbit.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/AnalyticData/NumericData.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/Equations.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/TMPL.hpp"

namespace GrSelfForce::AnalyticData {

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GrSelfForce.NumericData",
                  "[PointwiseFunctions][Unit]") {
  // Test that NumericData (h5-based) agrees with CircularOrbit (analytic)
  // for a 1st-order h5 dataset. Three checks:
  //   1. Seff inside the worldtube (2D mesh, field_is_regularized=true)
  //   2. hS and its normal derivative at the Left face (r = r_wt_left)
  //   3. hS and its normal derivative at the Bottom face (theta = theta_wt_bot)

  const std::string h5_file =
      "/u/namni/spectre_copy/data/D2G_m2_a0.6_r8.0.h5";
  if (not file_system::check_if_file_exists(h5_file)) {
    SUCCEED("Skipping test: h5 file not available on this machine");
    return;
  }
  const double bh_mass = 1.;
  const double bh_spin = 0.6;
  const double orbital_radius = 8.;
  const int m_mode = 2;
  // Transitions from RetRetV/T/U grid bounds in h5
  const std::array<double, 4> transitions{3.8667, 3.8667, 14.2, 14.2};

  // Worldtube face coordinates (from Seff.dat attributes)
  const double r_wt_left = 5.933333333333333;
  const double r_wt_right = 10.066666666666666;
  // thetaMin = pi/3 -> cos = 0.5;  thetaMax = 2*pi/3 -> cos = -0.5
  const double cos_wt_bot = 0.5;
  const double cos_wt_top = -0.5;

  const NumericData numeric_data{h5_file,    bh_mass,         bh_spin,
                                 orbital_radius, m_mode, transitions,
                                 true,      false};
  const CircularOrbit circular_orbit{bh_mass, bh_spin, orbital_radius,
                                     m_mode, transitions, true};

  const Approx approx = Approx::custom().epsilon(1.e-5).scale(1.);

  // -----------------------------------------------------------------------
  // Test 1: Seff on a 2D interior mesh (field_is_regularized=true uses the
  // high-resolution 500x500 Seff.dat grid)
  // -----------------------------------------------------------------------
  {
    const size_t npoints = 10;
    // Domain well inside worldtube bounds
    const domain::creators::Rectangle domain_creator{
        {{6.5, -0.3}}, {{9.5, 0.3}},
        {{0, 0}}, {{npoints, npoints}}, {{false, false}}};
    const auto domain = domain_creator.create_domain();
    const ElementMap<2, Frame::Inertial> element_map{ElementId<2>{0},
                                                     domain.blocks()[0]};
    const Mesh<2> mesh{npoints, Spectral::Basis::Legendre,
                       Spectral::Quadrature::Gauss};
    const auto x = element_map(logical_coordinates(mesh));

    const auto nd_vars =
        numeric_data.variables(x, NumericData::source_tags{}, true);
    const auto co_vars =
        circular_orbit.variables(x, CircularOrbit::source_tags{});
    const auto& nd_seff = get<::Tags::FixedSource<Tags::MMode>>(nd_vars);
    const auto& co_seff = get<::Tags::FixedSource<Tags::MMode>>(co_vars);
    for (size_t i = 0; i < nd_seff.size(); ++i) {
      CHECK_ITERABLE_CUSTOM_APPROX(nd_seff[i], co_seff[i], approx);
    }
  }

  // -----------------------------------------------------------------------
  // Test 2: hS and dhS/dr at Left face (r = r_wt_left, cos_theta varies)
  // NumericData fills singular_field from Left.dat (1D, theta-parameterized)
  // and stores dhS/dr in deriv_singular_field.get(0,...); theta-deriv = 0.
  // -----------------------------------------------------------------------
  {
    const size_t nface = 10;
    tnsr::I<DataVector, 2> x_left{};
    get<0>(x_left) = DataVector(nface, r_wt_left);
    get<1>(x_left) = DataVector(nface, 0.);
    for (size_t i = 0; i < nface; ++i) {
      // cos_theta strictly inside worldtube (avoid corners)
      get<1>(x_left)[i] =
          cos_wt_top + (cos_wt_bot - cos_wt_top) *
                           (static_cast<double>(i) + 1.) /
                           (static_cast<double>(nface) + 1.);
    }

    const auto nd_vars =
        numeric_data.variables(x_left, NumericData::source_tags{}, true);
    const auto co_vars =
        circular_orbit.variables(x_left, CircularOrbit::source_tags{});
    const auto& nd_hS = get<Tags::SingularField>(nd_vars);
    const auto& co_hS = get<Tags::SingularField>(co_vars);
    const auto& nd_dhS = get<::Tags::deriv<Tags::SingularField, tmpl::size_t<2>,
                                           Frame::Inertial>>(nd_vars);
    const auto& co_dhS = get<::Tags::deriv<Tags::SingularField, tmpl::size_t<2>,
                                           Frame::Inertial>>(co_vars);

    for (size_t i = 0; i < nd_hS.size(); ++i) {
      CHECK_ITERABLE_CUSTOM_APPROX(nd_hS[i], co_hS[i], approx);
    }
    // Only the r-derivative (index 0) is populated at Left face
    for (size_t a1 = 0; a1 < 4; ++a1) {
      for (size_t b = 0; b <= a1; ++b) {
        CHECK_ITERABLE_CUSTOM_APPROX(nd_dhS.get(0, a1, b),
                                     co_dhS.get(0, a1, b), approx);
      }
    }
  }

  // -----------------------------------------------------------------------
  // Test 3: hS and dhS/dtheta at Bottom face (cos_theta = cos_wt_bot, r varies)
  // NumericData fills from Bottom.dat (1D, r-parameterized)
  // and stores dhS/dtheta in deriv_singular_field.get(1,...); r-deriv = 0.
  // -----------------------------------------------------------------------
  {
    const size_t nface = 10;
    tnsr::I<DataVector, 2> x_bot{};
    get<0>(x_bot) = DataVector(nface, 0.);
    get<1>(x_bot) = DataVector(nface, cos_wt_bot);
    for (size_t i = 0; i < nface; ++i) {
      // r strictly inside worldtube (avoid corners)
      get<0>(x_bot)[i] =
          r_wt_left + (r_wt_right - r_wt_left) *
                          (static_cast<double>(i) + 1.) /
                          (static_cast<double>(nface) + 1.);
    }

    const auto nd_vars =
        numeric_data.variables(x_bot, NumericData::source_tags{}, true);
    const auto co_vars =
        circular_orbit.variables(x_bot, CircularOrbit::source_tags{});
    const auto& nd_hS = get<Tags::SingularField>(nd_vars);
    const auto& co_hS = get<Tags::SingularField>(co_vars);
    const auto& nd_dhS = get<::Tags::deriv<Tags::SingularField, tmpl::size_t<2>,
                                           Frame::Inertial>>(nd_vars);
    const auto& co_dhS = get<::Tags::deriv<Tags::SingularField, tmpl::size_t<2>,
                                           Frame::Inertial>>(co_vars);

    for (size_t i = 0; i < nd_hS.size(); ++i) {
      CHECK_ITERABLE_CUSTOM_APPROX(nd_hS[i], co_hS[i], approx);
    }
    // Only the theta-derivative (index 1) is populated at Bottom face
    for (size_t a1 = 0; a1 < 4; ++a1) {
      for (size_t b = 0; b <= a1; ++b) {
        CHECK_ITERABLE_CUSTOM_APPROX(nd_dhS.get(1, a1, b),
                                     co_dhS.get(1, a1, b), approx);
      }
    }
  }
  // -----------------------------------------------------------------------
  // Test 4: RetRet on a 2D interior mesh (field_is_regularized=false)
  // Domain in v-region (r < 3.8667), uses RetRetV interpolator.
  // Checks: fixed_source is nonzero. singular_field is exactly zero.
  // -----------------------------------------------------------------------
  {
    const size_t npoints = 10;
    const domain::creators::Rectangle domain_creator{
        {{2.0, -0.3}}, {{3.5, 0.3}},
        {{0, 0}}, {{npoints, npoints}}, {{false, false}}};
    const auto domain = domain_creator.create_domain();
    const ElementMap<2, Frame::Inertial> element_map{ElementId<2>{0},
                                                     domain.blocks()[0]};
    const Mesh<2> mesh{npoints, Spectral::Basis::Legendre,
                       Spectral::Quadrature::Gauss};
    const auto x = element_map(logical_coordinates(mesh));

    const auto nd_vars =
        numeric_data.variables(x, NumericData::source_tags{}, false);  // false!

    const auto& nd_seff = get<::Tags::FixedSource<Tags::MMode>>(nd_vars);
    const auto& nd_hS   = get<Tags::SingularField>(nd_vars);

    // Fixed source must be nonzero: RetRetV has 0.1 in all BL-frame components,
    // which maps to nonzero values after BL->VR conversion.
    bool any_nonzero = false;
    for (size_t i = 0; i < nd_seff.size(); ++i) {
      if (max(abs(real(nd_seff[i]))) > 0. or max(abs(imag(nd_seff[i]))) > 0.) {
        any_nonzero = true;
        break;
      }
    }
    CHECK(any_nonzero);

    // Singular field must be exactly zero outside worldtube.
    for (size_t i = 0; i < nd_hS.size(); ++i) {
      CHECK(nd_hS[i] == ComplexDataVector(npoints * npoints, 0.0));
    }
  }

}

}  // namespace GrSelfForce::AnalyticData
