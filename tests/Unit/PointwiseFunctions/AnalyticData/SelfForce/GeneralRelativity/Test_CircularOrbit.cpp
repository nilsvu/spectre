// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <complex>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/Equations.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "PointwiseFunctions/AnalyticData/SelfForce/GeneralRelativity/CircularOrbit.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace GrSelfForce::AnalyticData {

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GrSelfForce.CircularOrbit",
                  "[PointwiseFunctions][Unit]") {
  // Set up a rectangular grid
  const double theta_offset = M_PI / 8.;
  const double delta_theta = M_PI / 40.;
  const double rstar_offset = 0.;
  const double delta_rstar = 5.;
  const size_t npoints = 20;
  const domain::creators::Brick domain_creator{
      {{rstar_offset, M_PI_2 + theta_offset, 0.}},
      {{rstar_offset + delta_rstar, M_PI_2 + theta_offset + delta_theta, 1.}},
      {{0, 0, 0}},
      {{npoints, npoints, 1}},
      {{false, false}}};
  const auto domain = domain_creator.create_domain();
  const auto& block = domain.blocks()[0];
  const ElementId<3> element_id{0};
  const ElementMap<3, Frame::Inertial> element_map{element_id, block};
  const Mesh<3> mesh{{{npoints, npoints, 1}},
                     Spectral::Basis::Legendre,
                     Spectral::Quadrature::Gauss};
  const auto xi = logical_coordinates(mesh);
  const auto x = element_map(xi);
  const auto inv_jacobian = element_map.inv_jacobian(xi);
  CAPTURE(x);

  // Get the analytic fields
  const auto circular_orbit = CircularOrbit{1., 0.9, 6., 0};
  CAPTURE(circular_orbit.puncture_position());
  const auto background =
      circular_orbit.variables(x, CircularOrbit::background_tags{});
  const auto& alpha = get<Tags::Alpha>(background);
  const auto& beta = get<Tags::Beta>(background);
  const auto& gamma_rstar = get<Tags::GammaRstar>(background);
  const auto& gamma_theta = get<Tags::GammaTheta>(background);
  const auto vars = circular_orbit.variables(x, CircularOrbit::source_tags{});
  const auto& singular_field_complex = get<Tags::SingularField>(vars);
  tnsr::aa<DataVector, 3> singular_field_re{};
  tnsr::aa<DataVector, 3> singular_field_im{};
  for (size_t i = 0; i < singular_field_complex.size(); ++i) {
    singular_field_re[i] = real(singular_field_complex[i]);
    singular_field_im[i] = imag(singular_field_complex[i]);
  }
  const auto& deriv_singular_field_complex =
      get<::Tags::deriv<Tags::SingularField, tmpl::size_t<3>, Frame::Inertial>>(
          vars);
  tnsr::iaa<DataVector, 3> deriv_singular_field_re{};
  tnsr::iaa<DataVector, 3> deriv_singular_field_im{};
  for (size_t i = 0; i < deriv_singular_field_complex.size(); ++i) {
    deriv_singular_field_re[i] = real(deriv_singular_field_complex[i]);
    deriv_singular_field_im[i] = imag(deriv_singular_field_complex[i]);
  }
  const auto& effective_source_re =
      get<::Tags::FixedSource<Tags::MModeRe>>(vars);
  const auto& effective_source_im =
      get<::Tags::FixedSource<Tags::MModeIm>>(vars);

  // Take numeric derivative
  const auto numeric_deriv_singular_field_re =
      partial_derivative(singular_field_re, mesh, inv_jacobian);
  const auto numeric_deriv_singular_field_im =
      partial_derivative(singular_field_im, mesh, inv_jacobian);
  Approx custom_approx = Approx::custom().epsilon(1.e-4).scale(1.);
  for (size_t i = 0; i < deriv_singular_field_complex.size(); ++i) {
    CAPTURE(i);
    CHECK_ITERABLE_CUSTOM_APPROX(numeric_deriv_singular_field_re[i],
                                 deriv_singular_field_re[i], custom_approx);
    CHECK_ITERABLE_CUSTOM_APPROX(numeric_deriv_singular_field_im[i],
                                 deriv_singular_field_im[i], custom_approx);
  }

  Variables<
      tmpl::list<::Tags::Flux<Tags::MModeRe, tmpl::size_t<3>, Frame::Inertial>,
                 ::Tags::Flux<Tags::MModeIm, tmpl::size_t<3>, Frame::Inertial>>>
      fluxes{mesh.number_of_grid_points()};
  auto& flux_singular_field_re =
      get<::Tags::Flux<Tags::MModeRe, tmpl::size_t<3>, Frame::Inertial>>(
          fluxes);
  auto& flux_singular_field_im =
      get<::Tags::Flux<Tags::MModeIm, tmpl::size_t<3>, Frame::Inertial>>(
          fluxes);
  GrSelfForce::Fluxes::apply(make_not_null(&flux_singular_field_re),
                             make_not_null(&flux_singular_field_im), alpha, {},
                             {}, deriv_singular_field_re,
                             deriv_singular_field_im);
  auto divs = divergence(fluxes, mesh, inv_jacobian);
  auto& scalar_eqn_re = get<::Tags::div<
      ::Tags::Flux<Tags::MModeRe, tmpl::size_t<3>, Frame::Inertial>>>(divs);
  auto& scalar_eqn_im = get<::Tags::div<
      ::Tags::Flux<Tags::MModeIm, tmpl::size_t<3>, Frame::Inertial>>>(divs);
  for (size_t i = 0; i < scalar_eqn_re.size(); ++i) {
    scalar_eqn_re[i] *= -1.;
    scalar_eqn_im[i] *= -1.;
  }
  GrSelfForce::Sources::apply(make_not_null(&scalar_eqn_re),
                              make_not_null(&scalar_eqn_im), beta, gamma_rstar,
                              gamma_theta, singular_field_re, singular_field_im,
                              flux_singular_field_re, flux_singular_field_im);
  for (size_t i = 0; i < scalar_eqn_re.size(); ++i) {
    CAPTURE(i);
    CHECK_ITERABLE_CUSTOM_APPROX(scalar_eqn_re[i], effective_source_re[i],
                                 custom_approx);
    CHECK_ITERABLE_CUSTOM_APPROX(scalar_eqn_im[i], effective_source_im[i],
                                 custom_approx);
  }
}

}  // namespace GrSelfForce::AnalyticData
