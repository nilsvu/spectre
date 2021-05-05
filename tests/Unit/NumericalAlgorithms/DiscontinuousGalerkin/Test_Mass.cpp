// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Mass.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace dg {
namespace {

template <size_t Dim, typename... CoordMaps>
void test_mass(const Mesh<Dim>& mesh,
               const domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                                           CoordMaps...>& coordinate_map,
               const DataVector& scalar_field,
               const DataVector& expected_massive_scalar_field) noexcept {
  const size_t num_grid_points = mesh.number_of_grid_points();
  const auto logical_coords = logical_coordinates(mesh);
  const auto jacobian = coordinate_map.jacobian(logical_coords);
  const auto det_jacobian = determinant(jacobian);
  {
    INFO("Test with DataVector");
    auto result = scalar_field;
    result *= get(det_jacobian);
    apply_mass(make_not_null(&result), mesh);
    CHECK_ITERABLE_APPROX(result, expected_massive_scalar_field);
  }
  {
    INFO("Test with Variables");
    using tag1 = ::Tags::TempScalar<0>;
    using tag2 = ::Tags::TempScalar<1>;
    Variables<tmpl::list<tag1, tag2>> vars{num_grid_points};
    get<tag1>(vars) = Scalar<DataVector>(scalar_field);
    get<tag2>(vars) = Scalar<DataVector>(scalar_field);
    vars *= get(det_jacobian);
    apply_mass(make_not_null(&vars), mesh);
    CHECK_ITERABLE_APPROX(get(get<tag1>(vars)), expected_massive_scalar_field);
    CHECK_ITERABLE_APPROX(get(get<tag2>(vars)), expected_massive_scalar_field);
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.DiscontinuousGalerkin.Mass",
                  "[NumericalAlgorithms][Unit]") {
  using Affine = domain::CoordinateMaps::Affine;
  using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

  {
    INFO("1D");
    const Mesh<1> mesh{
        {{4}}, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto};
    const auto coord_map =
        domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(
            Affine{-1.0, 1.0, -0.3, 0.7});
    test_mass(mesh, coord_map, {1., 2., 3., 4.},
              DataVector{1., 10., 15., 4.} / 12.);
  }
  {
    INFO("2D");
    const Mesh<2> mesh{{{4, 2}},
                       Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
    const auto coord_map =
        domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(Affine2D{
            Affine{-1.0, 1.0, -0.3, 0.7}, Affine{-1.0, 1.0, 0.3, 0.55}});
    test_mass(mesh, coord_map, {1., 2., 3., 4., 5., 6., 7., 8.},
              DataVector{1., 10., 15., 4., 5., 30., 35., 8.} / 96.);
  }
  {
    INFO("3D");
    const Mesh<3> mesh{{{4, 2, 3}},
                       Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
    const auto coord_map =
        domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(
            Affine3D{Affine{-1.0, 1.0, -0.3, 0.7}, Affine{-1.0, 1.0, 0.3, 0.55},
                     Affine{-1.0, 1.0, 2.3, 2.8}});
    test_mass(mesh, coord_map,
              {1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12.,
               13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.},
              DataVector{1.,  10.,  15.,  4.,  5.,  30.,  35.,  8.,
                         36., 200., 220., 48., 52., 280., 300., 64.,
                         17., 90.,  95.,  20., 21., 110., 115., 24.} /
                  1152.);
  }
}

}  // namespace dg
