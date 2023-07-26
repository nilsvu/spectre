// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/Rectangle.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Elliptic/FiniteDifference/FdOperator.hpp"
#include "NumericalAlgorithms/LinearSolver/BuildMatrix.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

namespace elliptic::fd {

SPECTRE_TEST_CASE("Unit.Elliptic.FdOperator", "[NumericalAlgorithms][Unit]") {
  {
    INFO("1D");
    domain::creators::Interval domain_creator{{{-1.}}, {{1.}}, {{2}}, {{4}}};
    const auto domain = domain_creator.create_domain();
    const ElementId<1> element_id{0, {{SegmentId{2, 1}}}};
    const Mesh<1> mesh{5, Spectral::Basis::FiniteDifference,
                       Spectral::Quadrature::CellCentered};
    const auto logical_coords = logical_coordinates(mesh);
    const ElementMap<1, Frame::Inertial> element_map{
        element_id,
        domain.blocks()[element_id.block_id()].stationary_map().get_clone()};
    const auto inertial_coords = element_map(logical_coords);
    const auto inv_jacobian = element_map.inv_jacobian(logical_coords);
    const DataVector u = 2. * square(get<0>(inertial_coords));
    // const DataVector u = 2. * cube(get<0>(inertial_coords));
    // const DataVector expected_ddu{mesh.number_of_grid_points(), 4.};
    // const DataVector expected_ddu = 12. * get<0>(inertial_coords);
    DataVector ddu{mesh.number_of_grid_points()};
    apply_operator(make_not_null(&ddu), u, mesh, inv_jacobian);
    // CHECK_ITERABLE_APPROX(ddu, expected_ddu);

    const size_t num_points = mesh.number_of_grid_points();
    Matrix matrix(num_points, num_points);
    DataVector operand_buffer{num_points, 0.};
    DataVector result_buffer{num_points};
    LinearSolver::Serial::build_matrix(
        make_not_null(&matrix), make_not_null(&operand_buffer),
        make_not_null(&result_buffer),
        [&mesh, &inv_jacobian](const gsl::not_null<DataVector*> result,
                               const DataVector& operand) {
          apply_operator(result, operand, mesh, inv_jacobian);
        });

    CAPTURE(inertial_coords);
    CAPTURE(u);
    CAPTURE(matrix * u);
    CAPTURE(ddu);
    CAPTURE(blaze::inv(matrix));
    CHECK(false);
  }
}

}  // namespace elliptic::fd
