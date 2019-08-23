// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <blaze/math/Serialization.h>
#include <blaze/util/serialization/Archive.h>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"        // IWYU pragma: keep
#include "DataStructures/DenseMatrix.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/Wedge3D.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/Disk.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/Rectangle.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/RotatedBricks.hpp"
#include "Domain/Creators/RotatedIntervals.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/InitialElementIds.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/DiscontinuousGalerkin/StrongSecondOrderInternalPenalty.hpp"
#include "Elliptic/Systems/Poisson/Equations.hpp"
#include "Evolution/Initialization/Helpers.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/FluxCommunication2.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/InitializeDomain.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/InitializeInterfaces.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/InitializeMortars.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/Mass.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/AddOptionsToDataBox.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/Initialization/Actions/AddComputeTags.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "PointwiseFunctions/MathFunctions/PowX.hpp"
#include "PointwiseFunctions/MathFunctions/TensorProduct.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"
#include "tests/Unit/Elliptic/DiscontinuousGalerkin/OperatorMatrixTestHelpers.hpp"

#include "Parallel/Printf.hpp"

#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"

namespace {

struct ScalarFieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "ScalarFieldTag"; }
};

using FieldsTag = Tags::Variables<tmpl::list<ScalarFieldTag>>;
template <size_t Dim>
using JumpFluxTag =
    db::add_tag_prefix<Tags::NormalDot,
                       db::add_tag_prefix<Tags::Flux, FieldsTag,
                                          tmpl::size_t<Dim>, Frame::Inertial>>;
template <size_t Dim>
using JumpNormalFluxTag =
    db::add_tag_prefix<Tags::NormalFlux, FieldsTag, tmpl::size_t<Dim>,
                       Frame::Inertial>;

template <size_t VolumeDim, typename... CoordMaps>
void test_lifted_flux(
    const Mesh<VolumeDim>& volume_mesh,
    const domain::CoordinateMap<Frame::Logical, Frame::Inertial, CoordMaps...>&
        coordinate_map,
    const Direction<VolumeDim>& face_direction,
    const tnsr::i<DataVector, VolumeDim, Frame::Inertial>& face_normal,
    const DataVector& penalty,
    const tnsr::I<DataVector, VolumeDim, Frame::Inertial>&
        scalar_field_jump_normal_fluxes_on_face,
    const DataVector& scalar_field_jump_fluxes_on_face,
    const DataVector& expected_lifted_flux) noexcept {
  const size_t face_dimension = face_direction.dimension();
  const auto face_mesh = volume_mesh.slice_away(face_dimension);
  const auto volume_logical_coords = logical_coordinates(volume_mesh);
  const auto face_logical_coords =
      interface_logical_coordinates(face_mesh, face_direction);
  const auto inverse_volume_jacobian =
      coordinate_map.inv_jacobian(volume_logical_coords);
  const auto inverse_volume_jacobian_on_face =
      coordinate_map.inv_jacobian(face_logical_coords);

  DataVector perpendicular_inverse_determinant_square =
      square(inverse_volume_jacobian_on_face.get(face_dimension, 0));
  for (size_t d = 1; d < VolumeDim; d++) {
    perpendicular_inverse_determinant_square +=
        square(inverse_volume_jacobian_on_face.get(face_dimension, d));
  }
  const auto logical_to_inertial_surface_jacobian_determinant =
      Scalar<DataVector>(
          get(determinant(coordinate_map.jacobian(face_logical_coords))) *
          sqrt(perpendicular_inverse_determinant_square));

  db::item_type<JumpNormalFluxTag<VolumeDim>> jump_normal_fluxes_on_face{
      face_mesh.number_of_grid_points()};
  get<Tags::NormalFlux<ScalarFieldTag, tmpl::size_t<VolumeDim>,
                       Frame::Inertial>>(jump_normal_fluxes_on_face) =
      scalar_field_jump_normal_fluxes_on_face;
  db::item_type<JumpFluxTag<VolumeDim>> jump_fluxes_on_face{
      face_mesh.number_of_grid_points()};
  get<Tags::NormalDot<
      Tags::Flux<ScalarFieldTag, tmpl::size_t<VolumeDim>, Frame::Inertial>>>(
      jump_fluxes_on_face) =
      Scalar<DataVector>(scalar_field_jump_fluxes_on_face);

  const auto lifted_flux =
      elliptic::dg::Schemes::StrongSecondOrderInternalPenalty_detail::
          lifted_internal_flux<db::item_type<FieldsTag>>(
              jump_normal_fluxes_on_face, jump_fluxes_on_face, volume_mesh,
              face_direction, logical_to_inertial_surface_jacobian_determinant,
              inverse_volume_jacobian, face_normal,
              Scalar<DataVector>(penalty));

  CAPTURE_PRECISE(get(get<ScalarFieldTag>(lifted_flux)));
  CHECK(get(get<ScalarFieldTag>(lifted_flux)).size() ==
        expected_lifted_flux.size());
  CHECK_ITERABLE_APPROX(get(get<ScalarFieldTag>(lifted_flux)),
                        expected_lifted_flux);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.StrongSecondOrderInternalPenalty.LiftedFlux",
                  "[Elliptic][Unit]") {
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
    const auto face_direction = Direction<1>::lower_xi();
    tnsr::i<DataVector, 1, Frame::Inertial> face_normal{{{{-1.}}}};
    tnsr::I<DataVector, 1, Frame::Inertial>
        scalar_field_jump_normal_fluxes_on_face{{{{1.}}}};
    test_lifted_flux(
        mesh, coord_map, face_direction, face_normal, {24.},
        scalar_field_jump_normal_fluxes_on_face, {1.},
        {-20.5, -4.045084971874737, 1.545084971874738, -0.5000000000000007});
  }
  {
    INFO("2D");
    const Mesh<2> mesh{{{4, 3}},
                       Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
    const auto coord_map =
        domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(Affine2D{
            Affine{-1.0, 1.0, -0.3, 0.7}, Affine{-1.0, 1.0, 0.3, 0.55}});
    const auto face_direction = Direction<2>::lower_xi();
    tnsr::i<DataVector, 2, Frame::Inertial> face_normal{
        {{{-1., -1., -1.}, {0., 0., 0.}}}};
    tnsr::I<DataVector, 2, Frame::Inertial>
        scalar_field_jump_normal_fluxes_on_face{
            {{{1., 5., 9.}, {13., 17., 21.}}}};
    test_lifted_flux(
        mesh, coord_map, face_direction, face_normal, {24., 24., 24.},
        scalar_field_jump_normal_fluxes_on_face, {1., 5., 9.},
        {6.312500000000002, -0.16854520716144708, 0.06437854049478063,
         -0.020833333333333322, -14.416666666666666, -3.3709041432289473,
         1.287570809895615, -0.4166666666666672, -17.520833333333332,
         -1.5169068644530266, 0.5794068644530267, -0.18750000000000025});
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
    const auto face_direction = Direction<3>::lower_xi();
    tnsr::i<DataVector, 3, Frame::Inertial> face_normal{
        {{{6, -1.}, {6, 0.}, {6, 0.}}}};
    tnsr::I<DataVector, 3, Frame::Inertial>
        scalar_field_jump_normal_fluxes_on_face{
            {{{1., 5., 9., 13., 17., 21.},
              {25., 29., 33., 37., 41., 45.},
              {49., 53., 57., 61., 65., 69.}}}};
    test_lifted_flux(
        mesh, coord_map, face_direction, face_normal, {6, 24.},
        scalar_field_jump_normal_fluxes_on_face, {1., 5., 9., 13., 17., 21.},
        {3.9392361111111116,    -0.0983180375108441,   0.03755414862195536,
         -0.012152777777777768, 1.487847222222224,     -0.1544997732313266,
         0.059013662120215624,  -0.019097222222222227, -2.3263888888888866,
         -1.741633807334956,    0.6652449184460676,    -0.215277777777778,
         -15.131944444444443,   -1.9663607502168858,   0.7510829724391086,
         -0.24305555555555586,  -6.102430555555555,    -0.7724988661566338,
         0.2950683106010784,    -0.09548611111111123,  -10.053819444444445,
         -0.8286806018771163,   0.31652782409933866,   -0.10243055555555569});
  }
}

namespace {

template <size_t Dim>
using strong_second_order_internal_penalty_scheme =
    elliptic::dg::Schemes::StrongSecondOrderInternalPenalty<
        Dim, FieldsTag, FieldsTag, OperatorMatrixTestHelpers::TemporalIdTag,
        Poisson::ComputeFluxes<Dim, FieldsTag, ScalarFieldTag>,
        Poisson::ComputeNormalFluxes<Dim, FieldsTag, ScalarFieldTag>,
        std::nullptr_t>;
}

// [[TimeOut, 60]]
SPECTRE_TEST_CASE(
    "Unit.Elliptic.StrongSecondOrderInternalPenalty.OperatorMatrix",
    "[Elliptic][Unit]") {
  // These tests build the matrix representation of the DG operator and compare
  // it to a matrix that was computed independently using the code available at
  // https://github.com/nilsleiffischer/dgpy at commit c0a87ce.

  domain::creators::register_derived_with_charm();
  {
    INFO("1D");
    const domain::creators::Interval<Frame::Inertial> domain_creator{
        {{0.}}, {{M_PI}}, {{false}}, {{1}}, {{3}}};
    OperatorMatrixTestHelpers::test_operator_matrix<
        strong_second_order_internal_penalty_scheme<1>, tmpl::list<>,
        tmpl::list<>, ScalarFieldTag, double>(
        std::move(domain_creator), "DgPoissonOperator1DSample.dat", 1.5);
  }
  //   {
  //     INFO("2D");
  //     const domain::creators::Rectangle<Frame::Inertial> domain_creator{
  //         {{0., 0.}}, {{M_PI, M_PI}}, {{false, false}}, {{1, 1}}, {{3, 3}}};
  //     OperatorMatrixTestHelpers::test_operator_matrix<
  //         strong_second_order_internal_penalty_scheme<2>, ScalarFieldTag>(
  //         std::move(domain_creator), "DgPoissonOperator2DSample.dat", 1.5);
  //   }
  //   {
  //     INFO("3D");
  //     const domain::creators::Brick<Frame::Inertial> domain_creator{
  //         {{0., 0., 0.}},
  //         {{M_PI, M_PI, M_PI}},
  //         {{false, false, false}},
  //         {{1, 1, 1}},
  //         {{3, 3, 3}}};
  //     OperatorMatrixTestHelpers::test_operator_matrix<
  //         strong_second_order_internal_penalty_scheme<3>, ScalarFieldTag,
  //         double>( std::move(domain_creator),
  //         "DgPoissonOperator3DSample.dat", 1.5);
  //   }
  //   // These curved-mesh operator matrices are not strictly symmetric with
  //   the
  //   // current scheme. That is probably due to the discretization error of
  //   the
  //   // Jacobian factors which we call 'geometric aliasing' in Vincent2019. To
  //   fix
  //   // this, we can use GL quadrature to evaluate the volume and surface
  //   integrals
  //   // more precisely. We should also evaluate these integrals on the mortars
  //   to
  //   // avoid projection matrices, which a built from mass matrices and
  //   introduce
  //   // another source of geometric aliasing. Then all geometric factors are
  //   // integrated with the same precision and the mismatch between the terms
  //   that
  //   // caused the asymmetry is resolved. Also, using metric identities (see
  //   Bugner
  //   // 2017) or different integration schemes (e.g. Teukolsky 2015) could be
  //   // relevant.
  //   {
  //     INFO("Disk");
  //     const domain::creators::Disk<Frame::Inertial> domain_creator{
  //         2., 5., 0, {{3, 3}}, false};
  //     OperatorMatrixTestHelpers::test_operator_matrix<
  //         strong_second_order_internal_penalty_scheme<2>, ScalarFieldTag>(
  //         std::move(domain_creator), "DgPoissonOperator2DDisk.dat", 1.5);
  //   }
  //   {
  //     INFO("Sphere");
  //     const domain::creators::Sphere<Frame::Inertial> domain_creator{
  //         2., 5., 0, {{3, 3}}, false};
  //     OperatorMatrixTestHelpers::test_operator_matrix<
  //         strong_second_order_internal_penalty_scheme<3>, ScalarFieldTag>(
  //         std::move(domain_creator), "DgPoissonOperator3DSphere.dat", 1.5);
  //   }
}
