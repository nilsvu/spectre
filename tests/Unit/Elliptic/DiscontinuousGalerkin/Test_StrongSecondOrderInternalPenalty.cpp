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
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/Rectangle.hpp"
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
#include "ParallelAlgorithms/Initialization/Actions/AddComputeTags.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "PointwiseFunctions/MathFunctions/PowX.hpp"
#include "PointwiseFunctions/MathFunctions/TensorProduct.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"

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
using JumpSecondOrderFluxTag =
    db::add_tag_prefix<Tags::SecondOrderFlux, FieldsTag, tmpl::size_t<Dim>,
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
        scalar_field_jump_second_order_fluxes_on_face,
    const DataVector& scalar_field_jump_fluxes_on_face,
    const DataVector& expected_lifted_flux) noexcept {
  const size_t face_dimension = face_direction.dimension();
  const auto face_mesh = volume_mesh.slice_away(face_dimension);
  const auto volume_logical_coords = logical_coordinates(volume_mesh);
  const auto face_logical_coords =
      interface_logical_coordinates(face_mesh, face_direction);
  const auto volume_jacobian_on_face =
      coordinate_map.jacobian(face_logical_coords);
  const auto inverse_volume_jacobian =
      coordinate_map.inv_jacobian(volume_logical_coords);

  db::item_type<JumpSecondOrderFluxTag<VolumeDim>>
      jump_second_order_fluxes_on_face{face_mesh.number_of_grid_points()};
  get<Tags::SecondOrderFlux<ScalarFieldTag, tmpl::size_t<VolumeDim>,
                            Frame::Inertial>>(
      jump_second_order_fluxes_on_face) =
      scalar_field_jump_second_order_fluxes_on_face;
  db::item_type<JumpFluxTag<VolumeDim>> jump_fluxes_on_face{
      face_mesh.number_of_grid_points()};
  get<Tags::NormalDot<
      Tags::Flux<ScalarFieldTag, tmpl::size_t<VolumeDim>, Frame::Inertial>>>(
      jump_fluxes_on_face) =
      Scalar<DataVector>(scalar_field_jump_fluxes_on_face);

  const auto lifted_flux =
      elliptic::dg::BoundarySchemes::StrongSecondOrderInternalPenalty_detail::
          lifted_internal_flux<db::item_type<FieldsTag>>(
              jump_second_order_fluxes_on_face, jump_fluxes_on_face,
              volume_mesh, face_direction, volume_jacobian_on_face,
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
        scalar_field_jump_second_order_fluxes_on_face{{{{1.}}}};
    test_lifted_flux(
        mesh, coord_map, face_direction, face_normal, {24.},
        scalar_field_jump_second_order_fluxes_on_face, {1.},
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
        scalar_field_jump_second_order_fluxes_on_face{
            {{{1., 5., 9.}, {13., 17., 21.}}}};
    test_lifted_flux(
        mesh, coord_map, face_direction, face_normal, {24., 24., 24.},
        scalar_field_jump_second_order_fluxes_on_face, {1., 5., 9.},
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
        scalar_field_jump_second_order_fluxes_on_face{
            {{{1., 5., 9., 13., 17., 21.},
              {25., 29., 33., 37., 41., 45.},
              {49., 53., 57., 61., 65., 69.}}}};
    test_lifted_flux(
        mesh, coord_map, face_direction, face_normal, {6, 24.},
        scalar_field_jump_second_order_fluxes_on_face,
        {1., 5., 9., 13., 17., 21.},
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

template <typename Tag>
struct DgOperatorAppliedTo : db::PrefixTag, db::SimpleTag {
  using type = db::item_type<Tag>;
  using tag = Tag;
  static std::string name() noexcept {
    return "DgOperatorAppliedTo(" + db::tag_name<Tag>() + ")";
  }
};

struct TemporalIdTag : db::SimpleTag {
  using type = int;
  static std::string name() noexcept { return "TemporalIdTag"; }
  template <typename Tag>
  using step_prefix = DgOperatorAppliedTo<Tag>;
};

// For interfaces initializer
template <size_t Dim>
struct System {
  static constexpr size_t volume_dim = Dim;
  using variables_tag = FieldsTag;
  template <typename Tag>
  using magnitude_tag = Tags::EuclideanMagnitude<Tag>;
};

template <size_t Dim, typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementIndex<Dim>;
  using const_global_cache_tag_list = tmpl::list<>;
  using add_options_to_databox = Parallel::AddNoOptionsToDataBox;

  using dg_scheme = typename Metavariables::dg_scheme;
  using temporal_id_tag = typename dg_scheme::temporal_id_tag;
  using variables_tag = typename dg_scheme::variables_tag;
  using operator_applied_to_variables_tag =
      db::add_tag_prefix<temporal_id_tag::template step_prefix, variables_tag>;
  using second_order_fluxes_tag =
      db::add_tag_prefix<::Tags::SecondOrderFlux, variables_tag,
                         tmpl::size_t<Dim>, Frame::Inertial>;
  using fluxes_tag = db::add_tag_prefix<::Tags::Flux, variables_tag,
                                        tmpl::size_t<Dim>, Frame::Inertial>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<
              ActionTesting::InitializeDataBox<
                  tmpl::list<::Tags::Domain<Dim, Frame::Inertial>,
                             ::Tags::InitialExtents<Dim>, temporal_id_tag,
                             ::Tags::Next<temporal_id_tag>, variables_tag,
                             operator_applied_to_variables_tag>>,
              dg::Actions::InitializeDomain<Dim>,
              Initialization::Actions::AddComputeTags<tmpl::list<
                  ::Tags::JacobianCompute<
                      ::Tags::ElementMap<Dim>,
                      ::Tags::Coordinates<Dim, Frame::Logical>>,
                  ::Tags::JacobianInverseCompute<
                      ::Tags::ElementMap<Dim>,
                      ::Tags::Coordinates<Dim, Frame::Logical>>,
                  ::Tags::DerivCompute<
                      variables_tag,
                      ::Tags::Jacobian<Dim, Frame::Inertial, Frame::Logical>>,
                  Poisson::ComputeFluxes<Dim, variables_tag, ScalarFieldTag>,
                  ::Tags::DivCompute<
                      fluxes_tag,
                      ::Tags::Jacobian<Dim, Frame::Inertial, Frame::Logical>>>>,
              dg::Actions::InitializeInterfaces<
                  typename Metavariables::system,
                  dg::Initialization::slice_tags_to_face<
                      ::Tags::Jacobian<Dim, Frame::Logical, Frame::Inertial>,
                      fluxes_tag, variables_tag>,
                  dg::Initialization::slice_tags_to_exterior<>,
                  dg::Initialization::face_compute_tags<
                      ::Tags::NormalDotCompute<fluxes_tag, Dim,
                                               Frame::Inertial>,
                      Poisson::ComputeSecondOrderFluxes<Dim, variables_tag,
                                                        ScalarFieldTag>,
                      typename dg_scheme::compute_packaged_remote_data,
                      typename dg_scheme::compute_packaged_local_data>>,
              dg::Actions::InitializeMortars<
                  dg_scheme, false, Initialization::MergePolicy::Overwrite>>>,

      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<dg::Actions::SendDataForFluxes<dg_scheme>,
                     dg::Actions::ReceiveDataForFluxes<dg_scheme>>>>;
};

template <size_t Dim>
struct Metavariables {
  using system = System<Dim>;
  using temporal_id_tag = TemporalIdTag;
  using dg_scheme =
      elliptic::dg::BoundarySchemes::StrongSecondOrderInternalPenalty<
          Dim, typename system::variables_tag, temporal_id_tag>;
  using component_list = tmpl::list<ElementArray<Dim, Metavariables>>;
  using const_global_cache_tag_list = tmpl::list<>;
  enum class Phase { Initialization, Testing, Exit };
};

template <typename VariablesTag>
struct SetData {
  template <typename ParallelComponent, typename DataBox,
            typename Metavariables, size_t Dim,
            Requires<db::tag_is_retrievable_v<VariablesTag, DataBox>> = nullptr>
  static void apply(DataBox& box,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ElementIndex<Dim>& /*array_index*/,
                    const db::item_type<VariablesTag>& variables) noexcept {
    db::mutate<VariablesTag>(
        make_not_null(&box), [&variables](const gsl::not_null<
                                          db::item_type<VariablesTag>*>
                                              local_variables) noexcept {
          *local_variables = variables;
        });
  }
};

struct ApplyDgOperator {
  template <
      typename ParallelComponent, typename DataBox, typename Metavariables,
      size_t Dim, typename DgOperator,
      typename MortarDataTag =
          ::Tags::Mortars<typename DgOperator::mortar_data_tag, Dim>,
      Requires<db::tag_is_retrievable_v<MortarDataTag, DataBox>> = nullptr>
  static void apply(DataBox& box,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ElementIndex<Dim>& /*array_index*/,
                    const DgOperator& dg_operator) noexcept {
    using temporal_id_tag = typename DgOperator::temporal_id_tag;
    db::mutate_apply(dg_operator, make_not_null(&box));
    db::mutate<temporal_id_tag, Tags::Next<temporal_id_tag>>(
        make_not_null(&box), [](const gsl::not_null<int*> temporal_id,
                                const gsl::not_null<int*> next_temporal_id) {
          *temporal_id = *next_temporal_id;
          (*next_temporal_id)++;
        });
  }
};

template <size_t Dim>
void test_operator_matrix(
    const double penalty_parameter,
    const DomainCreator<Dim, Frame::Inertial>& domain_creator,
    const std::string& expected_matrix_filename) {
  using metavariables = Metavariables<Dim>;
  using element_array = ElementArray<Dim, metavariables>;

  using DgScheme = typename metavariables::dg_scheme;
  using variables_tag = typename DgScheme::variables_tag;
  using temporal_id_tag = typename DgScheme::temporal_id_tag;
  using operator_applied_to_variables_tag =
      db::add_tag_prefix<temporal_id_tag::template step_prefix, variables_tag>;

  const auto domain = domain_creator.create_domain();
  const auto initial_extents = domain_creator.initial_extents();
  const auto& block = domain.blocks().front();
  const auto initial_ref_levs =
      domain_creator.initial_refinement_levels()[block.id()];
  const auto element_ids = initial_element_ids(block.id(), initial_ref_levs);

  ActionTesting::MockRuntimeSystem<metavariables> runner{{}};

  size_t total_num_points = 0;

  // Set up all elements in the domain
  for (const auto& element_id : element_ids) {
    // The variables are set later for testing different values
    const size_t num_points =
        ::Initialization::element_mesh(initial_extents, element_id)
            .number_of_grid_points();
    db::item_type<variables_tag> vars{num_points};
    db::item_type<operator_applied_to_variables_tag> operator_applied_to_vars{
        num_points};
    total_num_points += num_points;

    ActionTesting::emplace_component_and_initialize<element_array>(
        &runner, element_id,
        {domain_creator.create_domain(), domain_creator.initial_extents(), 0, 1,
         std::move(vars), std::move(operator_applied_to_vars)});
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
  }

  runner.set_phase(metavariables::Phase::Testing);

  DgScheme dg_operator{penalty_parameter};
  DenseMatrix<double, blaze::rowMajor> dg_operator_matrix{total_num_points,
                                                          total_num_points};

  // Build the matrix by applying the operator to unit vectors
  size_t i_across_elements = 0;
  size_t j_across_elements = 0;
  for (const auto& active_element : element_ids) {
    const size_t num_points =
        ActionTesting::get_databox_tag<element_array, ::Tags::Mesh<Dim>>(
            runner, active_element)
            .number_of_grid_points();
    for (size_t i = 0; i < num_points; i++) {
      for (const auto& element_id : element_ids) {
        const auto& mesh =
            ActionTesting::get_databox_tag<element_array, ::Tags::Mesh<Dim>>(
                runner, element_id);

        // Construct a unit vector
        db::item_type<variables_tag> vars{mesh.number_of_grid_points(), 0.};
        if (element_id == active_element) {
          get(get<ScalarFieldTag>(vars))[i] = 1.;
        }
        ActionTesting::simple_action<element_array, SetData<variables_tag>>(
            make_not_null(&runner), element_id, std::move(vars));

        // Send data
        ActionTesting::next_action<element_array>(make_not_null(&runner),
                                                  element_id);
      }
      // Split the loop to have all elements send their data before receiving
      for (const auto& element_id : element_ids) {
        // Receive data
        ActionTesting::next_action<element_array>(make_not_null(&runner),
                                                  element_id);

        // Apply the operator
        ActionTesting::simple_action<element_array, ApplyDgOperator>(
            make_not_null(&runner), element_id, dg_operator);
        const auto& operator_applied_to_vars =
            ActionTesting::get_databox_tag<element_array,
                                           operator_applied_to_variables_tag>(
                runner, element_id);

        // Store result in matrix
        for (size_t j = 0; j < operator_applied_to_vars.number_of_grid_points();
             j++) {
          dg_operator_matrix(i_across_elements, j_across_elements) =
              get(get<DgOperatorAppliedTo<ScalarFieldTag>>(
                  operator_applied_to_vars))[j];
          j_across_elements++;
        }
      }
      i_across_elements++;
      j_across_elements = 0;
    }
  }

  // Load expected matrix from file
  std::ifstream expected_matrix_file{unit_test_path() +
                                     "/Elliptic/DiscontinuousGalerkin/" +
                                     expected_matrix_filename};
  std::istream_iterator<double> expected_matrix_file_element{
      expected_matrix_file};
  DenseMatrix<double, blaze::rowMajor> expected_matrix{total_num_points,
                                                       total_num_points};
  for (size_t row = 0; row < total_num_points; row++) {
    for (DenseMatrix<double, blaze::rowMajor>::Iterator
             expected_matrix_element = expected_matrix.begin(row);
         expected_matrix_element != expected_matrix.end(row);
         expected_matrix_element++) {
      *expected_matrix_element = *expected_matrix_file_element;
      expected_matrix_file_element++;
    }
  }
  // Make sure we have reached the end of the file
  CHECK(expected_matrix_file_element == std::istream_iterator<double>{});

  CHECK_MATRIX_APPROX(dg_operator_matrix, expected_matrix);
}
}  // namespace

// [[TimeOut, 20]]
SPECTRE_TEST_CASE(
    "Unit.Elliptic.StrongSecondOrderInternalPenalty.OperatorMatrix",
    "[Elliptic][Unit]") {
  // These tests build the matrix representation of the DG operator and compare
  // it to a matrix that was computed independently using the code available at
  // https://github.com/nilsleiffischer/dgpy at commit c0a87ce.

  {
    INFO("1D");
    const domain::creators::Interval<Frame::Inertial> domain_creator{
        {{0.}}, {{M_PI}}, {{false}}, {{1}}, {{3}}};
    // Register the coordinate map for serialization
    PUPable_reg(
        SINGLE_ARG(domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                                         domain::CoordinateMaps::Affine>));

    test_operator_matrix(1.5, std::move(domain_creator),
                         "DgPoissonOperator1DSample.dat");
  }
  {
    INFO("2D");
    const domain::creators::Rectangle<Frame::Inertial> domain_creator{
        {{0., 0.}}, {{M_PI, M_PI}}, {{false, false}}, {{1, 1}}, {{3, 3}}};
    // Register the coordinate map for serialization
    PUPable_reg(
        SINGLE_ARG(domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                                         domain::CoordinateMaps::ProductOf2Maps<
                                             domain::CoordinateMaps::Affine,
                                             domain::CoordinateMaps::Affine>>));

    test_operator_matrix(1.5, std::move(domain_creator),
                         "DgPoissonOperator2DSample.dat");
  }
  {
    INFO("3D");
    const domain::creators::Brick<Frame::Inertial> domain_creator{
        {{0., 0., 0.}},
        {{M_PI, M_PI, M_PI}},
        {{false, false, false}},
        {{1, 1, 1}},
        {{3, 3, 3}}};
    // Register the coordinate map for serialization
    PUPable_reg(SINGLE_ARG(
        domain::CoordinateMap<
            Frame::Logical, Frame::Inertial,
            domain::CoordinateMaps::ProductOf3Maps<
                domain::CoordinateMaps::Affine, domain::CoordinateMaps::Affine,
                domain::CoordinateMaps::Affine>>));

    test_operator_matrix(1.5, std::move(domain_creator),
                         "DgPoissonOperator3DSample.dat");
  }
}
