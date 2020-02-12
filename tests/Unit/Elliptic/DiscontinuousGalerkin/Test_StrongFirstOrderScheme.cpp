// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesHelpers.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/Rectangle.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Elliptic/DiscontinuousGalerkin/NumericalFluxes/FirstOrderInternalPenalty.hpp"
#include "Elliptic/Systems/Poisson/FirstOrderSystem.hpp"
#include "Elliptic/Tags.hpp"  // Needed by the numerical flux (for now)
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/StrongFirstOrder/Equations.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleBoundaryData.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/Elliptic/DiscontinuousGalerkin/TestHelpers.hpp"

#include "Parallel/Printf.hpp"

namespace helpers = TestHelpers::elliptic::dg;

namespace {

/*
 * Makes the following assumptions:
 * - First-order Poisson system
 * - Strong first-order internal penalty DG scheme
 * - Diagonal mass matrix approximation ("mass-lumping")
 * - The `penalty_parameter` is used directly as the prefactor to the penalty
 * term in the numerical flux.
 * - The elements are ordered by block first, and then by segment index of each
 * dimension in turn.
 * - Within each element this is the data layout:
 *   - The primal field precedes the auxiliary field
 *   - Tensor components are ordered first by index, then by dimension (as one
 *   would expect)
 *   - Grid points for each component are stored in column-major format
 *   (corresponding to the order called 'F' in Numpy).
 */
template <size_t Dim>
DenseMatrix<double> build_poisson_strong_first_order_operator(
    const DomainCreator<Dim>& domain_creator, const double penalty_parameter) {
  static constexpr size_t volume_dim = Dim;

  // Choose a system
  using system = Poisson::FirstOrderSystem<volume_dim>;
  const typename system::fluxes fluxes_computer{};

  // Choose a numerical flux
  using NumericalFlux =
      elliptic::dg::NumericalFluxes::FirstOrderInternalPenalty<
          volume_dim, elliptic::Tags::FluxesComputer<typename system::fluxes>,
          typename system::primal_fields, typename system::auxiliary_fields>;
  const NumericalFlux numerical_fluxes_computer{penalty_parameter};

  // Shortcuts for tags
  using field_tag = Poisson::Tags::Field;
  using field_gradient_tag =
      ::Tags::deriv<field_tag, tmpl::size_t<volume_dim>, Frame::Inertial>;
  using all_fields_tags =
      db::get_variables_tags_list<typename system::fields_tag>;
  using fluxes_tags =
      db::wrap_tags_in<::Tags::Flux, all_fields_tags, tmpl::size_t<volume_dim>,
                       Frame::Inertial>;
  using div_fluxes_tags = db::wrap_tags_in<::Tags::div, fluxes_tags>;
  using n_dot_fluxes_tags =
      db::wrap_tags_in<::Tags::NormalDotFlux, all_fields_tags>;

  // Define the boundary scheme
  // We use the StrongFirstOrder scheme, so we'll need the n.F on the boundaries
  // and the data needed by the numerical flux.
  using BoundaryData = dg::SimpleBoundaryData<
      tmpl::remove_duplicates<tmpl::append<
          n_dot_fluxes_tags, typename NumericalFlux::package_field_tags>>,
      typename NumericalFlux::package_extra_tags>;
  const auto package_boundary_data =
      [&numerical_fluxes_computer, &fluxes_computer](
          const tnsr::i<DataVector, volume_dim>& face_normal,
          const Variables<n_dot_fluxes_tags>& n_dot_fluxes,
          const Variables<div_fluxes_tags>& div_fluxes) -> BoundaryData {
    BoundaryData boundary_data{n_dot_fluxes.number_of_grid_points()};
    boundary_data.field_data.assign_subset(n_dot_fluxes);
    dg::NumericalFluxes::package_data(
        make_not_null(&boundary_data), numerical_fluxes_computer,
        get<::Tags::NormalDotFlux<field_gradient_tag>>(n_dot_fluxes),
        get<::Tags::div<::Tags::Flux<
            field_gradient_tag, tmpl::size_t<volume_dim>, Frame::Inertial>>>(
            div_fluxes),
        fluxes_computer, face_normal);
    return boundary_data;
  };
  const auto apply_boundary_contribution =
      [&numerical_fluxes_computer](
          const gsl::not_null<Variables<all_fields_tags>*> result,
          const BoundaryData& local_boundary_data,
          const BoundaryData& remote_boundary_data,
          const Scalar<DataVector>& magnitude_of_face_normal,
          const Mesh<volume_dim>& mesh,
          const helpers::MortarId<volume_dim>& mortar_id,
          const Mesh<volume_dim - 1>& mortar_mesh,
          const helpers::MortarSizes<volume_dim - 1>& mortar_size) {
        const size_t dimension = mortar_id.first.dimension();
        auto boundary_contribution =
            dg::BoundarySchemes::strong_first_order_boundary_flux<
                all_fields_tags>(
                local_boundary_data, remote_boundary_data,
                numerical_fluxes_computer, magnitude_of_face_normal,
                mesh.extents(dimension), mesh.slice_away(dimension),
                mortar_mesh, mortar_size);
        add_slice_to_data(result, std::move(boundary_contribution),
                          mesh.extents(), dimension,
                          index_to_slice_at(mesh.extents(), mortar_id.first));
      };

  // Build the operator matrix
  return helpers::build_operator_matrix<system>(domain_creator, fluxes_computer,
                                                package_boundary_data,
                                                apply_boundary_contribution);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.DG.TestHelpers", "[Unit][Elliptic]") {
  {
    INFO("1D");
    const domain::creators::Interval domain_creator{
        {{-2.}}, {{2.}}, {{false}}, {{1}}, {{3}}};
    const double penalty_parameter = 6.75;
    CAPTURE(penalty_parameter);
    const DenseMatrix<double> expected_operator_matrix{
        {36.0, 6.0, -1.5, -1.5, -2.0, 0.5, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0},
        {-0.0, -0.0, -0.0, 0.5, -0.0, -0.5, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0},
        {-0.75, 3.0, 18.0, -0.5, 2.0, 1.5, -18.0, -3.0, 0.75, -0.0, -0.0, -0.0},
        {-1.5, -2.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.5, 0.0, -0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {-0.5, 2.0, 0.0, 0.0, 0.0, 1.0, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.75, -3.0, -18.0, -0.0, -0.0, -0.0, 18.0, 3.0, -0.75, -1.5, -2.0,
         0.5},
        {-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.5, -0.0, -0.5},
        {-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -1.5, 6.0, 36.0, -0.5, 2.0, 1.5},
        {0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, -2.0, 0.5, 1.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, -0.5, 0.0, 1.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 2.0, 1.5, 0.0, 0.0, 1.0}};
    const auto operator_matrix = build_poisson_strong_first_order_operator(
        domain_creator, penalty_parameter);
    CHECK_MATRIX_APPROX(operator_matrix, expected_operator_matrix);
  }
  {
    INFO("2D");
    const domain::creators::Rectangle domain_creator{
        {{-2., -2.}}, {{2., 2.}}, {{false, false}}, {{1, 1}}, {{3, 3}}};
    const double penalty_parameter = 6.75;
    CAPTURE(penalty_parameter);
    const auto operator_matrix = build_poisson_strong_first_order_operator(
        domain_creator, penalty_parameter);
    std::vector<double> expected_sum_over_each_row{
        75.0, 37.5, 40.5, 37.5, 0.0,  3.0,  40.5, 3.0,  6.0,  -2.0, 1.0,  1.0,
        -2.0, 1.0,  1.0,  -2.0, 1.0,  1.0,  -2.0, -2.0, -2.0, 1.0,  1.0,  1.0,
        1.0,  1.0,  1.0,  34.5, -3.0, 0.0,  37.5, 0.0,  3.0,  81.0, 43.5, 46.5,
        -2.0, 1.0,  1.0,  -2.0, 1.0,  1.0,  -2.0, 1.0,  1.0,  1.0,  1.0,  1.0,
        1.0,  1.0,  1.0,  4.0,  4.0,  4.0,  34.5, 37.5, 81.0, -3.0, 0.0,  43.5,
        0.0,  3.0,  46.5, 1.0,  1.0,  4.0,  1.0,  1.0,  4.0,  1.0,  1.0,  4.0,
        -2.0, -2.0, -2.0, 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  -6.0, -3.0, 40.5,
        -3.0, 0.0,  43.5, 40.5, 43.5, 87.0, 1.0,  1.0,  4.0,  1.0,  1.0,  4.0,
        1.0,  1.0,  4.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  4.0,  4.0,  4.0};
    for (size_t row = 0; row < operator_matrix.rows(); row++) {
      double sum_over_row = 0;
      for (size_t col = 0; col < operator_matrix.columns(); col++) {
        sum_over_row += operator_matrix(row, col);
      }
      CAPTURE(row);
      CHECK(sum_over_row == expected_sum_over_each_row.at(row));
    }
  }
  {
    INFO("3D");
    const domain::creators::Brick domain_creator{{{-2., -2., -2.}},
                                                     {{2., 2., 2.}},
                                                     {{false, false, false}},
                                                     {{1, 1, 1}},
                                                     {{3, 3, 3}}};
    const double penalty_parameter = 6.75;
    CAPTURE(penalty_parameter);
    const auto operator_matrix = build_poisson_strong_first_order_operator(
        domain_creator, penalty_parameter);
    std::vector<double> expected_sum_over_each_row{
        112.5, 75.0,  78.0,  75.0, 37.5, 40.5,  78.0, 40.5,  43.5, 75.0, 37.5,
        40.5,  37.5,  0.0,   3.0,  40.5, 3.0,   6.0,  78.0,  40.5, 43.5, 40.5,
        3.0,   6.0,   43.5,  6.0,  9.0,  -2.0,  1.0,  1.0,   -2.0, 1.0,  1.0,
        -2.0,  1.0,   1.0,   -2.0, 1.0,  1.0,   -2.0, 1.0,   1.0,  -2.0, 1.0,
        1.0,   -2.0,  1.0,   1.0,  -2.0, 1.0,   1.0,  -2.0,  1.0,  1.0,  -2.0,
        -2.0,  -2.0,  1.0,   1.0,  1.0,  1.0,   1.0,  1.0,   -2.0, -2.0, -2.0,
        1.0,   1.0,   1.0,   1.0,  1.0,  1.0,   -2.0, -2.0,  -2.0, 1.0,  1.0,
        1.0,   1.0,   1.0,   1.0,  -2.0, -2.0,  -2.0, -2.0,  -2.0, -2.0, -2.0,
        -2.0,  -2.0,  1.0,   1.0,  1.0,  1.0,   1.0,  1.0,   1.0,  1.0,  1.0,
        1.0,   1.0,   1.0,   1.0,  1.0,  1.0,   1.0,  1.0,   1.0,  72.0, 34.5,
        37.5,  34.5,  -3.0,  0.0,  37.5, 0.0,   3.0,  75.0,  37.5, 40.5, 37.5,
        0.0,   3.0,   40.5,  3.0,  6.0,  118.5, 81.0, 84.0,  81.0, 43.5, 46.5,
        84.0,  46.5,  49.5,  -2.0, 1.0,  1.0,   -2.0, 1.0,   1.0,  -2.0, 1.0,
        1.0,   -2.0,  1.0,   1.0,  -2.0, 1.0,   1.0,  -2.0,  1.0,  1.0,  -2.0,
        1.0,   1.0,   -2.0,  1.0,  1.0,  -2.0,  1.0,  1.0,   -2.0, -2.0, -2.0,
        1.0,   1.0,   1.0,   1.0,  1.0,  1.0,   -2.0, -2.0,  -2.0, 1.0,  1.0,
        1.0,   1.0,   1.0,   1.0,  -2.0, -2.0,  -2.0, 1.0,   1.0,  1.0,  1.0,
        1.0,   1.0,   1.0,   1.0,  1.0,  1.0,   1.0,  1.0,   1.0,  1.0,  1.0,
        1.0,   1.0,   1.0,   1.0,  1.0,  1.0,   1.0,  1.0,   1.0,  4.0,  4.0,
        4.0,   4.0,   4.0,   4.0,  4.0,  4.0,   4.0,  72.0,  34.5, 37.5, 75.0,
        37.5,  40.5,  118.5, 81.0, 84.0, 34.5,  -3.0, 0.0,   37.5, 0.0,  3.0,
        81.0,  43.5,  46.5,  37.5, 0.0,  3.0,   40.5, 3.0,   6.0,  84.0, 46.5,
        49.5,  -2.0,  1.0,   1.0,  -2.0, 1.0,   1.0,  -2.0,  1.0,  1.0,  -2.0,
        1.0,   1.0,   -2.0,  1.0,  1.0,  -2.0,  1.0,  1.0,   -2.0, 1.0,  1.0,
        -2.0,  1.0,   1.0,   -2.0, 1.0,  1.0,   1.0,  1.0,   1.0,  1.0,  1.0,
        1.0,   4.0,   4.0,   4.0,  1.0,  1.0,   1.0,  1.0,   1.0,  1.0,  4.0,
        4.0,   4.0,   1.0,   1.0,  1.0,  1.0,   1.0,  1.0,   4.0,  4.0,  4.0,
        -2.0,  -2.0,  -2.0,  -2.0, -2.0, -2.0,  -2.0, -2.0,  -2.0, 1.0,  1.0,
        1.0,   1.0,   1.0,   1.0,  1.0,  1.0,   1.0,  1.0,   1.0,  1.0,  1.0,
        1.0,   1.0,   1.0,   1.0,  1.0,  31.5,  -6.0, -3.0,  34.5, -3.0, 0.0,
        78.0,  40.5,  43.5,  34.5, -3.0, 0.0,   37.5, 0.0,   3.0,  81.0, 43.5,
        46.5,  78.0,  40.5,  43.5, 81.0, 43.5,  46.5, 124.5, 87.0, 90.0, -2.0,
        1.0,   1.0,   -2.0,  1.0,  1.0,  -2.0,  1.0,  1.0,   -2.0, 1.0,  1.0,
        -2.0,  1.0,   1.0,   -2.0, 1.0,  1.0,   -2.0, 1.0,   1.0,  -2.0, 1.0,
        1.0,   -2.0,  1.0,   1.0,  1.0,  1.0,   1.0,  1.0,   1.0,  1.0,  4.0,
        4.0,   4.0,   1.0,   1.0,  1.0,  1.0,   1.0,  1.0,   4.0,  4.0,  4.0,
        1.0,   1.0,   1.0,   1.0,  1.0,  1.0,   4.0,  4.0,   4.0,  1.0,  1.0,
        1.0,   1.0,   1.0,   1.0,  1.0,  1.0,   1.0,  1.0,   1.0,  1.0,  1.0,
        1.0,   1.0,   1.0,   1.0,  1.0,  4.0,   4.0,  4.0,   4.0,  4.0,  4.0,
        4.0,   4.0,   4.0,   72.0, 75.0, 118.5, 34.5, 37.5,  81.0, 37.5, 40.5,
        84.0,  34.5,  37.5,  81.0, -3.0, 0.0,   43.5, 0.0,   3.0,  46.5, 37.5,
        40.5,  84.0,  0.0,   3.0,  46.5, 3.0,   6.0,  49.5,  1.0,  1.0,  4.0,
        1.0,   1.0,   4.0,   1.0,  1.0,  4.0,   1.0,  1.0,   4.0,  1.0,  1.0,
        4.0,   1.0,   1.0,   4.0,  1.0,  1.0,   4.0,  1.0,   1.0,  4.0,  1.0,
        1.0,   4.0,   -2.0,  -2.0, -2.0, 1.0,   1.0,  1.0,   1.0,  1.0,  1.0,
        -2.0,  -2.0,  -2.0,  1.0,  1.0,  1.0,   1.0,  1.0,   1.0,  -2.0, -2.0,
        -2.0,  1.0,   1.0,   1.0,  1.0,  1.0,   1.0,  -2.0,  -2.0, -2.0, -2.0,
        -2.0,  -2.0,  -2.0,  -2.0, -2.0, 1.0,   1.0,  1.0,   1.0,  1.0,  1.0,
        1.0,   1.0,   1.0,   1.0,  1.0,  1.0,   1.0,  1.0,   1.0,  1.0,  1.0,
        1.0,   31.5,  34.5,  78.0, -6.0, -3.0,  40.5, -3.0,  0.0,  43.5, 34.5,
        37.5,  81.0,  -3.0,  0.0,  43.5, 0.0,   3.0,  46.5,  78.0, 81.0, 124.5,
        40.5,  43.5,  87.0,  43.5, 46.5, 90.0,  1.0,  1.0,   4.0,  1.0,  1.0,
        4.0,   1.0,   1.0,   4.0,  1.0,  1.0,   4.0,  1.0,   1.0,  4.0,  1.0,
        1.0,   4.0,   1.0,   1.0,  4.0,  1.0,   1.0,  4.0,   1.0,  1.0,  4.0,
        -2.0,  -2.0,  -2.0,  1.0,  1.0,  1.0,   1.0,  1.0,   1.0,  -2.0, -2.0,
        -2.0,  1.0,   1.0,   1.0,  1.0,  1.0,   1.0,  -2.0,  -2.0, -2.0, 1.0,
        1.0,   1.0,   1.0,   1.0,  1.0,  1.0,   1.0,  1.0,   1.0,  1.0,  1.0,
        1.0,   1.0,   1.0,   1.0,  1.0,  1.0,   1.0,  1.0,   1.0,  1.0,  1.0,
        1.0,   4.0,   4.0,   4.0,  4.0,  4.0,   4.0,  4.0,   4.0,  4.0,  31.5,
        34.5,  78.0,  34.5,  37.5, 81.0, 78.0,  81.0, 124.5, -6.0, -3.0, 40.5,
        -3.0,  0.0,   43.5,  40.5, 43.5, 87.0,  -3.0, 0.0,   43.5, 0.0,  3.0,
        46.5,  43.5,  46.5,  90.0, 1.0,  1.0,   4.0,  1.0,   1.0,  4.0,  1.0,
        1.0,   4.0,   1.0,   1.0,  4.0,  1.0,   1.0,  4.0,   1.0,  1.0,  4.0,
        1.0,   1.0,   4.0,   1.0,  1.0,  4.0,   1.0,  1.0,   4.0,  1.0,  1.0,
        1.0,   1.0,   1.0,   1.0,  4.0,  4.0,   4.0,  1.0,   1.0,  1.0,  1.0,
        1.0,   1.0,   4.0,   4.0,  4.0,  1.0,   1.0,  1.0,   1.0,  1.0,  1.0,
        4.0,   4.0,   4.0,   -2.0, -2.0, -2.0,  -2.0, -2.0,  -2.0, -2.0, -2.0,
        -2.0,  1.0,   1.0,   1.0,  1.0,  1.0,   1.0,  1.0,   1.0,  1.0,  1.0,
        1.0,   1.0,   1.0,   1.0,  1.0,  1.0,   1.0,  1.0,   -9.0, -6.0, 37.5,
        -6.0,  -3.0,  40.5,  37.5, 40.5, 84.0,  -6.0, -3.0,  40.5, -3.0, 0.0,
        43.5,  40.5,  43.5,  87.0, 37.5, 40.5,  84.0, 40.5,  43.5, 87.0, 84.0,
        87.0,  130.5, 1.0,   1.0,  4.0,  1.0,   1.0,  4.0,   1.0,  1.0,  4.0,
        1.0,   1.0,   4.0,   1.0,  1.0,  4.0,   1.0,  1.0,   4.0,  1.0,  1.0,
        4.0,   1.0,   1.0,   4.0,  1.0,  1.0,   4.0,  1.0,   1.0,  1.0,  1.0,
        1.0,   1.0,   4.0,   4.0,  4.0,  1.0,   1.0,  1.0,   1.0,  1.0,  1.0,
        4.0,   4.0,   4.0,   1.0,  1.0,  1.0,   1.0,  1.0,   1.0,  4.0,  4.0,
        4.0,   1.0,   1.0,   1.0,  1.0,  1.0,   1.0,  1.0,   1.0,  1.0,  1.0,
        1.0,   1.0,   1.0,   1.0,  1.0,  1.0,   1.0,  1.0,   4.0,  4.0,  4.0,
        4.0,   4.0,   4.0,   4.0,  4.0,  4.0};
    for (size_t row = 0; row < operator_matrix.rows(); row++) {
      double sum_over_row = 0;
      for (size_t col = 0; col < operator_matrix.columns(); col++) {
        sum_over_row += operator_matrix(row, col);
      }
      CAPTURE(row);
      CHECK(sum_over_row == expected_sum_over_each_row.at(row));
    }
  }
}
