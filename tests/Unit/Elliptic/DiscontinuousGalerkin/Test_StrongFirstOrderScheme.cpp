// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesHelpers.hpp"
#include "Domain/Creators/Interval.hpp"
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

SPECTRE_TEST_CASE("Unit.Elliptic.DG.TestHelpers", "[Unit][Elliptic]") {
  static constexpr size_t volume_dim = 1;

  // Choose a domain
  const domain::creators::Interval domain_creator{
      {{-2.}}, {{2.}}, {{false}}, {{1}}, {{3}}};

  // Choose a system
  using system = Poisson::FirstOrderSystem<volume_dim>;
  const typename system::fluxes fluxes_computer{};

  // Choose a numerical flux
  using NumericalFlux =
      elliptic::dg::NumericalFluxes::FirstOrderInternalPenalty<
          volume_dim, elliptic::Tags::FluxesComputer<typename system::fluxes>,
          typename system::primal_fields, typename system::auxiliary_fields>;
  const NumericalFlux numerical_fluxes_computer{6.75};  // C=1.5

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
  const auto operator_matrix = helpers::build_operator_matrix<system>(
      domain_creator, fluxes_computer, package_boundary_data,
      apply_boundary_contribution);

  Parallel::printf("\n\nDG operator matrix:\n\n");
  Parallel::printf("[");
  for (size_t i = 0; i < operator_matrix.rows(); i++) {
    Parallel::printf("[");
    for (size_t j = 0; j < operator_matrix.columns(); j++) {
      Parallel::printf("%e,", operator_matrix(i, j));
    }
    Parallel::printf("],\n");
  }
  Parallel::printf("]\n");
}
