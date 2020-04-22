// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/SubdomainHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic {
namespace dg {
namespace SubdomainOperator_detail {

template <size_t Dim>
struct Weighting {
  using argument_tags =
      tmpl::list<domain::Tags::Coordinates<Dim, Frame::Logical>>;

  template <typename SubdomainData>
  static void apply(
      const tnsr::I<DataVector, Dim, Frame::Logical>& central_logical_coords,
      const gsl::not_null<SubdomainData*> subdomain_data) noexcept {
    // TODO: The central element will receive overlap contributions from its
    // face neighbors, so should we weight the subdomain solution with each
    // neighbor's _incoming_ overlap width?
    // TODO: Is this the correct way to handle h-refined mortars?
    // TODO: The overlap width we'll receive may be different to the overlap
    // we're sending because of p-refinement. Should we weight with the expected
    // incoming contribution's width or with the one we're sending?
    // TODO: We'll have to keep in mind that the weighting operation should
    // preserve symmetry of the linear operator
    for (auto& mortar_id_and_overlap : subdomain_data->boundary_data) {
      const auto& mortar_id = mortar_id_and_overlap.first;
      auto& overlap_data = mortar_id_and_overlap.second;
      const auto& direction = mortar_id.first;
      const size_t dimension = direction.dimension();
      const auto& central_logical_coord_in_dim =
          central_logical_coords.get(dimension);
      // Use incoming or outgoing overlap width here?
      // const double overlap_width = overlap_width(
      //     mesh.slice_through(dimension),
      //     overlap_extent(
      //         mesh.extents(dimension),
      //         get<LinearSolver::Tags::Overlap<OptionsGroup>>(box)));
      const double overlap_width = overlap_data.overlap_width();
      // Parallel::printf(
      //     "%s  Weighting center with width %f for overlap with %s\n",
      //     element_index, overlap_width_in_center, mortar_id);
      // Parallel::printf("%s  Logical coords for overlap with %s: %s\n",
      //                  element_index, mortar_id, logical_coord);
      const auto weight_in_central_element =
          LinearSolver::schwarz_detail::weight(central_logical_coord_in_dim,
                                               overlap_width, direction.side());
      // Parallel::printf("%s  Weights:\n%s\n", element_index, w);
      subdomain_data->element_data *= weight_in_central_element;

      const auto neighbor_logical_coords = overlap_data.logical_coordinates();

      const DataVector extended_logical_coord_in_dim =
          neighbor_logical_coords.get(dimension) + direction.sign() * 2.;
      overlap_data.field_data *= LinearSolver::schwarz_detail::weight(
          extended_logical_coord_in_dim, overlap_width, direction.side());
      for (const auto& other_mortar_id_and_overlap :
           subdomain_data->boundary_data) {
        const auto& other_mortar_id = other_mortar_id_and_overlap.first;
        const auto& other_direction = other_mortar_id.first;
        const size_t other_dim = other_direction.dimension();
        if (other_dim == dimension) {
          // Neither other (h-refined) overlaps on this side nor on the opposite
          // side contribute to this overlap's weighting
          continue;
        }
        const auto& other_overlap_data = other_mortar_id_and_overlap.second;
        const double other_overlap_width = other_overlap_data.overlap_width();
        const auto& logical_coord_in_other_dim =
            neighbor_logical_coords.get(other_dim);
        overlap_data.field_data *= LinearSolver::schwarz_detail::weight(
            logical_coord_in_other_dim, other_overlap_width,
            other_direction.side());
      }
    }
  }
};
}  // namespace SubdomainOperator_detail
}  // namespace dg
}  // namespace elliptic
