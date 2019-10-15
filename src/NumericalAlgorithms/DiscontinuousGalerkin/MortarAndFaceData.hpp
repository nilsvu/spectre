// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Variables.hpp"
#include "Domain/OrientationMapHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace dg {

/*!
 * \brief Distinguishes between mortar data, which can be projected to the
 * mortar, and face data, which will not be projected.
 */
template <typename MortarTags, typename FaceTags = tmpl::list<>>
struct MortarAndFaceData {
  using mortar_tags = MortarTags;
  using face_tags = FaceTags;

  /// Data projected to the mortar mesh
  Variables<MortarTags> mortar_data;

  /// Data on the element face that needs no projection to the mortar mesh.
  /// This is a `TaggedTuple` to support non-tensor quantities. It also helps
  /// supporting an empty list of `FaceTags`.
  tuples::tagged_tuple_from_typelist<FaceTags> face_data;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept {
    p | mortar_data;
    p | face_data;
  }

  template <size_t MortarDim>
  MortarAndFaceData<MortarTags, FaceTags> project_to_mortar(
      const Mesh<MortarDim>& face_mesh, const Mesh<MortarDim>& mortar_mesh,
      const std::array<Spectral::MortarSize, MortarDim>& mortar_size) const
      noexcept {
    MortarAndFaceData<MortarTags, FaceTags> projected_data{};
    projected_data.mortar_data = dg::project_to_mortar(
        this->mortar_data, face_mesh, mortar_mesh, mortar_size);
    projected_data.face_data = this->face_data;
    return projected_data;
  }

  template <size_t MortarDim>
  void orient_on_slice(
      const Index<MortarDim>& slice_extents, const size_t sliced_dim,
      const OrientationMap<MortarDim + 1>& orientation_of_neighbor) noexcept {
    this->mortar_data = orient_variables_on_slice(
        this->mortar_data, slice_extents, sliced_dim, orientation_of_neighbor);
  }
};

}  // namespace dg
