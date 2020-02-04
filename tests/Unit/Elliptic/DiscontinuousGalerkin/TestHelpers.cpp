// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/Elliptic/DiscontinuousGalerkin/TestHelpers.hpp"

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/CreateInitialMesh.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/InitialElementIds.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace TestHelpers {
namespace elliptic {
namespace dg {

template <size_t Dim>
DgElementArray<Dim> make_elements(
    const DomainCreator<Dim>& domain_creator) noexcept {
  const auto domain = domain_creator.create_domain();
  const auto domain_extents = domain_creator.initial_extents();
  const auto refinement_levels = domain_creator.initial_refinement_levels();
  DgElementArray<Dim> elements{};
  for (const auto& block : domain.blocks()) {
    const auto element_ids =
        initial_element_ids(block.id(), refinement_levels[block.id()]);
    for (const auto& element_id : element_ids) {
      const auto mesh = domain::Initialization::create_initial_mesh(
          domain_extents, element_id);
      ElementMap<Dim, Frame::Inertial> element_map{
          element_id, block.coordinate_map().get_clone()};
      const auto logical_coords = logical_coordinates(mesh);
      auto inv_jacobian = element_map.inv_jacobian(logical_coords);
      elements.emplace(
          element_id,
          DgElement<Dim>{
              mesh,
              domain::Initialization::create_initial_element(element_id, block),
              std::move(element_map), std::move(inv_jacobian)});
    }
  }
  return elements;
}

template <size_t VolumeDim>
MortarMap<VolumeDim, std::pair<Mesh<VolumeDim - 1>, MortarSizes<VolumeDim - 1>>>
make_mortars(const ElementId<VolumeDim>& element_id,
             const DgElementArray<VolumeDim>& dg_elements) noexcept {
  const auto& dg_element = dg_elements.at(element_id);
  MortarMap<VolumeDim,
            std::pair<Mesh<VolumeDim - 1>, MortarSizes<VolumeDim - 1>>>
      mortars{};
  for (const auto& direction_and_neighbors : dg_element.element.neighbors()) {
    const auto& direction = direction_and_neighbors.first;
    const auto& neighbors = direction_and_neighbors.second;
    const size_t dimension = direction.dimension();
    const auto face_mesh = dg_element.mesh.slice_away(dimension);
    for (const auto& neighbor : neighbors) {
      MortarId<VolumeDim> mortar_id{direction, neighbor};
      mortars.emplace(
          std::move(mortar_id),
          std::make_pair(
              ::dg::mortar_mesh(face_mesh,
                                dg_elements.at(neighbor).mesh.slice_away(
                                    direction.dimension())),
              ::dg::mortar_size(element_id, neighbor, dimension,
                                neighbors.orientation())));
    }
  }
  for (const auto& direction : dg_element.element.external_boundaries()) {
    const size_t dimension = direction.dimension();
    auto face_mesh = dg_element.mesh.slice_away(dimension);
    MortarId<VolumeDim> mortar_id{direction,
                                  ElementId<VolumeDim>::external_boundary_id()};
    mortars.emplace(
        std::move(mortar_id),
        std::make_pair(std::move(face_mesh),
                       make_array<VolumeDim - 1>(Spectral::MortarSize::Full)));
  }
  return mortars;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(r, data)                                                 \
  template DgElementArray<DIM(data)> make_elements(                          \
      const DomainCreator<DIM(data)>& domain_creator) noexcept;              \
  template MortarMap<                                                        \
      DIM(data), std::pair<Mesh<DIM(data) - 1>, MortarSizes<DIM(data) - 1>>> \
  make_mortars(const ElementId<DIM(data)>& element_id,                       \
               const DgElementArray<DIM(data)>& dg_elements) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE

}  // namespace dg
}  // namespace elliptic
}  // namespace TestHelpers
