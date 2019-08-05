// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Xcts/Actions/LapseAtOrigin.hpp"

#include <array>
#include <boost/none.hpp>
#include <boost/optional.hpp>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/IdPair.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BlockId.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementLogicalCoordinates.hpp"
#include "NumericalAlgorithms/LinearOperators/ApplyMatrices.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace Xcts {
namespace detail {

namespace {
template <size_t Dim>
tnsr::I<DataVector, Dim, Frame::Inertial> make_origin_tensor() noexcept;
template <>
tnsr::I<DataVector, 1, Frame::Inertial> make_origin_tensor() noexcept {
  return tnsr::I<DataVector, 1, Frame::Inertial>{{{DataVector{1, 0.}}}};
}
template <>
tnsr::I<DataVector, 2, Frame::Inertial> make_origin_tensor() noexcept {
  return tnsr::I<DataVector, 2, Frame::Inertial>{
      {{DataVector{1, 0.}, DataVector{1, 0.}}}};
}
template <>
tnsr::I<DataVector, 3, Frame::Inertial> make_origin_tensor() noexcept {
  return tnsr::I<DataVector, 3, Frame::Inertial>{
      {{DataVector{1, 0.}, DataVector{1, 0.}, DataVector{1, 0.}}}};
}
}  // namespace

template <size_t Dim>
boost::optional<tnsr::I<DataVector, Dim, Frame::Logical>>
origin_logical_coordinates(
    const ElementId<Dim>& element_id,
    const Domain<Dim, Frame::Inertial>& domain) noexcept {
  const auto origin = make_origin_tensor<Dim>();
  auto block_logical_coords = block_logical_coordinates(domain, origin);
  const auto element_ids_and_logical_coords = element_logical_coordinates(
      std::vector<ElementId<Dim>>{element_id}, std::move(block_logical_coords));
  auto found_element = element_ids_and_logical_coords.find(element_id);
  if (found_element != element_ids_and_logical_coords.end()) {
    return found_element->second.element_logical_coords;
  } else {
    return boost::none;
  }
}

namespace {
template <size_t Dim>
std::array<Matrix, Dim> interpolate_to_origin_matrices(
    const Mesh<Dim>& mesh, const tnsr::I<DataVector, Dim, Frame::Logical>&
                               origin_logical_coords) noexcept;
template <>
std::array<Matrix, 1> interpolate_to_origin_matrices(
    const Mesh<1>& mesh, const tnsr::I<DataVector, 1, Frame::Logical>&
                             origin_logical_coords) noexcept {
  return {{Spectral::interpolation_matrix(mesh.slice_through(0),
                                          get<0>(origin_logical_coords))}};
}
std::array<Matrix, 2> interpolate_to_origin_matrices(
    const Mesh<2>& mesh, const tnsr::I<DataVector, 2, Frame::Logical>&
                             origin_logical_coords) noexcept {
  return {{Spectral::interpolation_matrix(mesh.slice_through(0),
                                          get<0>(origin_logical_coords)),
           Spectral::interpolation_matrix(mesh.slice_through(1),
                                          get<1>(origin_logical_coords))}};
}
std::array<Matrix, 3> interpolate_to_origin_matrices(
    const Mesh<3>& mesh, const tnsr::I<DataVector, 3, Frame::Logical>&
                             origin_logical_coords) noexcept {
  return {{Spectral::interpolation_matrix(mesh.slice_through(0),
                                          get<0>(origin_logical_coords)),
           Spectral::interpolation_matrix(mesh.slice_through(1),
                                          get<1>(origin_logical_coords)),
           Spectral::interpolation_matrix(mesh.slice_through(2),
                                          get<2>(origin_logical_coords))}};
}
}  // namespace

template <size_t Dim>
std::tuple<size_t, double> lapse_at_origin(
    const DataVector& lapse,
    const boost::optional<tnsr::I<DataVector, Dim, Frame::Logical>>&
        origin_logical_coords,
    const Mesh<Dim>& mesh) noexcept {
  if (origin_logical_coords) {
    return {1, apply_matrices(
                   interpolate_to_origin_matrices(mesh, *origin_logical_coords),
                   lapse, mesh.extents())[0]};
  } else {
    return {0, 0.};
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                 \
  template boost::optional<tnsr::I<DataVector, DIM(data), Frame::Logical>>   \
  origin_logical_coordinates(                                                \
      const ElementId<DIM(data)>& element_id,                                \
      const Domain<DIM(data), Frame::Inertial>& domain) noexcept;            \
  template std::tuple<size_t, double> lapse_at_origin(                       \
      const DataVector& lapse,                                               \
      const boost::optional<tnsr::I<DataVector, DIM(data), Frame::Logical>>& \
          origin_logical_coords,                                             \
      const Mesh<DIM(data)>& mesh) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE

}  // namespace detail
}  // namespace Xcts
