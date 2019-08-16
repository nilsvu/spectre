// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Declares function unnormalized_face_normal

#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <string>
#include <unordered_map>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain {
template <typename, typename, size_t>
class CoordinateMapBase;
}  // namespace domain
class DataVector;
template <size_t>
class Direction;
template <size_t Dim, typename Frame>
class ElementMap;
template <size_t>
class Mesh;
/// \endcond

// @{
/*!
 * \ingroup ComputationalDomainGroup
 * \brief Compute the outward grid normal on a face of an Element
 *
 * \returns outward grid-frame one-form holding the normal
 *
 * \details
 * Computes the grid-frame normal by taking the logical-frame unit
 * one-form in the given Direction and mapping it to the grid frame
 * with the given map.
 *
 * \example
 * \snippet Test_FaceNormal.cpp face_normal_example
 */
template <size_t VolumeDim, typename TargetFrame>
tnsr::i<DataVector, VolumeDim, TargetFrame> unnormalized_face_normal(
    const Mesh<VolumeDim - 1>& interface_mesh,
    const ElementMap<VolumeDim, TargetFrame>& map,
    const Direction<VolumeDim>& direction) noexcept;

template <size_t VolumeDim, typename TargetFrame>
tnsr::i<DataVector, VolumeDim, TargetFrame> unnormalized_face_normal(
    const Mesh<VolumeDim - 1>& interface_mesh,
    const domain::CoordinateMapBase<Frame::Logical, TargetFrame, VolumeDim>&
        map,
    const Direction<VolumeDim>& direction) noexcept;
// @}

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// The unnormalized face normal one form
template <size_t VolumeDim, typename Frame = ::Frame::Inertial>
struct UnnormalizedFaceNormal : db::SimpleTag {
  static std::string name() noexcept { return "UnnormalizedFaceNormal"; }
  using type = tnsr::i<DataVector, VolumeDim, Frame>;
};

template <size_t VolumeDim, typename Frame = ::Frame::Inertial>
struct UnnormalizedFaceNormalCompute
    : db::ComputeTag, UnnormalizedFaceNormal<VolumeDim, Frame> {
  using base = UnnormalizedFaceNormal<VolumeDim, Frame>;
  static constexpr tnsr::i<DataVector, VolumeDim, Frame> (*function)(
      const ::Mesh<VolumeDim - 1>&, const ::ElementMap<VolumeDim, Frame>&,
      const ::Direction<VolumeDim>&) = unnormalized_face_normal;
  using argument_tags =
      tmpl::list<Mesh<VolumeDim - 1>, ElementMap<VolumeDim, Frame>,
                 Direction<VolumeDim>>;
  using volume_tags = tmpl::list<ElementMap<VolumeDim, Frame>>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// Specialisation of UnnormalizedFaceNormal for the external boundaries which
/// inverts the normals. Since ExternalBoundariesDirections are meant to
/// represent ghost elements, the normals should correspond to the normals in
/// said element, which are inverted with respect to the current element.
template <size_t VolumeDim, typename Frame>
struct InterfaceComputeItem<Tags::BoundaryDirectionsExterior<VolumeDim>,
                            UnnormalizedFaceNormalCompute<VolumeDim, Frame>>
    : db::PrefixTag,
      db::ComputeTag,
      Tags::Interface<Tags::BoundaryDirectionsExterior<VolumeDim>,
                      Tags::UnnormalizedFaceNormal<VolumeDim, Frame>> {
  using dirs = BoundaryDirectionsExterior<VolumeDim>;

  static std::string name() noexcept {
    return "BoundaryDirectionsExterior<UnnormalizedFaceNormal>";
  }

  static auto function(
      const db::item_type<Tags::Interface<dirs, Mesh<VolumeDim - 1>>>& meshes,
      const db::item_type<Tags::ElementMap<VolumeDim, Frame>>& map) noexcept {
    std::unordered_map<::Direction<VolumeDim>,
                       tnsr::i<DataVector, VolumeDim, Frame>>
        normals{};
    for (const auto& direction_and_mesh : meshes) {
      const auto& direction = direction_and_mesh.first;
      const auto& mesh = direction_and_mesh.second;
      auto internal_face_normal =
          unnormalized_face_normal(mesh, map, direction);
      std::transform(internal_face_normal.begin(), internal_face_normal.end(),
                     internal_face_normal.begin(), std::negate<>());
      normals[direction] = std::move(internal_face_normal);
    }
    return normals;
  }

  using argument_tags = tmpl::list<Tags::Interface<dirs, Mesh<VolumeDim - 1>>,
                                   Tags::ElementMap<VolumeDim, Frame>>;
};

/// The face normal dotted into the Tensor or Variables in `Tag`
template <typename Tag, typename = std::nullptr_t>
struct NormalDot;

/// \cond
template <typename Tag>
struct NormalDot<Tag, Requires<tt::is_a_v<Tensor, db::item_type<Tag>>>>
    : db::PrefixTag, db::SimpleTag {
  using type = TensorMetafunctions::remove_first_index<db::item_type<Tag>>;
  using tag = Tag;
  static std::string name() noexcept {
    return "NormalDot(" + db::tag_name<Tag>() + ")";
  }
};

template <typename Tag>
struct NormalDot<Tag, Requires<tt::is_a_v<::Variables, db::item_type<Tag>>>>
    : db::PrefixTag, db::SimpleTag {
  using type = db::item_type<Tag>;
  using tag = Tag;
  static std::string name() noexcept {
    return "NormalDot(" + db::tag_name<Tag>() + ")";
  }
};
/// \endcond

}  // namespace Tags

/*!
 * \brief Compute the `normal` dotted into all tensors in the `variables`.
 */
template <typename VariablesTags, size_t VolumeDim, typename Frame>
static Variables<db::wrap_tags_in<Tags::NormalDot, VariablesTags>> normal_dot(
    const Variables<VariablesTags>& variables,
    const tnsr::i<DataVector, VolumeDim, Frame>& normal) noexcept {
  auto result = Variables<db::wrap_tags_in<Tags::NormalDot, VariablesTags>>(
      variables.number_of_grid_points(), 0.);
  // Check if this can be optimized for Variables
  tmpl::for_each<VariablesTags>(
      [&result, &variables, &normal ](auto local_tag) noexcept {
        using tensor_tag = tmpl::type_from<decltype(local_tag)>;
        auto& result_tensor = get<Tags::NormalDot<tensor_tag>>(result);
        const auto& tensor = get<tensor_tag>(variables);
        for (auto it = result_tensor.begin(); it != result_tensor.end(); ++it) {
          const auto result_indices = result_tensor.get_tensor_index(it);
          for (size_t d = 0; d < VolumeDim; ++d) {
            *it += normal.get(d) * tensor.get(prepend(result_indices, d));
          }
        }
      });
  return result;
}

namespace Tags {

/*!
 * \brief Compute the interface unit normal dotted into all tensors in the
 * `VariablesTag`.
 *
 * \see `normal_dot`
 */
template <typename VariablesTag, size_t VolumeDim, typename Frame>
struct NormalDotCompute : db::add_tag_prefix<NormalDot, VariablesTag>,
                          db::ComputeTag {
  using base = db::add_tag_prefix<NormalDot, VariablesTag>;
  using argument_tags = tmpl::list<
      VariablesTag,
      Tags::Normalized<Tags::UnnormalizedFaceNormal<VolumeDim, Frame>>>;
  static constexpr auto function =
      normal_dot<db::get_variables_tags_list<VariablesTag>, VolumeDim, Frame>;
};

}  // namespace Tags
