// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Direction.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/MaxNumberOfNeighbors.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/OrientationMapHelpers.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearSolver/InnerProduct.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/SubdomainHelpers.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Tags.hpp"

#include "Domain/LogicalCoordinates.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Tuple.hpp"

namespace elliptic {
namespace dg {
namespace SubdomainOperator_detail {

/*!
 * \brief Data on a region within an element that extends a certain number of
 * grid points from a face into the volume.
 */
template <size_t Dim, typename FieldTags, typename FluxesArgs,
          typename SourcesArgs>
struct OverlapData {
  static constexpr size_t volume_dim = Dim;
  using field_tags = FieldTags;

  // Variable quantities
  Variables<FieldTags> field_data{};

  // Geometric quantities
  // TODO: separate these from the variable quantities so they don't need to be
  // copied around when applying the operator
  Mesh<volume_dim> volume_mesh{};
  ElementMap<volume_dim, Frame::Inertial> element_map{};
  /// Direction from which the overlap extends into the overlapped element
  Direction<volume_dim> direction{};
  Index<volume_dim> overlap_extents{};
  ::dg::MortarMap<volume_dim, Mesh<volume_dim - 1>>
      perpendicular_mortar_meshes{};
  ::dg::MortarMap<volume_dim, ::dg::MortarSize<volume_dim - 1>>
      perpendicular_mortar_sizes{};
  tuple_from_typelist<FluxesArgs> fluxes_args;
  tuple_from_typelist<SourcesArgs> sources_args;

  void zero() noexcept {
    field_data = make_with_value<Variables<FieldTags>>(field_data, 0.);
  }

  size_t overlap_extent() const noexcept {
    return overlap_extents[direction.dimension()];
  }

  double overlap_width() const noexcept {
    return LinearSolver::schwarz_detail::overlap_width(
        volume_mesh.slice_through(direction.dimension()), overlap_extent());
  }

  tnsr::I<DataVector, volume_dim, Frame::Logical> logical_coordinates() const
      noexcept {
    return LinearSolver::schwarz_detail::restrict_to_overlap(
        ::logical_coordinates(volume_mesh), volume_mesh.extents(),
        overlap_extents, direction);
  }

  // OverlapData() = default;
  // OverlapData(const OverlapData&) noexcept = default;
  // OverlapData& operator=(const OverlapData&) noexcept = default;
  // OverlapData(OverlapData&&) noexcept = default;
  // OverlapData& operator=(OverlapData&&) noexcept = default;
  // ~OverlapData() noexcept = default;

  // explicit OverlapData(const size_t num_points) noexcept
  //     : field_data{num_points} {}

  /*!
   * \brief Extend the field data to the full mesh by filling it with zeros and
   * adding the overlapping slices.
   *
   * Note that `direction` must be consistent with the `volume_mesh` and the
   * `overlap_extents` in the sense that the `direction.dimension()` is the
   * overlapping dimension. This means the data must be oriented correctly.
   */
  Variables<FieldTags> extended_field_data() const noexcept {
    return LinearSolver::schwarz_detail::extended_overlap_data(
        field_data, volume_mesh.extents(), overlap_extents, direction);
  }

  void orient(const OrientationMap<Dim>& orientation) noexcept {
    if (orientation.is_aligned()) {
      return;
    }
    field_data = orient_variables(field_data, overlap_extents, orientation);
    // inv_jacobian =
    //     orient_tensor(inv_jacobian, volume_mesh.extents(), orientation);
    // magnitude_of_face_normal = orient_tensor_on_slice(
    //     magnitude_of_face_normal,
    //     volume_mesh.slice_away(direction.dimension()).extents(),
    //     direction.dimension(), orientation);

    // for (auto& mortar_id_and_mesh : perpendicular_mortar_meshes) {
    //   const auto& mortar_id = mortar_id_and_mesh.first;
    //   auto& mortar_mesh = mortar_id_and_mesh.second;
    //   auto& mortar_size = perpendicular_mortar_sizes[mortar_id];
    //   const auto& face_direction = mortar_id.first;
    //   const auto orientation_on_face =
    //       orientation.slice_away(face_direction.dimension());
    //   mortar_mesh = Mesh<Dim - 1>{orientation_on_face(mortar_mesh)};
    //   mortar_size = orientation_on_face.permute_from_neighbor(mortar_size);
    // }

    // TODO: orient fluxes and sources args

    // Orient these quantities last because previous calls may use them
    volume_mesh = orientation(volume_mesh);
    direction = orientation(direction);
    overlap_extents = Index<volume_dim>{
        orientation.permute_from_neighbor(overlap_extents.indices())};
  }

  template <typename FieldsType>
  void add_to(const gsl::not_null<FieldsType*> fields) const noexcept {
    *fields += this->extended_field_data();
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept {
    p | field_data;
    p | volume_mesh;
    p | element_map;
    p | direction;
    p | overlap_extents;
    p | perpendicular_mortar_meshes;
    p | perpendicular_mortar_sizes;
    p | sources_args;
    p | fluxes_args;
  }

  template <typename RhsFieldTags>
  OverlapData& operator+=(const OverlapData<Dim, RhsFieldTags, FluxesArgs,
                                            SourcesArgs>& rhs) noexcept {
    field_data += rhs.field_data;
    return *this;
  }
  template <typename RhsFieldTags>
  OverlapData& operator-=(const OverlapData<Dim, RhsFieldTags, FluxesArgs,
                                            SourcesArgs>& rhs) noexcept {
    field_data -= rhs.field_data;
    return *this;
  }
  OverlapData& operator/=(const double scalar) noexcept {
    field_data /= scalar;
    return *this;
  }
};

template <size_t Dim, typename LhsFieldTags, typename RhsFieldTags,
          typename FluxesArgs, typename SourcesArgs>
OverlapData<Dim, LhsFieldTags, FluxesArgs, SourcesArgs> operator-(
    const OverlapData<Dim, LhsFieldTags, FluxesArgs, SourcesArgs>& lhs,
    const OverlapData<Dim, RhsFieldTags, FluxesArgs, SourcesArgs>&
        rhs) noexcept {
  return {lhs.field_data - rhs.field_data,
          lhs.volume_mesh,
          lhs.element_map,
          lhs.direction,
          lhs.overlap_extents,
          lhs.perpendicular_mortar_meshes,
          lhs.perpendicular_mortar_sizes,
          lhs.fluxes_args,
          lhs.sources_args};
}

template <size_t Dim, typename FieldTags, typename FluxesArgs,
          typename SourcesArgs>
OverlapData<Dim, FieldTags, FluxesArgs, SourcesArgs> operator*(
    const double scalar,
    const OverlapData<Dim, FieldTags, FluxesArgs, SourcesArgs>& data) noexcept {
  return {scalar * data.field_data,
          data.volume_mesh,
          data.element_map,
          data.direction,
          data.overlap_extents,
          data.perpendicular_mortar_meshes,
          data.perpendicular_mortar_sizes,
          data.fluxes_args,
          data.sources_args};
}

}  // namespace SubdomainOperator_detail
}  // namespace dg
}  // namespace elliptic

namespace LinearSolver {
namespace InnerProductImpls {

template <size_t Dim, typename LhsFieldTags, typename RhsFieldTags,
          typename FluxesArgs, typename SourcesArgs>
struct InnerProductImpl<elliptic::dg::SubdomainOperator_detail::OverlapData<
                            Dim, LhsFieldTags, FluxesArgs, SourcesArgs>,
                        elliptic::dg::SubdomainOperator_detail::OverlapData<
                            Dim, RhsFieldTags, FluxesArgs, SourcesArgs>> {
  static double apply(
      const elliptic::dg::SubdomainOperator_detail::OverlapData<
          Dim, LhsFieldTags, FluxesArgs, SourcesArgs>& lhs,
      const elliptic::dg::SubdomainOperator_detail::OverlapData<
          Dim, RhsFieldTags, FluxesArgs, SourcesArgs>& rhs) noexcept {
    return inner_product(lhs.field_data, rhs.field_data);
  }
};

}  // namespace InnerProductImpls
}  // namespace LinearSolver

namespace elliptic {
namespace dg {
namespace SubdomainOperator_detail {

template <size_t Dim, typename TagsList, typename OptionsGroup,
          typename FluxesArgumentTags, typename FluxesArgs,
          typename SourcesArgumentTags, typename SourcesArgs>
struct CollectOverlapData {
  using const_global_cache_tags =
      tmpl::list<LinearSolver::schwarz_detail::Tags::Overlap<OptionsGroup>>;
  using argument_tags = tmpl::append<
      tmpl::list<domain::Tags::Mesh<Dim>, domain::Tags::ElementMap<Dim>,
                 domain::Tags::Direction<Dim>,
                 LinearSolver::schwarz_detail::Tags::Overlap<OptionsGroup>,
                 ::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>,
                 ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>>,
      FluxesArgumentTags, SourcesArgumentTags>;
  using volume_tags =
      tmpl::list<domain::Tags::Mesh<Dim>, domain::Tags::ElementMap<Dim>,
                 LinearSolver::schwarz_detail::Tags::Overlap<OptionsGroup>,
                 ::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>,
                 ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>>;

  // Call operator instead of static apply function because `interface_apply`
  // currently doesn't support apply function templates
  template <typename FieldsType, typename... FluxesAndSourcesArgs>
  auto operator()(
      const Mesh<Dim>& mesh,
      const ElementMap<Dim, Frame::Inertial>& element_map,
      const Direction<Dim>& direction, const size_t overlap,
      const db::const_item_type<
          ::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>>& mortar_meshes,
      const db::const_item_type<
          ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>>& mortar_sizes,
      const FluxesAndSourcesArgs&... expanded_fluxes_and_sources_args,
      const FieldsType& fields) const noexcept {
    const size_t dimension = direction.dimension();
    const auto overlap_extents = LinearSolver::schwarz_detail::overlap_extents(
        mesh.extents(), overlap, dimension);
    const auto fields_on_overlap =
        LinearSolver::schwarz_detail::data_on_overlap(
            fields, mesh.extents(), overlap_extents, direction);
    const auto perpendicular_mortar_meshes =
        LinearSolver::schwarz_detail::perpendicular(mortar_meshes, direction);
    const auto perpendicular_mortar_sizes =
        LinearSolver::schwarz_detail::perpendicular(mortar_sizes, direction);
    const std::tuple<FluxesAndSourcesArgs...> fluxes_and_sources_args{
        expanded_fluxes_and_sources_args...};
    const auto fluxes_args =
        tuple_head<sizeof...(FluxesAndSourcesArgs)>(fluxes_and_sources_args);
    const auto sources_args =
        tuple_tail<sizeof...(FluxesAndSourcesArgs)>(fluxes_and_sources_args);
    // TODO: Can infer the FluxesArgs and SourcesArgs?
    return OverlapData<Dim, TagsList, FluxesArgs, SourcesArgs>{
        Variables<TagsList>{fields_on_overlap},
        mesh,
        element_map,
        direction,
        overlap_extents,
        perpendicular_mortar_meshes,
        perpendicular_mortar_sizes,
        fluxes_args,
        sources_args};
  }
};

}  // namespace SubdomainOperator_detail
}  // namespace dg
}  // namespace elliptic
