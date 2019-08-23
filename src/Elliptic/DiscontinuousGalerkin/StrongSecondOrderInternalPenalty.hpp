// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function lift_flux.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesHelpers.hpp"
#include "Domain/SurfaceJacobian.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarData.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/LinearOperators/Mass.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Utilities/TMPL.hpp"

#include "Parallel/Printf.hpp"

namespace elliptic {
namespace dg {
namespace Schemes {

namespace StrongSecondOrderInternalPenalty_detail {

Scalar<DataVector> penalty(const Scalar<DataVector>& element_size,
                           const size_t polynomial_degree,
                           const double penalty_parameter) noexcept {
  return Scalar<DataVector>(penalty_parameter * square(polynomial_degree + 1) /
                            get(element_size));
}

template <typename ResultVariables, typename JumpNormalFluxesTags,
          typename JumpFluxesTags, size_t VolumeDim>
ResultVariables lifted_internal_flux(
    const Variables<JumpNormalFluxesTags>& jump_normal_fluxes_on_face,
    const Variables<JumpFluxesTags>& jump_fluxes_on_face,
    const Mesh<VolumeDim>& volume_mesh,
    const Direction<VolumeDim>& face_direction,
    const Scalar<DataVector>& logical_to_inertial_surface_jacobian_determinant,
    const InverseJacobian<DataVector, VolumeDim, Frame::Logical,
                          Frame::Inertial>& inverse_volume_jacobian,
    const tnsr::i<DataVector, VolumeDim, Frame::Inertial>& face_normal,
    const Scalar<DataVector>& penalty) noexcept {
  const size_t face_dimension = face_direction.dimension();
  const Mesh<VolumeDim - 1> face_mesh = volume_mesh.slice_away(face_dimension);

  const auto mass_term_on_face = ResultVariables(mass_on_face(
      Variables<JumpFluxesTags>(
          get(penalty) * normal_dot(jump_normal_fluxes_on_face, face_normal) +
          0.5 * jump_fluxes_on_face),
      face_mesh, logical_to_inertial_surface_jacobian_determinant));

  const auto div_term_on_face = mass_on_face(
      // Here we should compute the normal-fluxes from the
      // normal-times-field-jumps _on the face_ instead of using the
      // the normal-fluxes that were computed on each side of the mortar. This
      // only makes a difference if the flux-computation depends on the field
      // values. We are making the approximation that this is the case, since it
      // most commonly is and we can then avoid communicating the
      // normal-times-fields separately. Even if the approximation does not
      // hold, we can hope that the scheme converges anyway. This should be
      // revisited if it turns out to be problematic.
      jump_normal_fluxes_on_face, face_mesh,
      logical_to_inertial_surface_jacobian_determinant);
  Variables<db::wrap_tags_in<::Tags::Mass, JumpNormalFluxesTags>>
      div_term_in_volume{volume_mesh.number_of_grid_points(), 0.};
  add_slice_to_data(make_not_null(&div_term_in_volume), div_term_on_face,
                    volume_mesh.extents(), face_dimension,
                    index_to_slice_at(volume_mesh.extents(), face_direction));

  auto result =
      ResultVariables(-0.5 * stiffness(div_term_in_volume, volume_mesh,
                                       inverse_volume_jacobian));
  add_slice_to_data(make_not_null(&result), mass_term_on_face,
                    volume_mesh.extents(), face_dimension,
                    index_to_slice_at(volume_mesh.extents(), face_direction));
  return result;
}

template <typename ResultVariables, typename NormalFluxesTags, size_t VolumeDim>
ResultVariables lifted_dirichlet_flux(
    const Variables<NormalFluxesTags>& normal_fluxes_on_face,
    const Mesh<VolumeDim>& volume_mesh,
    const Direction<VolumeDim>& face_direction,
    const Scalar<DataVector>& logical_to_inertial_surface_jacobian_determinant,
    const InverseJacobian<DataVector, VolumeDim, Frame::Logical,
                          Frame::Inertial>& inverse_volume_jacobian,
    const tnsr::i<DataVector, VolumeDim, Frame::Inertial>& face_normal,
    const Scalar<DataVector>& penalty) noexcept {
  const size_t face_dimension = face_direction.dimension();
  const Mesh<VolumeDim - 1> face_mesh = volume_mesh.slice_away(face_dimension);

  const auto mass_term_on_face = ResultVariables(mass_on_face(
      Variables<db::wrap_tags_in<::Tags::NormalDot, NormalFluxesTags>>(
          2. * get(penalty) * normal_dot(normal_fluxes_on_face, face_normal)),
      face_mesh, logical_to_inertial_surface_jacobian_determinant));

  const auto div_term_on_face =
      mass_on_face(normal_fluxes_on_face, face_mesh,
                   logical_to_inertial_surface_jacobian_determinant);
  Variables<db::wrap_tags_in<::Tags::Mass, NormalFluxesTags>>
      div_term_in_volume{volume_mesh.number_of_grid_points(), 0.};
  add_slice_to_data(make_not_null(&div_term_in_volume), div_term_on_face,
                    volume_mesh.extents(), face_dimension,
                    index_to_slice_at(volume_mesh.extents(), face_direction));

  auto result = ResultVariables(-1. * stiffness(div_term_in_volume, volume_mesh,
                                                inverse_volume_jacobian));
  add_slice_to_data(make_not_null(&result), mass_term_on_face,
                    volume_mesh.extents(), face_dimension,
                    index_to_slice_at(volume_mesh.extents(), face_direction));
  return result;
}

namespace Tags {
struct PolynomialDegree : db::SimpleTag {
  using type = size_t;
  static std::string name() noexcept { return "PolynomialDegree"; }
};
struct ElementSize : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "ElementSize"; }
};
}  // namespace Tags

namespace OptionTags {
struct PenaltyParameter : db::SimpleTag {
  using type = double;
  using container_tag = PenaltyParameter;
  static std::string name() noexcept { return "PenaltyParameter"; }
  static constexpr OptionString help = {
      "The prefactor to the penalty term of the flux."};
  static double lower_bound() { return 1.; }
};
}  // namespace OptionTags

template <size_t Dim, typename RemoteData>
struct compute_packaged_remote_data_impl;

template <size_t Dim, typename... ArgsTags>
struct compute_packaged_remote_data_impl<
    Dim,
    ::dg::MortarData<tmpl::list<ArgsTags...>,
                     tmpl::list<Tags::PolynomialDegree, Tags::ElementSize>>>
    : ::Tags::PackagedData<::dg::MortarData<
          tmpl::list<ArgsTags...>,
          tmpl::list<Tags::PolynomialDegree, Tags::ElementSize>>>,
      db::ComputeTag {
  using RemoteData =
      ::dg::MortarData<tmpl::list<ArgsTags...>,
                       tmpl::list<Tags::PolynomialDegree, Tags::ElementSize>>;
  using base = ::Tags::PackagedData<RemoteData>;
  using argument_tags = tmpl::list<
      ::Tags::Direction<Dim>, ::Tags::Mesh<Dim>,
      ::Tags::Magnitude<::Tags::UnnormalizedFaceNormal<Dim, Frame::Inertial>>,
      ArgsTags...>;
  using volume_tags = tmpl::list<::Tags::Mesh<Dim>>;
  static auto function(const Direction<Dim>& face_direction,
                       const Mesh<Dim>& volume_mesh,
                       const Scalar<DataVector>& magnitude_of_face_normal,
                       const db::item_type<ArgsTags>&... args) noexcept {
    const size_t face_dimension = face_direction.dimension();
    RemoteData remote_data{};
    remote_data.mortar_data.initialize(
        volume_mesh.slice_away(face_dimension).number_of_grid_points());
    const auto helper = [&remote_data](const auto arg_tag_v,
                                       const auto arg) noexcept {
      using arg_tag = std::decay_t<decltype(arg_tag_v)>;
      get<arg_tag>(remote_data.mortar_data) = arg;
    };
    EXPAND_PACK_LEFT_TO_RIGHT(helper(ArgsTags{}, args));
    get<Tags::PolynomialDegree>(remote_data.face_data) =
        volume_mesh.extents(face_dimension) - 1;
    // Factor 2 is the size in logical coordinates
    get<Tags::ElementSize>(remote_data.face_data) =
        Scalar<DataVector>(2. / get(magnitude_of_face_normal));
    return remote_data;
  }
};

template <size_t Dim, typename LocalData>
struct compute_packaged_local_data_impl;

template <size_t Dim, typename... ArgsTags>
struct compute_packaged_local_data_impl<
    Dim,
    ::dg::MortarData<tmpl::list<ArgsTags...>,
                     tmpl::list<Tags::PolynomialDegree, Tags::ElementSize>>>
    : ::Tags::PackagedData<::dg::MortarData<
          tmpl::list<ArgsTags...>,
          tmpl::list<Tags::PolynomialDegree, Tags::ElementSize>>>,
      db::ComputeTag {
  using LocalData =
      ::dg::MortarData<tmpl::list<ArgsTags...>,
                       tmpl::list<Tags::PolynomialDegree, Tags::ElementSize>>;
  using PackagedRemoteDataTag = ::Tags::PackagedData<
      ::dg::MortarData<tmpl::list<ArgsTags...>,
                       tmpl::list<Tags::PolynomialDegree, Tags::ElementSize>>>;
  using base = ::Tags::PackagedData<LocalData>;
  using argument_tags = tmpl::list<PackagedRemoteDataTag>;
  static auto function(
      const db::item_type<PackagedRemoteDataTag>& remote_data) noexcept {
    LocalData local_data{};
    local_data.mortar_data = remote_data.mortar_data;
    get<Tags::PolynomialDegree>(local_data.face_data) =
        get<Tags::PolynomialDegree>(remote_data.face_data);
    get<Tags::ElementSize>(local_data.face_data) =
        get<Tags::ElementSize>(remote_data.face_data);
    return local_data;
  }
};

template <typename Scheme>
struct InitializeElement {
 private:
  static constexpr size_t volume_dim = Scheme::volume_dim;
  using temporal_id_tag = typename Scheme::temporal_id_tag;
  using variables_tag = typename Scheme::variables_tag;
  using operator_applied_to_variables_tag =
      db::add_tag_prefix<temporal_id_tag::template step_prefix, variables_tag>;
  using fluxes_tag =
      db::add_tag_prefix<::Tags::Flux, variables_tag, tmpl::size_t<volume_dim>,
                         Frame::Inertial>;

  template <typename Directions>
  using interior_face_tags = tmpl::list<
      ::Tags::Slice<Directions, fluxes_tag>,
      ::Tags::InterfaceComputeItem<
          Directions,
          ::Tags::NormalDotCompute<fluxes_tag, volume_dim, Frame::Inertial>>,
      ::Tags::InterfaceComputeItem<Directions,
                                   typename Scheme::compute_normal_fluxes>,
      ::Tags::Slice<Directions, ::Tags::JacobianDeterminant<Frame::Logical,
                                                            Frame::Inertial>>,
      ::Tags::InterfaceComputeItem<
          Directions, ::Tags::SurfaceJacobianDeterminantCompute<
                          volume_dim, Frame::Logical, Frame::Inertial>>,
      //   typename dg_scheme::compute_packaged_remote_data,
      ::Tags::InterfaceComputeItem<
          Directions, typename Scheme::compute_packaged_local_data>>;

  using exterior_face_tags = tmpl::list<
      ::Tags::InterfaceComputeItem<
          ::Tags::BoundaryDirectionsExterior<volume_dim>,
          typename Scheme::compute_normal_fluxes_of_fields>,
      ::Tags::Slice<
          ::Tags::BoundaryDirectionsExterior<volume_dim>,
          ::Tags::JacobianDeterminant<Frame::Logical, Frame::Inertial>>,
      ::Tags::InterfaceComputeItem<
          ::Tags::BoundaryDirectionsExterior<volume_dim>,
          ::Tags::SurfaceJacobianDeterminantCompute<volume_dim, Frame::Logical,
                                                    Frame::Inertial>>,
      ::Tags::InterfaceComputeItem<
          ::Tags::BoundaryDirectionsExterior<volume_dim>,
          typename Scheme::impose_boundary_conditions_on_fixed_source::
              compute_packaged_data>>;

 public:
  template <typename DataBox, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(DataBox& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>&
                    /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/, const ParallelComponent* const
                    /*meta*/) noexcept {
    const auto& mesh = db::get<::Tags::Mesh<volume_dim>>(box);
    const size_t num_grid_points = mesh.number_of_grid_points();
    db::item_type<operator_applied_to_variables_tag> operator_applied_to_vars{
        num_grid_points};

    using compute_tags = tmpl::flatten<tmpl::list<
        ::Tags::DerivCompute<
            variables_tag,
            ::Tags::Jacobian<volume_dim, Frame::Inertial, Frame::Logical>>,
        typename Scheme::compute_fluxes,
        ::Tags::DivCompute<
            fluxes_tag,
            ::Tags::Jacobian<volume_dim, Frame::Inertial, Frame::Logical>>,
        interior_face_tags<::Tags::InternalDirections<volume_dim>>,
        interior_face_tags<::Tags::BoundaryDirectionsInterior<volume_dim>>
        ,exterior_face_tags
        >>;

    return std::make_tuple(
        ::Initialization::merge_into_databox<
            InitializeElement,
            db::AddSimpleTags<operator_applied_to_variables_tag>, compute_tags>(
            std::move(box), std::move(operator_applied_to_vars)));
  }
};

template <size_t Dim, typename VariablesTag>
struct ImposeBoundaryConditionsOnSourceImpl;

}  // namespace StrongSecondOrderInternalPenalty_detail

template <size_t Dim, typename FieldsTag, typename VariablesTag,
          typename TemporalIdTag, typename ComputeFluxes,
          typename ComputeNormalFluxes, typename ComputeNormalFluxesOfFields>
struct StrongSecondOrderInternalPenalty {
  static constexpr size_t volume_dim = Dim;
  static constexpr bool use_external_mortars = false;
  using temporal_id_tag = TemporalIdTag;
  using variables_tag = VariablesTag;
  using operator_applied_to_variables_tag =
      db::add_tag_prefix<temporal_id_tag::template step_prefix, variables_tag>;
  using normal_fluxes_tag =
      db::add_tag_prefix<::Tags::NormalFlux, variables_tag, tmpl::size_t<Dim>,
                         Frame::Inertial>;
  using fluxes_tag = db::add_tag_prefix<::Tags::Flux, variables_tag,
                                        tmpl::size_t<Dim>, Frame::Inertial>;
  using div_fluxes_tag = db::add_tag_prefix<::Tags::div, fluxes_tag>;
  using normal_dot_fluxes_tag =
      db::add_tag_prefix<::Tags::NormalDot, fluxes_tag>;

  using compute_fluxes = ComputeFluxes;
  using compute_normal_fluxes = ComputeNormalFluxes;
  using compute_normal_fluxes_of_fields = ComputeNormalFluxesOfFields;

  /// Data that is communicated to and from the remote element.
  /// Must have `project_to_mortar` and `orient_on_slice` functions.
  using RemoteData = ::dg::MortarData<
      tmpl::append<db::get_variables_tags_list<normal_fluxes_tag>,
                   db::get_variables_tags_list<normal_dot_fluxes_tag>>,
      tmpl::list<
          StrongSecondOrderInternalPenalty_detail::Tags::PolynomialDegree,
          StrongSecondOrderInternalPenalty_detail::Tags::ElementSize>>;
  using packaged_remote_data_tag = ::Tags::PackagedData<RemoteData>;
  using compute_packaged_remote_data = StrongSecondOrderInternalPenalty_detail::
      compute_packaged_remote_data_impl<Dim, RemoteData>;

  /// Data from the local element that is needed to compute the lifted fluxes
  /// Must have a `project_to_mortar` function.
  using LocalData = RemoteData;
  using packaged_local_data_tag = packaged_remote_data_tag;
  using compute_packaged_local_data = compute_packaged_remote_data;

  /// The combined local and remote data on the mortar that is assembled
  /// during flux communication. Must have `local_insert` and `remote_insert`
  /// functions that take `LocalData` and `RemoteData`, respectively. Both
  /// functions will be called exactly once in each cycle of
  /// `SendDataForFluxes` and `ReceiveDataForFluxes`.
  using mortar_data_tag =
      ::Tags::SimpleBoundaryData<db::item_type<TemporalIdTag>, LocalData,
                                 RemoteData>;

  using initialize_element =
      StrongSecondOrderInternalPenalty_detail::InitializeElement<
          StrongSecondOrderInternalPenalty>;

  using return_tags = tmpl::list<operator_applied_to_variables_tag,
                                 ::Tags::Mortars<mortar_data_tag, Dim>>;
  using argument_tags = tmpl::list<
      div_fluxes_tag,
      ::Tags::Interface<::Tags::BoundaryDirectionsInterior<Dim>,
                        packaged_local_data_tag>,
      ::Tags::Mesh<Dim>,
      ::Tags::JacobianDeterminant<Frame::Logical, Frame::Inertial>,
      ::Tags::Jacobian<Dim, Frame::Inertial, Frame::Logical>,
      ::Tags::Interface<
          ::Tags::InternalDirections<Dim>,
          ::Tags::SurfaceJacobianDeterminant<Frame::Logical, Frame::Inertial>>,
      ::Tags::Interface<
          ::Tags::BoundaryDirectionsInterior<Dim>,
          ::Tags::SurfaceJacobianDeterminant<Frame::Logical, Frame::Inertial>>,
      ::Tags::Interface<::Tags::InternalDirections<Dim>,
                        ::Tags::Normalized<::Tags::UnnormalizedFaceNormal<
                            Dim, Frame::Inertial>>>,
      ::Tags::Interface<::Tags::BoundaryDirectionsInterior<Dim>,
                        ::Tags::Normalized<::Tags::UnnormalizedFaceNormal<
                            Dim, Frame::Inertial>>>,
      ::Tags::Mortars<::Tags::Mesh<Dim - 1>, Dim>,
      ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>,
      StrongSecondOrderInternalPenalty_detail::OptionTags::PenaltyParameter>;

  using const_global_cache_tags = tmpl::list<
      StrongSecondOrderInternalPenalty_detail::OptionTags::PenaltyParameter>;

  static void apply(
      const gsl::not_null<db::item_type<operator_applied_to_variables_tag>*>
          operator_applied_to_variables,
      const gsl::not_null<db::item_type<::Tags::Mortars<mortar_data_tag, Dim>>*>
          all_mortar_data,
      const db::item_type<div_fluxes_tag>& div_fluxes,
      const std::unordered_map<Direction<Dim>, LocalData>& all_boundary_data,
      const Mesh<Dim>& volume_mesh,
      const Scalar<DataVector>& logical_to_inertial_jacobian_determinant,
      const Jacobian<DataVector, Dim, Frame::Inertial, Frame::Logical>&
          inverse_volume_jacobian,
      const std::unordered_map<Direction<Dim>, Scalar<DataVector>>&
          logical_to_inertial_surface_jacobian_determinant_on_internal_faces,
      const std::unordered_map<Direction<Dim>, Scalar<DataVector>>&
          logical_to_inertial_surface_jacobian_determinant_on_boundary_faces,
      const std::unordered_map<Direction<Dim>,
                               tnsr::i<DataVector, Dim, Frame::Inertial>>&
          internal_face_normals,
      const std::unordered_map<Direction<Dim>,
                               tnsr::i<DataVector, Dim, Frame::Inertial>>&
          external_face_normals,
      const db::item_type<::Tags::Mortars<::Tags::Mesh<Dim - 1>, Dim>>&
          mortar_meshes,
      const db::item_type<::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>>&
          mortar_sizes,
      const double penalty_parameter) noexcept {
    // Apply mass matrix to volume operator
    *operator_applied_to_variables =
        db::item_type<operator_applied_to_variables_tag>(
            -1. * mass(div_fluxes, volume_mesh,
                       logical_to_inertial_jacobian_determinant));

    // Iterate over all mortars. They cover all internal faces of the element.
    for (auto& mortar_id_and_data : *all_mortar_data) {
      // Retrieve mortar data
      const auto& mortar_id = mortar_id_and_data.first;
      auto& mortar_data = mortar_id_and_data.second;
      const auto& direction = mortar_id.first;
      const size_t dimension = direction.dimension();
      const auto& mortar_mesh = mortar_meshes.at(mortar_id);
      const size_t mortar_num_points = mortar_mesh.number_of_grid_points();
      const auto& mortar_size = mortar_sizes.at(mortar_id);
      const auto face_mesh = volume_mesh.slice_away(dimension);

      // Extract local and remote data
      const auto& extracted_mortar_data = mortar_data.extract();
      const LocalData& local_data = extracted_mortar_data.first;
      const RemoteData& remote_data = extracted_mortar_data.second;

      // Combine local and remote data on mortars by computing jumps
      db::item_type<normal_fluxes_tag> normal_fluxes_local{mortar_num_points};
      db::item_type<normal_dot_fluxes_tag> normal_dot_fluxes_local{
          mortar_num_points};
      db::item_type<normal_fluxes_tag> normal_fluxes_remote{
          mortar_num_points};
      db::item_type<normal_dot_fluxes_tag> normal_dot_fluxes_remote{
          mortar_num_points};
      tmpl::for_each<db::get_variables_tags_list<variables_tag>>([
        &normal_fluxes_local, &normal_fluxes_remote, &normal_dot_fluxes_local,
        &normal_dot_fluxes_remote, &local_data, &remote_data
      ](const auto field_tag_v) noexcept {
        using field_tag = tmpl::type_from<decltype(field_tag_v)>;
        using normal_flux_tag =
            ::Tags::NormalFlux<field_tag, tmpl::size_t<Dim>, Frame::Inertial>;
        using normal_dot_flux_tag = ::Tags::NormalDot<
            ::Tags::Flux<field_tag, tmpl::size_t<Dim>, Frame::Inertial>>;
        get<normal_flux_tag>(normal_fluxes_local) =
            get<normal_flux_tag>(local_data.mortar_data);
        get<normal_flux_tag>(normal_fluxes_remote) =
            get<normal_flux_tag>(remote_data.mortar_data);
        get<normal_dot_flux_tag>(normal_dot_fluxes_local) =
            get<normal_dot_flux_tag>(local_data.mortar_data);
        get<normal_dot_flux_tag>(normal_dot_fluxes_remote) =
            get<normal_dot_flux_tag>(remote_data.mortar_data);
      });
      auto jump_normal_fluxes = db::item_type<normal_fluxes_tag>(
          normal_fluxes_local + normal_fluxes_remote);
      auto jump_fluxes = db::item_type<normal_dot_fluxes_tag>(
          normal_dot_fluxes_local + normal_dot_fluxes_remote);

      // Project from the mortar back to the face if needed
      db::item_type<normal_fluxes_tag> jump_normal_fluxes_on_face;
      db::item_type<normal_dot_fluxes_tag> jump_fluxes_on_face;
      if (face_mesh != mortar_mesh or
          std::any_of(
              mortar_size.begin(),
              mortar_size.end(), [](const Spectral::MortarSize s) noexcept {
                return s != Spectral::MortarSize::Full;
              })) {
        jump_normal_fluxes_on_face = ::dg::project_from_mortar(
            std::move(jump_normal_fluxes), face_mesh, mortar_mesh, mortar_size);
        jump_fluxes_on_face = ::dg::project_from_mortar(
            std::move(jump_fluxes), face_mesh, mortar_mesh, mortar_size);
      } else {
        jump_normal_fluxes_on_face = std::move(jump_normal_fluxes);
        jump_fluxes_on_face = std::move(jump_fluxes);
      }

      const auto penalty = StrongSecondOrderInternalPenalty_detail::penalty(
          Scalar<DataVector>(min(
              get(get<
                  StrongSecondOrderInternalPenalty_detail::Tags::ElementSize>(
                  local_data.face_data)),
              get(get<
                  StrongSecondOrderInternalPenalty_detail::Tags::ElementSize>(
                  remote_data.face_data)))),
          std::max(get<StrongSecondOrderInternalPenalty_detail::Tags::
                           PolynomialDegree>(local_data.face_data),
                   get<StrongSecondOrderInternalPenalty_detail::Tags::
                           PolynomialDegree>(remote_data.face_data)),
          penalty_parameter);

      auto lifted_flux =
          StrongSecondOrderInternalPenalty_detail::lifted_internal_flux<
              db::item_type<operator_applied_to_variables_tag>>(
              jump_normal_fluxes_on_face, jump_fluxes_on_face, volume_mesh,
              direction,
              logical_to_inertial_surface_jacobian_determinant_on_internal_faces
                  .at(direction),
              inverse_volume_jacobian, internal_face_normals.at(direction),
              penalty);

      // Add flux for this mortar to full operator
      *operator_applied_to_variables += std::move(lifted_flux);
    }

    // Iterate over the external boundary faces of the element.
    // These are all assumed to be Dirichlet boundaries. Other boundary
    // conditions require slightly fluxes.
    for (auto& direction_and_data : all_boundary_data) {
      const auto& direction = direction_and_data.first;
      const size_t dimension = direction.dimension();
      const auto face_mesh = volume_mesh.slice_away(dimension);
      const size_t face_num_points = face_mesh.number_of_grid_points();
      const auto& local_data = direction_and_data.second;

      // Instead of jumps, just retrieve the normal dotted into the fluxes
      db::item_type<normal_fluxes_tag> normal_fluxes{face_num_points};
      tmpl::for_each<db::get_variables_tags_list<variables_tag>>([
        &normal_fluxes, &local_data
      ](const auto field_tag_v) noexcept {
        using field_tag = tmpl::type_from<decltype(field_tag_v)>;
        using normal_flux_tag =
            ::Tags::NormalFlux<field_tag, tmpl::size_t<Dim>, Frame::Inertial>;
        get<normal_flux_tag>(normal_fluxes) =
            get<normal_flux_tag>(local_data.mortar_data);
      });

      const auto penalty = StrongSecondOrderInternalPenalty_detail::penalty(
          get<StrongSecondOrderInternalPenalty_detail::Tags::ElementSize>(
              local_data.face_data),
          get<StrongSecondOrderInternalPenalty_detail::Tags::PolynomialDegree>(
              local_data.face_data),
          penalty_parameter);

      auto lifted_flux =
          StrongSecondOrderInternalPenalty_detail::lifted_dirichlet_flux<
              db::item_type<operator_applied_to_variables_tag>>(
              normal_fluxes, volume_mesh, direction,
              logical_to_inertial_surface_jacobian_determinant_on_boundary_faces
                  .at(direction),
              inverse_volume_jacobian, external_face_normals.at(direction),
              penalty);

      // Add flux for this mortar to full operator
      *operator_applied_to_variables += std::move(lifted_flux);
    }
  }

  using impose_boundary_conditions_on_fixed_source =
      StrongSecondOrderInternalPenalty_detail::
          ImposeBoundaryConditionsOnSourceImpl<Dim, FieldsTag>;
};

namespace StrongSecondOrderInternalPenalty_detail {

template <size_t Dim, typename VariablesTag>
struct ImposeBoundaryConditionsOnSourceImpl {
  using variables_tag = VariablesTag;
  using fixed_sources_tag =
      db::add_tag_prefix<::Tags::FixedSource, variables_tag>;
  using normal_fluxes_tag =
      db::add_tag_prefix<::Tags::NormalFlux, variables_tag, tmpl::size_t<Dim>,
                         Frame::Inertial>;

  using PackagedData = ::dg::MortarData<
      tmpl::append<db::get_variables_tags_list<normal_fluxes_tag>>,
      tmpl::list<
          StrongSecondOrderInternalPenalty_detail::Tags::PolynomialDegree,
          StrongSecondOrderInternalPenalty_detail::Tags::ElementSize>>;
  using packaged_data_tag = ::Tags::PackagedData<PackagedData>;
  using compute_packaged_data = StrongSecondOrderInternalPenalty_detail::
      compute_packaged_remote_data_impl<Dim, PackagedData>;

  using return_tags = tmpl::list<fixed_sources_tag>;
  using argument_tags = tmpl::list<
      ::Tags::Interface<::Tags::BoundaryDirectionsExterior<Dim>,
                        packaged_data_tag>,
      ::Tags::Mesh<Dim>,
      ::Tags::JacobianDeterminant<Frame::Logical, Frame::Inertial>,
      ::Tags::Jacobian<Dim, Frame::Inertial, Frame::Logical>,
      ::Tags::Interface<
          ::Tags::BoundaryDirectionsExterior<Dim>,
          ::Tags::SurfaceJacobianDeterminant<Frame::Logical, Frame::Inertial>>,
      ::Tags::Interface<::Tags::BoundaryDirectionsInterior<Dim>,
                        ::Tags::Normalized<::Tags::UnnormalizedFaceNormal<
                            Dim, Frame::Inertial>>>,
      StrongSecondOrderInternalPenalty_detail::OptionTags::PenaltyParameter>;

  using const_global_cache_tags = tmpl::list<
      StrongSecondOrderInternalPenalty_detail::OptionTags::PenaltyParameter>;

  static void apply(
      const gsl::not_null<db::item_type<fixed_sources_tag>*> fixed_sources,
      const std::unordered_map<Direction<Dim>, PackagedData>& all_boundary_data,
      const Mesh<Dim>& volume_mesh,
      const Scalar<DataVector>& logical_to_inertial_jacobian_determinant,
      const Jacobian<DataVector, Dim, Frame::Inertial, Frame::Logical>&
          inverse_volume_jacobian,
      const std::unordered_map<Direction<Dim>, Scalar<DataVector>>&
          logical_to_inertial_surface_jacobian_determinant_on_boundary_faces,
      const std::unordered_map<Direction<Dim>,
                               tnsr::i<DataVector, Dim, Frame::Inertial>>&
          face_normals,
      const double penalty_parameter) noexcept {
    // Apply mass matrix to volume operator
    *fixed_sources = db::item_type<fixed_sources_tag>(mass(
        *fixed_sources, volume_mesh, logical_to_inertial_jacobian_determinant));

    // Iterate over the external boundary faces of the element.
    // These are all assumed to be Dirichlet boundaries. Other boundary
    // conditions require slightly fluxes.
    // NONE OF THIS IS DIFFERENT TO THE OPERATOR ABOVE
    for (auto& direction_and_data : all_boundary_data) {
      const auto& direction = direction_and_data.first;
      const size_t dimension = direction.dimension();
      const auto face_mesh = volume_mesh.slice_away(dimension);
      const size_t face_num_points = face_mesh.number_of_grid_points();
      const auto& local_data = direction_and_data.second;

      // Instead of jumps, just retrieve the normal dotted into the fluxes
      db::item_type<normal_fluxes_tag> normal_fluxes{face_num_points};
      tmpl::for_each<db::get_variables_tags_list<variables_tag>>([
        &normal_fluxes, &local_data
      ](const auto field_tag_v) noexcept {
        using field_tag = tmpl::type_from<decltype(field_tag_v)>;
        using normal_flux_tag =
            ::Tags::NormalFlux<field_tag, tmpl::size_t<Dim>, Frame::Inertial>;
        get<normal_flux_tag>(normal_fluxes) =
            get<normal_flux_tag>(local_data.mortar_data);
      });

      const auto penalty = StrongSecondOrderInternalPenalty_detail::penalty(
          get<StrongSecondOrderInternalPenalty_detail::Tags::ElementSize>(
              local_data.face_data),
          get<StrongSecondOrderInternalPenalty_detail::Tags::PolynomialDegree>(
              local_data.face_data),
          penalty_parameter);

      auto lifted_flux =
          StrongSecondOrderInternalPenalty_detail::lifted_dirichlet_flux<
              db::item_type<fixed_sources_tag>>(
              normal_fluxes, volume_mesh, direction,
              logical_to_inertial_surface_jacobian_determinant_on_boundary_faces
                  .at(direction),
              inverse_volume_jacobian,
              // Note that this is the face normal of the element, not that of
              // the exterior (ghost) element.
              face_normals.at(direction), penalty);

      // Add flux for this mortar to full operator
      // ONLY HERE IT IS ADDED TO THE SOURCE INSTEAD OF THE OPERATOR
      // WITH NEGATIVE SIGN BECAUSE WE MOVE IT TO THE RHS
      *fixed_sources -= std::move(lifted_flux);
    }
  }
};

}  // namespace StrongSecondOrderInternalPenalty_detail
}  // namespace Schemes
}  // namespace dg
}  // namespace elliptic
