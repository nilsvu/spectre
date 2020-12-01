// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>
#include <boost/range/join.hpp>
#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/ApplyFace.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/Tags.hpp"
#include "Elliptic/FirstOrderOperator.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/BoundaryData.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Protocols.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TupleSlice.hpp"

namespace elliptic::dg::subdomain_operator {

/// Wrap the `Tag` in `LinearSolver::Schwarz::Tags::Overlaps`, except if it is
/// included in `TakeFromCenterTags`.
template <typename Tag, typename Dim, typename OptionsGroup,
          typename TakeFromCenterTags>
struct make_overlap_tag {
  using type = tmpl::conditional_t<
      tmpl::list_contains_v<TakeFromCenterTags, Tag>, Tag,
      LinearSolver::Schwarz::Tags::Overlaps<Tag, Dim::value, OptionsGroup>>;
};

/*!
 * \brief A first-order DG operator on an element-centered subdomain
 *
 * This operator is a restriction of the full DG-operator to an element-centered
 * subdomain with a few points overlap into neighboring elements. It is a
 * `LinearSolver::Schwarz::protocols::SubdomainOperator` to be used with the
 * Schwarz linear solver when it solves a first-order DG operator.
 *
 * This operator requires the following tags are available on overlap regions
 * with neighboring elements:
 *
 * - Geometric quantities provided by
 *   `elliptic::dg::subdomain_operator::InitializeElement`.
 * - All `FluxesComputerTag::type::argument_tags` and
 *   `SourcesComputer::argument_tags`, except those listed in
 *   `FluxesArgsTagsFromCenter` and `SourcesArgsTagsFromCenter` which will be
 *   taken from the central element's DataBox.
 * - The `FluxesComputerTag::type::argument_tags` wrapped in
 *   `domain::Tags::Face`, except those listed in
 *   `FluxesComputerTag::type::volume_tags`.
 */
template <size_t Dim, typename PrimalFields, typename AuxiliaryFields,
          typename FluxesComputerTag, typename SourcesComputer,
          typename NumericalFluxesComputerTag, typename OptionsGroup,
          typename FluxesArgsTagsFromCenter = tmpl::list<>,
          typename SourcesArgsTagsFromCenter = tmpl::list<>>
struct SubdomainOperator
    : tt::ConformsTo<LinearSolver::Schwarz::protocols::SubdomainOperator> {
 public:
  static constexpr size_t volume_dim = Dim;
  using primal_fields = PrimalFields;
  using auxiliary_fields = AuxiliaryFields;
  using fluxes_computer_tag = FluxesComputerTag;
  using sources_computer = SourcesComputer;
  using numerical_fluxes_computer_tag = NumericalFluxesComputerTag;
  using options_group = OptionsGroup;
  using fluxes_args_tags_from_center = FluxesArgsTagsFromCenter;
  using sources_args_tags_from_center = SourcesArgsTagsFromCenter;

 private:
  using all_fields_tags = tmpl::append<PrimalFields, AuxiliaryFields>;
  using vars_tag = ::Tags::Variables<all_fields_tags>;
  using fluxes_tag = db::add_tag_prefix<::Tags::Flux, vars_tag,
                                        tmpl::size_t<Dim>, Frame::Inertial>;
  using div_fluxes_tag = db::add_tag_prefix<::Tags::div, fluxes_tag>;
  using FluxesComputerType = typename FluxesComputerTag::type;
  using fluxes_args_tags = typename FluxesComputerType::argument_tags;
  static constexpr size_t num_fluxes_args = tmpl::size<fluxes_args_tags>::value;
  using sources_args_tags = typename SourcesComputer::argument_tags;
  static constexpr size_t num_sources_args =
      tmpl::size<sources_args_tags>::value;
  using NumericalFluxesComputerType = typename NumericalFluxesComputerTag::type;

  // Shortcuts to wrap tags and types in overlap maps
  template <typename Tag>
  using overlaps_tag =
      LinearSolver::Schwarz::Tags::Overlaps<Tag, Dim, OptionsGroup>;
  template <typename ValueType>
  using overlaps = LinearSolver::Schwarz::OverlapMap<Dim, ValueType>;

  // A map where we cache boundary data between invocations of the operator in
  // differend faces.
  using NeighborsBoundaryDataCache = std::unordered_map<
      std::pair<ElementId<Dim>, ::dg::MortarId<Dim>>,
      ::dg::FirstOrderScheme::BoundaryData<NumericalFluxesComputerType>,
      boost::hash<std::pair<ElementId<Dim>, ::dg::MortarId<Dim>>>>;
  struct NeighborsBoundaryDataCacheTag {
    using type = NeighborsBoundaryDataCache;
  };

  template <typename Tag>
  struct NeighborsBufferTag {
    using type = FixedHashMap<maximum_number_of_neighbors(Dim), ElementId<Dim>,
                              typename Tag::type>;
  };

  struct ExtendedResultBufferTag {
    using type = typename vars_tag::type;
  };

  using Buffer = tuples::TaggedTuple<
      // These quantities are cached to share data between invocations of the
      // operator on the volume and on the faces
      fluxes_tag, div_fluxes_tag, NeighborsBoundaryDataCacheTag,
      // The following quantities are buffered only to avoid re-allocating
      // memory between consecutive operator applications, not to keep the data
      // around
      domain::Tags::Faces<Dim, fluxes_tag>,
      domain::Tags::Faces<Dim, div_fluxes_tag>, NeighborsBufferTag<vars_tag>,
      NeighborsBufferTag<fluxes_tag>, NeighborsBufferTag<div_fluxes_tag>,
      NeighborsBufferTag<domain::Tags::Faces<Dim, fluxes_tag>>,
      NeighborsBufferTag<domain::Tags::Faces<Dim, div_fluxes_tag>>,
      NeighborsBufferTag<ExtendedResultBufferTag>>;

 public:
  explicit SubdomainOperator(const size_t central_num_points) noexcept
      : buffer_{typename fluxes_tag::type{central_num_points},
                typename div_fluxes_tag::type{central_num_points},
                NeighborsBoundaryDataCache{},
                DirectionMap<Dim, typename fluxes_tag::type>{},
                DirectionMap<Dim, typename div_fluxes_tag::type>{},
                typename NeighborsBufferTag<vars_tag>::type{},
                typename NeighborsBufferTag<fluxes_tag>::type{},
                typename NeighborsBufferTag<div_fluxes_tag>::type{},
                typename NeighborsBufferTag<
                    domain::Tags::Faces<Dim, fluxes_tag>>::type{},
                typename NeighborsBufferTag<
                    domain::Tags::Faces<Dim, div_fluxes_tag>>::type{},
                typename NeighborsBufferTag<ExtendedResultBufferTag>::type{}} {}

  struct element_operator {
    using argument_tags =
        tmpl::append<tmpl::list<domain::Tags::Mesh<Dim>,
                                domain::Tags::InverseJacobian<
                                    Dim, Frame::Logical, Frame::Inertial>,
                                FluxesComputerTag>,
                     fluxes_args_tags, sources_args_tags>;

    template <typename... RemainingArgs>
    static void apply(
        const Mesh<Dim>& mesh,
        const InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>&
            inv_jacobian,
        const FluxesComputerType& fluxes_computer,
        const RemainingArgs&... expanded_remaining_args) noexcept {
      const auto remaining_args =
          std::forward_as_tuple(expanded_remaining_args...);
      // The LinearSolver::Schwarz::protocols::SubdomainOperator defines the
      // order and type of these three trailing arguments
      const auto& [operand, result, subdomain_operator] =
          tuple_tail<3>(remaining_args);
      elliptic::first_order_operator<PrimalFields, AuxiliaryFields,
                                     SourcesComputer>(
          make_not_null(&(result->element_data)),
          make_not_null(&get<fluxes_tag>(subdomain_operator->buffer_)),
          make_not_null(&get<div_fluxes_tag>(subdomain_operator->buffer_)),
          operand.element_data, mesh, inv_jacobian, fluxes_computer,
          tuple_head<num_fluxes_args>(remaining_args),
          tuple_slice<num_fluxes_args, num_fluxes_args + num_sources_args>(
              remaining_args));
    }
  };

  template <typename Directions>
  struct face_operator {
   private:
    static constexpr bool is_external_boundary =
        std::is_same_v<Directions,
                       domain::Tags::BoundaryDirectionsInterior<Dim>>;
    // These tags we need on overlaps with neighboring elements
    using overlap_tags = db::wrap_tags_in<
        overlaps_tag,
        tmpl::list<
            Tags::ExtrudingExtent, domain::Tags::Mesh<Dim>,
            domain::Tags::Element<Dim>,
            domain::Tags::InverseJacobian<Dim, Frame::Logical, Frame::Inertial>,
            domain::Tags::Faces<
                Dim,
                ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>,
            domain::Tags::Faces<
                Dim,
                ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>>,
            ::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>,
            ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>,
            Tags::NeighborMortars<domain::Tags::Mesh<Dim>, Dim>,
            Tags::NeighborMortars<
                ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>,
                Dim>,
            Tags::NeighborMortars<domain::Tags::Mesh<Dim - 1>, Dim>,
            Tags::NeighborMortars<::Tags::MortarSize<Dim - 1>, Dim>>>;

    // Process arguments needed for fluxes and sources on overlaps
    template <typename Tag, typename VolumeTags>
    struct make_face_tag {
      using type = tmpl::conditional_t<tmpl::list_contains_v<VolumeTags, Tag>,
                                       Tag, domain::Tags::Faces<Dim, Tag>>;
    };
    using overlap_fluxes_args_tags =
        tmpl::transform<fluxes_args_tags,
                        make_overlap_tag<tmpl::_1, tmpl::pin<tmpl::size_t<Dim>>,
                                         tmpl::pin<OptionsGroup>,
                                         tmpl::pin<FluxesArgsTagsFromCenter>>>;
    using overlap_fluxes_args_are_from_center = tmpl::transform<
        fluxes_args_tags,
        tmpl::bind<tmpl::list_contains, tmpl::pin<FluxesArgsTagsFromCenter>,
                   tmpl::_1>>;
    using fluxes_face_args_tags = tmpl::transform<
        fluxes_args_tags,
        make_face_tag<tmpl::_1,
                      tmpl::pin<get_volume_tags<FluxesComputerType>>>>;
    using overlap_fluxes_face_args_tags =
        tmpl::transform<fluxes_face_args_tags,
                        make_overlap_tag<tmpl::_1, tmpl::pin<tmpl::size_t<Dim>>,
                                         tmpl::pin<OptionsGroup>,
                                         tmpl::pin<FluxesArgsTagsFromCenter>>>;
    using overlap_fluxes_args_are_in_volume = tmpl::transform<
        fluxes_args_tags,
        tmpl::bind<tmpl::list_contains,
                   tmpl::pin<get_volume_tags<FluxesComputerType>>, tmpl::_1>>;
    using overlap_sources_args_tags =
        tmpl::transform<sources_args_tags,
                        make_overlap_tag<tmpl::_1, tmpl::pin<tmpl::size_t<Dim>>,
                                         tmpl::pin<OptionsGroup>,
                                         tmpl::pin<SourcesArgsTagsFromCenter>>>;
    using overlap_sources_args_are_from_center = tmpl::transform<
        sources_args_tags,
        tmpl::bind<tmpl::list_contains, tmpl::pin<SourcesArgsTagsFromCenter>,
                   tmpl::_1>>;

   public:
    using argument_tags = tmpl::flatten<tmpl::list<
        domain::Tags::Direction<Dim>, domain::Tags::Element<Dim>,
        domain::Tags::Mesh<Dim>, FluxesComputerTag, NumericalFluxesComputerTag,
        ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>,
        ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>,
        ::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>,
        ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>, overlap_tags,
        fluxes_args_tags, overlap_fluxes_args_tags,
        overlap_fluxes_face_args_tags, overlap_sources_args_tags>>;
    using volume_tags = tmpl::flatten<tmpl::list<
        domain::Tags::Element<Dim>, domain::Tags::Mesh<Dim>, FluxesComputerTag,
        NumericalFluxesComputerTag,
        ::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>,
        ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>, overlap_tags,
        get_volume_tags<FluxesComputerType>, overlap_fluxes_args_tags,
        overlap_fluxes_face_args_tags, overlap_sources_args_tags>>;

    template <typename... Args>
    static void apply(const Direction<Dim>& direction,
                      const Args&... expanded_args) noexcept {
      // We split the arguments in those that the `apply_face` function takes
      // normally and those that it takes packaged in tuples. The args are:
      // 1. The "standard" arguments, i.e. the `argument_tags` up to
      //    `fluxes_args_tags`
      // 2. The `fluxes_args_tags`, `overlap_fluxes_args_tags`,
      //    `overlap_fluxes_face_args_tags` and `overlap_sources_args_tags`
      // 3. The operand, result and subdomain operator, as defined by the
      //    `LinearSolver::Schwarz::protocols::SubdomainOperator`
      // 4. The buffer quantities
      const auto all_args = std::forward_as_tuple(expanded_args...);
      static constexpr size_t num_standard_args =
          sizeof...(expanded_args) - 3 * num_fluxes_args - num_sources_args - 3;
      std::apply(
          [&direction,
           &all_args](const auto&... expanded_standard_args) noexcept {
            const auto& [operand, result, subdomain_operator] =
                tuple_tail<3>(all_args);
            apply_face<is_external_boundary, PrimalFields, AuxiliaryFields,
                       SourcesComputer>(
                result, direction, expanded_standard_args...,
                // fluxes_args_tags
                tuple_slice<num_standard_args,
                            num_standard_args + num_fluxes_args>(all_args),
                // overlap_fluxes_args_tags
                tuple_slice<num_standard_args + num_fluxes_args,
                            num_standard_args + 2 * num_fluxes_args>(all_args),
                overlap_fluxes_args_are_from_center{},
                // overlap_fluxes_face_args_tags
                tuple_slice<num_standard_args + 2 * num_fluxes_args,
                            num_standard_args + 3 * num_fluxes_args>(all_args),
                overlap_fluxes_args_are_in_volume{},
                // overlap_sources_args_tags
                tuple_slice<num_standard_args + 3 * num_fluxes_args,
                            num_standard_args + 3 * num_fluxes_args +
                                num_sources_args>(all_args),
                overlap_sources_args_are_from_center{}, operand,
                get<fluxes_tag>(subdomain_operator->buffer_),
                get<div_fluxes_tag>(subdomain_operator->buffer_),
                make_not_null(&get<NeighborsBoundaryDataCacheTag>(
                    subdomain_operator->buffer_)),
                make_not_null(&get<domain::Tags::Faces<Dim, fluxes_tag>>(
                    subdomain_operator->buffer_)[direction]),
                make_not_null(&get<domain::Tags::Faces<Dim, div_fluxes_tag>>(
                    subdomain_operator->buffer_)[direction]),
                make_not_null(&get<NeighborsBufferTag<vars_tag>>(
                    subdomain_operator->buffer_)),
                make_not_null(&get<NeighborsBufferTag<fluxes_tag>>(
                    subdomain_operator->buffer_)),
                make_not_null(&get<NeighborsBufferTag<div_fluxes_tag>>(
                    subdomain_operator->buffer_)),
                make_not_null(&get<NeighborsBufferTag<
                                  domain::Tags::Faces<Dim, fluxes_tag>>>(
                    subdomain_operator->buffer_)),
                make_not_null(&get<NeighborsBufferTag<
                                  domain::Tags::Faces<Dim, div_fluxes_tag>>>(
                    subdomain_operator->buffer_)),
                make_not_null(&get<NeighborsBufferTag<ExtendedResultBufferTag>>(
                    subdomain_operator->buffer_)));
          },
          tuple_head<num_standard_args>(all_args));
    }
  };

 private:
  Buffer buffer_;
};

}  // namespace elliptic::dg::subdomain_operator
