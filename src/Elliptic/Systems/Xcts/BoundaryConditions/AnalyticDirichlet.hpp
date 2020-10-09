// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <ostream>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/BoundaryConditions.hpp"
#include "Elliptic/Protocols.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Xcts::BoundaryConditions {

template <size_t Dim, typename ProviderTag, typename FieldTags>
struct AnalyticDirichlet;

template <typename FieldTags>
struct LinearizedAnalyticDirichlet;

template <size_t Dim, typename ProviderTag, typename... FieldTags>
struct AnalyticDirichlet<Dim, ProviderTag, tmpl::list<FieldTags...>> {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "AnalyticDirichlet boundary conditions"};

  using linearization = LinearizedAnalyticDirichlet<tmpl::list<FieldTags...>>;

  AnalyticDirichlet() = default;
  AnalyticDirichlet(const AnalyticDirichlet&) noexcept = default;
  AnalyticDirichlet& operator=(const AnalyticDirichlet&) noexcept = default;
  AnalyticDirichlet(AnalyticDirichlet&&) noexcept = default;
  AnalyticDirichlet& operator=(AnalyticDirichlet&&) noexcept = default;
  ~AnalyticDirichlet() noexcept = default;

  template <typename Tag>
  static elliptic::BoundaryCondition boundary_condition_type(
      const tnsr::I<DataVector, Dim>& /*x*/,
      const Direction<Dim>& /*direction*/, Tag /*meta*/) {
    return elliptic::BoundaryCondition::Dirichlet;
  }

  using argument_tags =
      tmpl::list<domain::Tags::Coordinates<Dim, Frame::Inertial>, ProviderTag>;
  using volume_tags = tmpl::list<ProviderTag>;

  template <typename Provider>
  static void apply(
      const gsl::not_null<typename FieldTags::type*>... fields,
      const gsl::not_null<typename FieldTags::type*>... /*n_dot_fluxes*/,
      const tnsr::i<DataVector, Dim>& /*inward_pointing_face_normal*/,
      const tnsr::I<DataVector, Dim>& x, const Provider& provider) noexcept {
    const auto dirichlet_fields =
        provider.variables(x, tmpl::list<FieldTags...>{});
    const auto helper = [&dirichlet_fields](auto field_tag_v,
                                            const auto field) noexcept {
      using field_tag = decltype(field_tag_v);
      *field = std::move(get<field_tag>(dirichlet_fields));
    };
    EXPAND_PACK_LEFT_TO_RIGHT(helper(FieldTags{}, fields));
  }

  void pup(PUP::er& /* p */) noexcept {}  // NOLINT
};

template <typename... FieldTags>
struct LinearizedAnalyticDirichlet<tmpl::list<FieldTags...>> {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "LinearizedAnalyticDirichlet boundary conditions"};

  LinearizedAnalyticDirichlet() = default;
  LinearizedAnalyticDirichlet(const LinearizedAnalyticDirichlet&) noexcept =
      default;
  LinearizedAnalyticDirichlet& operator=(
      const LinearizedAnalyticDirichlet&) noexcept = default;
  LinearizedAnalyticDirichlet(LinearizedAnalyticDirichlet&&) noexcept = default;
  LinearizedAnalyticDirichlet& operator=(
      LinearizedAnalyticDirichlet&&) noexcept = default;
  ~LinearizedAnalyticDirichlet() noexcept = default;

  template <size_t Dim, typename Tag>
  static elliptic::BoundaryCondition boundary_condition_type(
      const tnsr::I<DataVector, Dim>& /*x*/,
      const Direction<Dim>& /*direction*/, Tag /*meta*/) {
    return elliptic::BoundaryCondition::Dirichlet;
  }

  using argument_tags = tmpl::list<>;
  using volume_tags = tmpl::list<>;

  // The linearization of the Dirichlet conditions is just zero
  template <size_t Dim>
  static void apply(
      const gsl::not_null<typename FieldTags::type*>... fields,
      const gsl::not_null<typename FieldTags::type*>... /*n_dot_fluxes*/,
      const tnsr::i<DataVector, Dim>& inward_pointing_face_normal) noexcept {
    const auto helper = [&inward_pointing_face_normal](
                            auto field_tag_v, const auto field) noexcept {
      using field_tag = decltype(field_tag_v);
      *field = make_with_value<typename field_tag::type>(
          inward_pointing_face_normal, 0.);
    };
    EXPAND_PACK_LEFT_TO_RIGHT(helper(FieldTags{}, fields));
  }

  void pup(PUP::er& /* p */) noexcept {}  // NOLINT
};

}  // namespace Xcts::BoundaryConditions
