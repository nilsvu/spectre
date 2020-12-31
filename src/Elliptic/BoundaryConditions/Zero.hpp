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
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace elliptic::BoundaryConditions {

/// \cond
template <size_t Dim, typename FieldTags, typename Registrars>
struct Zero;

namespace Registrars {
template <size_t Dim, typename FieldTags>
struct Zero {
  template <typename Registrars>
  using f = BoundaryConditions::Zero<Dim, FieldTags, Registrars>;
};
}  // namespace Registrars
/// \endcond

template <size_t Dim, typename... FieldTags, typename Registrars>
class Zero<Dim, tmpl::list<FieldTags...>, Registrars>
    : public BoundaryCondition<Dim, Registrars> {
 private:
  template <typename Tag>
  struct BoundaryConditionType {
    static std::string name() noexcept { return db::tag_name<Tag>(); }
    using type = elliptic::BoundaryConditionType;
  };

 public:
  Zero() = default;
  Zero(const Zero&) noexcept = default;
  Zero& operator=(const Zero&) noexcept = default;
  Zero(Zero&&) noexcept = default;
  Zero& operator=(Zero&&) noexcept = default;
  ~Zero() noexcept = default;

  explicit Zero(const typename BoundaryConditionType<
                FieldTags>::type... boundary_condition_types) noexcept
      : boundary_condition_types_{boundary_condition_types...} {}

  using linearized_registrar =
      BoundaryConditions::Registrars::Zero<Dim, tmpl::list<FieldTags...>>;

  const BoundaryConditionBase<Dim, typename Base::linearization_registrars>&
  linearization() const noexcept {
    return *this;
  }

  const tuples::TaggedTuple<
      elliptic::Tags::BoundaryConditionType<FieldTags>...>&
  boundary_condition_types() const noexcept {
    return boundary_condition_types_;
  }

  using argument_tags = tmpl::list<>;
  using volume_tags = tmpl::list<>;

  void operator()(
      const gsl::not_null<typename FieldTags::type*>... fields,
      const gsl::not_null<typename FieldTags::type*>... n_dot_fluxes)
      const noexcept {
    const auto helper = [this](auto field_tag_v, const auto field,
                               const auto n_dot_flux) noexcept {
      using field_tag = decltype(field_tag_v);
      switch (get<field_tag>(boundary_condition_types_)) {
        case elliptic::BoundaryConditions::Dirichlet:
          std::fill(field->begin(), field->end(), 0.);
          break;
        case elliptic::BoundaryConditions::Neumann:
          std::fill(n_dot_flux->begin(), n_dot_flux->end(), 0.);
          break;
        default:
          ERROR("Unsupported boundary condition type: "
                << get<field_tag>(boundary_condition_types_));
      }
    };
    EXPAND_PACK_LEFT_TO_RIGHT(helper(FieldTags{}, fields, n_dot_fluxes));
  }

  // NOLINTNEXTLINE
  void pup(PUP::er& p) noexcept { p | boundary_condition_types_; }

 private:
  tuples::TaggedTuple<elliptic::Tags::BoundaryConditionType<FieldTags>...>
      boundary_condition_types_{};
};

}  // namespace elliptic::BoundaryConditions
