// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <ostream>
#include <pup.h>
#include <string>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Elliptic/BoundaryConditions/Tags.hpp"
#include "ErrorHandling/Error.hpp"
#include "Options/Options.hpp"
#include "Utilities/Gsl.hpp"
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

  using linearization_registrar = Zero;
};
}  // namespace Registrars

template <size_t Dim, typename FieldTags,
          typename Registrars =
              tmpl::list<BoundaryConditions::Registrars::Zero<Dim, FieldTags>>>
struct Zero;
/// \endcond

template <size_t Dim, typename... FieldTags, typename Registrars>
class Zero<Dim, tmpl::list<FieldTags...>, Registrars>
    : public BoundaryCondition<Dim, Registrars> {
 private:
  using Base = BoundaryCondition<Dim, Registrars>;

 public:
  using options =
      tmpl::list<elliptic::OptionTags::BoundaryConditionType<FieldTags>...>;
  static constexpr Options::String help =
      "Homogeneous (zero) boundary conditions";

  Zero() = default;
  Zero(const Zero&) noexcept = default;
  Zero& operator=(const Zero&) noexcept = default;
  Zero(Zero&&) noexcept = default;
  Zero& operator=(Zero&&) noexcept = default;
  ~Zero() noexcept = default;

  /// \cond
  explicit Zero(CkMigrateMessage* m) noexcept : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Zero);  // NOLINT
  /// \endcond

  explicit Zero(const typename elliptic::OptionTags::BoundaryConditionType<
                FieldTags>::type... boundary_condition_types) noexcept
      : boundary_condition_types_{boundary_condition_types...} {}

  std::unique_ptr<
      BoundaryConditionBase<Dim, typename Base::linearization_registrars>>
  linearization() const noexcept override {
    return std::make_unique<Zero<Dim, tmpl::list<FieldTags...>,
                                 typename Base::linearization_registrars>>(
        get<elliptic::Tags::BoundaryConditionType<FieldTags>>(
            boundary_condition_types_)...);
  }

  const auto& boundary_condition_types() const noexcept {
    return boundary_condition_types_;
  }

  using argument_tags = tmpl::list<>;
  using volume_tags = tmpl::list<>;

  void operator()(
      const gsl::not_null<typename FieldTags::type*>... fields,
      const gsl::not_null<typename FieldTags::type*>... n_dot_fluxes) const
      noexcept {
    const auto helper = [this](auto field_tag_v, const auto field,
                               const auto n_dot_flux) noexcept {
      using field_tag = decltype(field_tag_v);
      switch (get<elliptic::Tags::BoundaryConditionType<field_tag>>(
          boundary_condition_types_)) {
        case elliptic::BoundaryConditionType::Dirichlet:
          std::fill(field->begin(), field->end(), 0.);
          break;
        case elliptic::BoundaryConditionType::Neumann:
          std::fill(n_dot_flux->begin(), n_dot_flux->end(), 0.);
          break;
        default:
          ERROR("Unsupported boundary condition type: "
                << get<elliptic::Tags::BoundaryConditionType<field_tag>>(
                       boundary_condition_types_));
      }
    };
    EXPAND_PACK_LEFT_TO_RIGHT(helper(FieldTags{}, fields, n_dot_fluxes));
  }

  // NOLINTNEXTLINE
  void pup(PUP::er& p) noexcept override {
    Base::pup(p);
    p | boundary_condition_types_;
  }

 private:
  tuples::TaggedTuple<elliptic::Tags::BoundaryConditionType<FieldTags>...>
      boundary_condition_types_{};
};

/// \cond
template <size_t Dim, typename... FieldTags, typename Registrars>
PUP::able::PUP_ID Zero<Dim, tmpl::list<FieldTags...>, Registrars>::my_PUP_ID =
    0;  // NOLINT
/// \endcond

}  // namespace elliptic::BoundaryConditions
