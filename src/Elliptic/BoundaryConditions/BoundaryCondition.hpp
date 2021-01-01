// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>

#include "DataStructures/DataBox/ApplyAt.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/InterfaceHelpers.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/FakeVirtual.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace elliptic {
/// Boundary conditions for elliptic systems
namespace BoundaryConditions {

/// Note that almost all boundary conditions are actually nonlinear because even
/// those that depend only linearly on the dynamic fields typically contribute
/// non-zero field values. For example, a standard Dirichlet boundary condition
/// \f$u(x=0)=u_0\f$ is nonlinear for any \f$u_0\neq 0\f$.
template <size_t Dim, typename Registrars>
class BoundaryConditionBase : public PUP::able {
 public:
  static constexpr size_t volume_dim = Dim;
  using registrars = Registrars;

 protected:
  /// \cond
  BoundaryConditionBase() = default;
  BoundaryConditionBase(const BoundaryConditionBase&) = default;
  BoundaryConditionBase(BoundaryConditionBase&&) = default;
  BoundaryConditionBase& operator=(const BoundaryConditionBase&) = default;
  BoundaryConditionBase& operator=(BoundaryConditionBase&&) = default;
  /// \endcond

 public:
  ~BoundaryConditionBase() override = default;

  /// \cond
  explicit BoundaryConditionBase(CkMigrateMessage* m) noexcept : PUP::able(m) {}
  WRAPPED_PUPable_abstract(BoundaryConditionBase);  // NOLINT
  /// \endcond

  using creatable_classes = Registration::registrants<registrars>;

  template <typename DbTagsList, typename... FieldsAndFluxes>
  void operator()(
      const db::DataBox<DbTagsList>& box, const Direction<Dim>& direction,
      const gsl::not_null<FieldsAndFluxes*>... fields_and_fluxes) const
      noexcept {
    call_with_dynamic_type<void, creatable_classes>(
        this, [&box, &direction,
               &fields_and_fluxes...](auto* const boundary_condition) noexcept {
          using Derived =
              std::decay_t<std::remove_pointer_t<decltype(boundary_condition)>>;
          db::apply_at<
              tmpl::transform<
                  typename Derived::argument_tags,
                  make_interface_tag<
                      tmpl::_1,
                      tmpl::pin<domain::Tags::BoundaryDirectionsExterior<Dim>>,
                      tmpl::pin<get_volume_tags<Derived>>>>,
              get_volume_tags<Derived>>(*boundary_condition, box, direction,
                                        fields_and_fluxes...);
        });
  }
};

template <size_t Dim, typename Registrars>
class BoundaryCondition : public BoundaryConditionBase<Dim, Registrars> {
 private:
  using Base = BoundaryConditionBase<Dim, Registrars>;

  template <typename T>
  struct get_linearization_registrar {
    using type = typename T::linearization_registrar;
  };

 public:
  using linearization_registrars = tmpl::remove_duplicates<
      tmpl::transform<Registrars, get_linearization_registrar<tmpl::_1>>>;
  using linearization_type =
      BoundaryConditionBase<Dim, linearization_registrars>;

 protected:
  /// \cond
  BoundaryCondition() = default;
  BoundaryCondition(const BoundaryCondition&) = default;
  BoundaryCondition(BoundaryCondition&&) = default;
  BoundaryCondition& operator=(const BoundaryCondition&) = default;
  BoundaryCondition& operator=(BoundaryCondition&&) = default;
  /// \endcond

 public:
  ~BoundaryCondition() override = default;

  /// \cond
  explicit BoundaryCondition(CkMigrateMessage* m) noexcept : Base(m) {}
  WRAPPED_PUPable_abstract(BoundaryCondition);  // NOLINT
  /// \endcond

  virtual std::unique_ptr<linearization_type> linearization() const
      noexcept = 0;
};

}  // namespace BoundaryConditions
}  // namespace elliptic
