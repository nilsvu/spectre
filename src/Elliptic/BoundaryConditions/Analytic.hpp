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
#include "Elliptic/BoundaryConditions/Zero.hpp"
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
template <size_t Dim, typename ProviderTag, typename FieldTags,
          typename Registrars>
struct Analytic;

namespace Registrars {
template <size_t Dim, typename ProviderTag, typename FieldTags>
struct Analytic {
  template <typename Registrars>
  using f =
      BoundaryConditions::Analytic<Dim, ProviderTag, FieldTags, Registrars>;
};
}  // namespace Registrars
/// \endcond

template <size_t Dim, typename ProviderTag, typename... FieldTags,
          typename Registrars>
class Analytic<Dim, ProviderTag, tmpl::list<FieldTags...>, Registrars>
    : public BoundaryCondition<Dim, Registrars> {
 private:
  using Base = BoundaryCondition<Dim, Registrars>;

  template <typename Tag>
  struct BoundaryConditionType {
    static std::string name() noexcept { return db::tag_name<Tag>(); }
    using type = elliptic::BoundaryConditionType;
  };

 public:
  static std::string name() noexcept { return db::tag_name<ProviderTag>(); }
  using options = tmpl::list<BoundaryConditionType<FieldTags>...>;
  static constexpr Options::String help{
      "Boundary conditions from an analytic solution or analytic data"};

  Analytic() = default;
  Analytic(const Analytic&) noexcept = default;
  Analytic& operator=(const Analytic&) noexcept = default;
  Analytic(Analytic&&) noexcept = default;
  Analytic& operator=(Analytic&&) noexcept = default;
  ~Analytic() noexcept = default;

  explicit Analytic(const typename BoundaryConditionType<
                    FieldTags>::type... boundary_condition_types) noexcept
      : boundary_condition_types_{boundary_condition_types...} {}

  // The linearization is always zero since the boundary conditions are
  // independent of the dynamic fields
  using linearized_registrar =
      BoundaryConditions::Registrars::Zero<Dim, tmpl::list<FieldTags...>>;

  const BoundaryConditionBase<Dim, typename Base::linearization_registrars>&
  linearization() const noexcept {
    return linearization_;
  }

  const tuples::TaggedTuple<
      elliptic::Tags::BoundaryConditionType<FieldTags>...>&
  boundary_condition_types() const noexcept {
    return boundary_condition_types_;
  }

  // Argument tags are taken from exterior faces, i.e. the face normal points
  // into the domain.
  using argument_tags =
      tmpl::list<ProviderTag, domain::Tags::Coordinates<Dim, Frame::Inertial>,
                 domain::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<
                     Dim, Frame::Inertial>>>;
  using volume_tags = tmpl::list<ProviderTag>;

  template <typename Provider>
  void operator()(
      const Provider& provider, const tnsr::I<DataVector, Dim>& x,
      const tnsr::i<DataVector, Dim>& face_normal,
      const gsl::not_null<typename FieldTags::type*>... fields,
      const gsl::not_null<typename FieldTags::type*>... n_dot_fluxes)
      const noexcept {
    auto analytic_fields_and_fluxes = provider.variables(
        x, tmpl::list<FieldTags..., ::Tags::Flux<FieldTags, tmpl::size_t<Dim>,
                                                 Frame::Inertial>>{});
    const auto helper = [&analytic_fields_and_fluxes, &face_normal](
                            auto field_tag_v, const auto field,
                            const auto n_dot_flux) noexcept {
      using field_tag = decltype(field_tag_v);
      switch (get<field_tag>(boundary_condition_types_)) {
        case elliptic::BoundaryConditions::Dirichlet:
          *field = std::move(get<field_tag>(analytic_fields_and_fluxes));
          break;
        case elliptic::BoundaryConditions::Neumann:
          normal_dot_flux(
              n_dot_flux, face_normal,
              std::move(get<::Tags::Flux<field_tag, tmpl::size_t<Dim>,
                                         Frame::Inertial>>(
                  analytic_fields_and_fluxes)));
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
  elliptic::BoundaryConditions::Zero<Dim, tmpl::list<FieldTags...>,
                                     typename Base::linearization_registrars>
      linearization_;
};

}  // namespace elliptic::BoundaryConditions
