// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <pup.h>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Elasticity/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/ProductOfSinusoids.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

#include "Parallel/Printf.hpp"

namespace Elasticity::Solutions {

namespace detail {
template <typename ElasticityTag>
struct CorrespondingPoissonTag;
template <size_t Dim>
struct CorrespondingPoissonTag<Tags::Displacement<Dim>> {
  using type = Poisson::Tags::Field;
  static void elasticity_value(
      const gsl::not_null<tnsr::I<DataVector, Dim>*> displacement,
      std::array<Scalar<DataVector>, Dim> poisson_fields) noexcept {
    for (size_t i = 0; i < Dim; ++i) {
      displacement->get(i) = std::move(get(poisson_fields[i]));
    }
  }
};
template <size_t Dim>
struct CorrespondingPoissonTag<Tags::MinusStress<Dim>> {
  using type =
      ::Tags::Flux<Poisson::Tags::Field, tmpl::size_t<Dim>, Frame::Inertial>;
  static void elasticity_value(
      const gsl::not_null<tnsr::II<DataVector, Dim>*> minus_stress,
      std::array<tnsr::I<DataVector, Dim>, Dim> poisson_field_fluxes) noexcept {
    get<Dim - 1, Dim - 1>(*minus_stress) = -get<0>(poisson_field_fluxes[0]);
    for (size_t i = 1; i < Dim; ++i) {
      get<Dim - 1, Dim - 1>(*minus_stress) -= poisson_field_fluxes[i].get(i);
    }
    for (size_t i = 0; i < Dim; ++i) {
      minus_stress->get(i, i) = get<Dim - 1, Dim - 1>(*minus_stress) +
                                2. * poisson_field_fluxes[i].get(i);
      for (size_t j = 0; j < i; ++j) {
        minus_stress->get(i, j) =
            poisson_field_fluxes[j].get(i) + poisson_field_fluxes[i].get(j);
      }
    }
  }
};
template <size_t Dim>
struct CorrespondingPoissonTag<::Tags::FixedSource<Tags::Displacement<Dim>>> {
  using type = ::Tags::FixedSource<Poisson::Tags::Field>;
  static void elasticity_value(const gsl::not_null<tnsr::I<DataVector, Dim>*>
                                   fixed_source_for_displacement,
                               std::array<Scalar<DataVector>, Dim>
                                   fixed_source_for_poisson_fields) noexcept {
    for (size_t i = 0; i < Dim; ++i) {
      fixed_source_for_displacement->get(i) =
          std::move(get(fixed_source_for_poisson_fields[i]));
    }
  }
};
}  // namespace detail

/// \cond
template <size_t Dim, typename Registrars>
struct CoupledPoisson;

namespace Registrars {
template <size_t Dim>
struct CoupledPoisson {
  template <typename Registrars>
  using f = Solutions::CoupledPoisson<Dim, Registrars>;
};
}  // namespace Registrars
/// \endcond

template <size_t Dim,
          typename Registrars =
              tmpl::list<Solutions::Registrars::CoupledPoisson<Dim>>>
class CoupledPoisson : public AnalyticSolution<Dim, Registrars> {
 public:
  using constitutive_relation_type =
      Elasticity::ConstitutiveRelations::IsotropicHomogeneous<Dim>;
  using poisson_solution_registrars =
      tmpl::list<Poisson::Solutions::Registrars::ProductOfSinusoids<Dim>>;
  using PoissonSolutionType =
      Poisson::Solutions::AnalyticSolution<Dim, poisson_solution_registrars>;

  struct PoissonSolutions {
    using type = std::array<std::unique_ptr<PoissonSolutionType>, Dim>;
    static constexpr Options::String help{
        "A solution to the Poisson equation in each dimension. They are only "
        "coupled through boundary conditions."};
  };

  using options = tmpl::list<PoissonSolutions>;
  static constexpr Options::String help{
      "A set of Poisson solutions that are only coupled through boundary "
      "conditions."};

  CoupledPoisson() = default;
  CoupledPoisson(const CoupledPoisson&) noexcept = default;
  CoupledPoisson& operator=(const CoupledPoisson&) noexcept = default;
  CoupledPoisson(CoupledPoisson&&) noexcept = default;
  CoupledPoisson& operator=(CoupledPoisson&&) noexcept = default;
  ~CoupledPoisson() noexcept override = default;

  /// \cond
  explicit CoupledPoisson(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(CoupledPoisson);  // NOLINT
  /// \endcond

  CoupledPoisson(std::array<std::unique_ptr<PoissonSolutionType>, Dim>
                     poisson_solutions) noexcept
      : poisson_solutions_(std::move(poisson_solutions)) {
    if constexpr (Dim == 3) {
      constitutive_relation_ = constitutive_relation_type{-1. / 3., 1.};
    } else if constexpr (Dim == 2) {
      constitutive_relation_ = constitutive_relation_type{0., 1.};
    }
  }

  std::array<PoissonSolutionType, Dim> poisson_solutions() const noexcept {
    return poisson_solutions_;
  }

  const constitutive_relation_type& constitutive_relation()
      const noexcept override {
    return constitutive_relation_;
  }

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, Dim>& x,
      tmpl::list<RequestedTags...> /*meta*/) const noexcept {
    std::array<tuples::TaggedTuple<typename detail::CorrespondingPoissonTag<
                   RequestedTags>::type...>,
               Dim>
        poisson_vars{};
    for (size_t i = 0; i < Dim; ++i) {
      poisson_vars[i] = poisson_solutions_[i]->variables(
          x, tmpl::list<typename detail::CorrespondingPoissonTag<
                 RequestedTags>::type...>{});
    }
    tuples::TaggedTuple<RequestedTags...> elasticity_vars{};
    const auto helper = [&elasticity_vars,
                         &poisson_vars](auto elasticity_tag_v) {
      using elasticity_tag = std::decay_t<decltype(elasticity_tag_v)>;
      using poisson_tag =
          typename detail::CorrespondingPoissonTag<elasticity_tag>::type;
      std::array<typename poisson_tag::type, Dim> local_poisson_vars{};
      for (size_t i = 0; i < Dim; ++i) {
        local_poisson_vars[i] = std::move(get<poisson_tag>(poisson_vars[i]));
      }
      detail::CorrespondingPoissonTag<elasticity_tag>::elasticity_value(
          make_not_null(&get<elasticity_tag>(elasticity_vars)),
          std::move(local_poisson_vars));
    };
    EXPAND_PACK_LEFT_TO_RIGHT(helper(RequestedTags{}));
    return elasticity_vars;
  }

  void pup(PUP::er& p) noexcept override {
    p | constitutive_relation_;
    p | poisson_solutions_;
  }

 private:
  constitutive_relation_type constitutive_relation_{};
  std::array<std::unique_ptr<PoissonSolutionType>, Dim> poisson_solutions_{};
};

/// \cond
template <size_t Dim, typename Registrars>
PUP::able::PUP_ID CoupledPoisson<Dim, Registrars>::my_PUP_ID = 0;  // NOLINT
/// \endcond

}  // namespace Elasticity::Solutions
