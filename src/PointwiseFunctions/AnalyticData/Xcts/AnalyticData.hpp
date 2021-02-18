// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "Utilities/FakeVirtual.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Xcts {
/// Analytic data for the XCTS system
namespace AnalyticData {

namespace detail {

struct DerivedVariables {
  static constexpr size_t Dim = 3;
  using Cache = CachedTempBuffer<
      DerivedVariables,
      Tags::InverseConformalMetric<DataVector, Dim, Frame::Inertial>,
      Tags::ConformalChristoffelFirstKind<DataVector, Dim, Frame::Inertial>,
      Tags::ConformalChristoffelSecondKind<DataVector, Dim, Frame::Inertial>,
      Tags::ConformalChristoffelContracted<DataVector, Dim, Frame::Inertial>>;

  const tnsr::I<DataVector, Dim>& x;
  const tnsr::ii<DataVector, Dim>& conformal_metric;
  const tnsr::ijj<DataVector, Dim>& deriv_conformal_metric;

  void operator()(
      gsl::not_null<tnsr::II<DataVector, Dim>*> inv_conformal_metric,
      gsl::not_null<Cache*> cache,
      Tags::InverseConformalMetric<DataVector, Dim, Frame::Inertial> /*meta*/)
      const noexcept;
  void operator()(gsl::not_null<tnsr::ijj<DataVector, Dim>*>
                      conformal_christoffel_first_kind,
                  gsl::not_null<Cache*> cache,
                  Tags::ConformalChristoffelFirstKind<DataVector, Dim,
                                                      Frame::Inertial> /*meta*/)
      const noexcept;
  void operator()(
      gsl::not_null<tnsr::Ijj<DataVector, Dim>*>
          conformal_christoffel_second_kind,
      gsl::not_null<Cache*> cache,
      Tags::ConformalChristoffelSecondKind<
          DataVector, Dim, Frame::Inertial> /*meta*/) const noexcept;
  void operator()(
      gsl::not_null<tnsr::i<DataVector, Dim>*> conformal_christoffel_contracted,
      gsl::not_null<Cache*> cache,
      Tags::ConformalChristoffelContracted<
          DataVector, Dim, Frame::Inertial> /*meta*/) const noexcept;
};

struct DerivativeVariables {
  static constexpr size_t Dim = 3;
  using Cache = CachedTempBuffer<
      DerivativeVariables,
      ::Tags::deriv<Tags::ConformalChristoffelSecondKind<DataVector, Dim,
                                                         Frame::Inertial>,
                    tmpl::size_t<Dim>, Frame::Inertial>,
      Tags::ConformalRicciTensor<DataVector, Dim, Frame::Inertial>,
      Tags::ConformalRicciScalar<DataVector>>;

  const tnsr::I<DataVector, Dim>& x;
  const Mesh<Dim>& mesh;
  const InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>&
      inv_jacobian;
  const tnsr::II<DataVector, Dim>& inv_conformal_metric;
  const tnsr::Ijj<DataVector, Dim>& conformal_christoffel_second_kind;

  void operator()(gsl::not_null<tnsr::iJkk<DataVector, Dim>*>
                      deriv_conformal_christoffel_second_kind,
                  gsl::not_null<Cache*> cache,
                  ::Tags::deriv<Tags::ConformalChristoffelSecondKind<
                                    DataVector, Dim, Frame::Inertial>,
                                tmpl::size_t<Dim>, Frame::Inertial> /*meta*/)
      const noexcept;
  void operator()(
      gsl::not_null<tnsr::ii<DataVector, Dim>*> conformal_ricci_tensor,
      gsl::not_null<Cache*> cache,
      Tags::ConformalRicciTensor<DataVector, Dim, Frame::Inertial> /*meta*/)
      const noexcept;
  void operator()(
      gsl::not_null<Scalar<DataVector>*> conformal_ricci_scalar,
      gsl::not_null<Cache*> cache,
      Tags::ConformalRicciScalar<DataVector> /*meta*/) const noexcept;
};

}  // namespace detail

template <typename Registrars>
class AnalyticData : public ::AnalyticData<3, Registrars> {
 private:
  using Base = ::AnalyticData<3, Registrars>;
  static constexpr size_t Dim = 3;

 protected:
  /// \cond
  AnalyticData() = default;
  AnalyticData(const AnalyticData&) = default;
  AnalyticData(AnalyticData&&) = default;
  AnalyticData& operator=(const AnalyticData&) = default;
  AnalyticData& operator=(AnalyticData&&) = default;
  /// \endcond

 public:
  ~AnalyticData() override = default;

  /// \cond
  explicit AnalyticData(CkMigrateMessage* m) noexcept : Base(m) {}
  WRAPPED_PUPable_abstract(AnalyticData);  // NOLINT
  /// \endcond

  template <typename DataType, typename... RequestedTags>
  Variables<tmpl::list<RequestedTags...>> variables(
      const tnsr::I<DataType, Dim>& x,
      tmpl::list<RequestedTags...> /*meta*/) const noexcept {
    using DerivedVarsComputer = detail::DerivedVariables;
    using original_tags =
        tmpl::list_difference<tmpl::list<RequestedTags...>,
                              typename DerivedVarsComputer::Cache::tags_list>;
    using derived_tags =
        tmpl::list_difference<tmpl::list<RequestedTags...>, original_tags>;
    if constexpr (std::is_same_v<derived_tags, tmpl::list<>>) {
      return variables_from_tagged_tuple(
          this->original_variables(x, original_tags{}));
    } else {
      auto vars =
          make_with_value<Variables<tmpl::list<RequestedTags...>>>(x, 0.);
      // Retrieve original data from the derived class, including dependencies
      // for the derived data
      auto original_vars = this->original_variables(
          x, tmpl::remove_duplicates<tmpl::push_back<
                 original_tags,
                 Tags::ConformalMetric<DataVector, Dim, Frame::Inertial>,
                 ::Tags::deriv<
                     Tags::ConformalMetric<DataVector, Dim, Frame::Inertial>,
                     tmpl::size_t<Dim>, Frame::Inertial>>>{});
      tmpl::for_each<original_tags>(
          [&vars, &original_vars](auto tag_v) noexcept {
            using tag = tmpl::type_from<std::decay_t<decltype(tag_v)>>;
            get<tag>(vars) = get<tag>(original_vars);
          });
      // Fill in derived data
      typename DerivedVarsComputer::Cache derived_vars_cache{
          vars.number_of_grid_points(),
          DerivedVarsComputer{
              x,
              get<Tags::ConformalMetric<DataVector, Dim, Frame::Inertial>>(
                  original_vars),
              get<::Tags::deriv<
                  Tags::ConformalMetric<DataVector, Dim, Frame::Inertial>,
                  tmpl::size_t<Dim>, Frame::Inertial>>(original_vars)}};
      tmpl::for_each<derived_tags>(
          [&vars, &derived_vars_cache](auto tag_v) noexcept {
            using tag = tmpl::type_from<std::decay_t<decltype(tag_v)>>;
            get<tag>(vars) = derived_vars_cache.get_var(tag{});
          });
      return vars;
    }
  }

  template <typename... RequestedTags>
  Variables<tmpl::list<RequestedTags...>> variables(
      const tnsr::I<DataVector, Dim>& x, const Mesh<Dim>& mesh,
      const InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>&
          inv_jacobian,
      tmpl::list<RequestedTags...> /*meta*/) const noexcept {
    using DerivativeVarsComputer = detail::DerivativeVariables;
    using pointwise_tags = tmpl::list_difference<
        tmpl::list<RequestedTags...>,
        typename DerivativeVarsComputer::Cache::tags_list>;
    using derivative_tags =
        tmpl::list_difference<tmpl::list<RequestedTags...>, pointwise_tags>;
    if constexpr (std::is_same_v<derivative_tags, tmpl::list<>>) {
      (void)mesh;
      (void)inv_jacobian;
      return this->variables(x, pointwise_tags{});
    } else {
      const size_t num_points = mesh.number_of_grid_points();
      Variables<tmpl::list<RequestedTags...>> vars{num_points};
      // Retrieve pointwise data from the derived class, including dependencies
      // for the derived data
      auto pointwise_vars = this->variables(
          x, tmpl::remove_duplicates<tmpl::push_back<
                 pointwise_tags,
                 Tags::ConformalMetric<DataVector, Dim, Frame::Inertial>,
                 ::Tags::deriv<
                     Tags::ConformalMetric<DataVector, Dim, Frame::Inertial>,
                     tmpl::size_t<Dim>, Frame::Inertial>>>{});
      tmpl::for_each<pointwise_tags>(
          [&vars, &pointwise_vars](auto tag_v) noexcept {
            using tag = tmpl::type_from<std::decay_t<decltype(tag_v)>>;
            get<tag>(vars) = get<tag>(pointwise_vars);
          });
      // Fill in derivative data
      typename DerivativeVarsComputer::Cache derivative_vars_cache{
          num_points,
          DerivativeVarsComputer{
              x, mesh, inv_jacobian,
              get<Tags::InverseConformalMetric<
                  DataVector, Dim, Frame::Inertial>>(pointwise_vars),
              get<Tags::ConformalChristoffelSecondKind<
                  DataVector, Dim, Frame::Inertial>>(pointwise_vars)}};
      tmpl::for_each<derivative_tags>(
          [&vars, &derivative_vars_cache](auto tag_v) noexcept {
            using tag = tmpl::type_from<std::decay_t<decltype(tag_v)>>;
            get<tag>(vars) = derivative_vars_cache.get_var(tag{});
          });
      return vars;
    }
  }

 private:
  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> original_variables(
      const tnsr::I<DataType, Dim>& x,
      tmpl::list<RequestedTags...> /*meta*/) const noexcept {
    return call_with_dynamic_type<tuples::TaggedTuple<RequestedTags...>,
                                  typename Base::creatable_classes>(
        this, [&x](auto* const derived) noexcept {
          return derived->variables(x, tmpl::list<RequestedTags...>{});
        });
  }
};
}  // namespace AnalyticData
}  // namespace Xcts
