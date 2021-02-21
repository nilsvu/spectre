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

struct DerivativeVariables {
  static constexpr size_t Dim = 3;
  using Cache = CachedTempBuffer<
      DerivativeVariables,
      ::Tags::deriv<Tags::ConformalChristoffelSecondKind<DataVector, Dim,
                                                         Frame::Inertial>,
                    tmpl::size_t<Dim>, Frame::Inertial>,
      Tags::ConformalRicciTensor<DataVector, Dim, Frame::Inertial>,
      Tags::ConformalRicciScalar<DataVector>,
      ::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataVector>,
                    tmpl::size_t<Dim>, Frame::Inertial>,
      Tags::ShiftDotDerivExtrinsicCurvatureTrace<DataVector>>;

  const tnsr::I<DataVector, Dim>& x;
  const Mesh<Dim>& mesh;
  const InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>&
      inv_jacobian;
  const tnsr::II<DataVector, Dim>& inv_conformal_metric;
  const tnsr::Ijj<DataVector, Dim>& conformal_christoffel_second_kind;
  const Scalar<DataVector>& extrinsic_curvature_trace;
  const tnsr::I<DataVector, Dim>& shift;

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
  void operator()(
      gsl::not_null<tnsr::i<DataVector, Dim>*> deriv_extrinsic_curvature_trace,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataVector>,
                    tmpl::size_t<Dim>, Frame::Inertial> /*meta*/)
      const noexcept;
  void operator()(
      gsl::not_null<Scalar<DataVector>*>
          shift_dot_deriv_extrinsic_curvature_trace,
      gsl::not_null<Cache*> cache,
      Tags::ShiftDotDerivExtrinsicCurvatureTrace<DataVector> /*meta*/)
      const noexcept;
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
    return variables_from_tagged_tuple(
        call_with_dynamic_type<tuples::TaggedTuple<RequestedTags...>,
                               typename Base::creatable_classes>(
            this, [&x](auto* const derived) noexcept {
              return derived->variables(x, tmpl::list<RequestedTags...>{});
            }));
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
                 Tags::InverseConformalMetric<DataVector, Dim, Frame::Inertial>,
                 Tags::ConformalChristoffelSecondKind<DataVector, Dim,
                                                      Frame::Inertial>,
                 gr::Tags::TraceExtrinsicCurvature<DataVector>,
                 gr::Tags::Shift<Dim, Frame::Inertial, DataVector>>>{});
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
                  DataVector, Dim, Frame::Inertial>>(pointwise_vars),
              get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(
                  pointwise_vars),
              get<gr::Tags::Shift<Dim, Frame::Inertial, DataVector>>(
                  pointwise_vars)}};
      tmpl::for_each<derivative_tags>(
          [&vars, &derivative_vars_cache](auto tag_v) noexcept {
            using tag = tmpl::type_from<std::decay_t<decltype(tag_v)>>;
            get<tag>(vars) = derivative_vars_cache.get_var(tag{});
          });
      return vars;
    }
  }
};
}  // namespace AnalyticData
}  // namespace Xcts
