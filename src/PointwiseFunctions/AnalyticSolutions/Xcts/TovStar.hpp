// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <ostream>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/TovStar.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/CommonVariables.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags/Conformal.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Xcts::Solutions {
namespace tov_detail {

using TovCoordinates = gr::Solutions::TovCoordinates;

template <typename DataType>
using TovVariablesCache = cached_temp_buffer_from_typelist<tmpl::push_back<
    common_tags<DataType>,
    ::Tags::deriv<gr::Tags::Lapse<DataType>, tmpl::size_t<3>, Frame::Inertial>,
    gr::Tags::Conformal<gr::Tags::EnergyDensity<DataType>, 0>,
    gr::Tags::Conformal<gr::Tags::StressTrace<DataType>, 0>,
    gr::Tags::Conformal<gr::Tags::MomentumDensity<3, Frame::Inertial, DataType>,
                        0>>>;

template <typename DataType>
struct TovVariables : CommonVariables<DataType, TovVariablesCache<DataType>> {
  static constexpr size_t Dim = 3;
  static constexpr int ConformalMatterScale = 0;
  using Cache = TovVariablesCache<DataType>;
  using Base = CommonVariables<DataType, TovVariablesCache<DataType>>;
  using Base::operator();

  const tnsr::I<DataType, 3>& x;
  const DataType& radius;
  const RelativisticEuler::Solutions::TovStar& tov_star;

  TovVariables(
      std::optional<std::reference_wrapper<const Mesh<Dim>>> local_mesh,
      std::optional<std::reference_wrapper<const InverseJacobian<
          DataVector, Dim, Frame::ElementLogical, Frame::Inertial>>>
          local_inv_jacobian,
      const tnsr::I<DataType, 3>& local_x, const DataType& local_radius,
      const RelativisticEuler::Solutions::TovStar& local_tov_star)
      : Base(std::move(local_mesh), std::move(local_inv_jacobian)),
        x(local_x),
        radius(local_radius),
        tov_star(local_tov_star) {}

  void operator()(gsl::not_null<tnsr::ii<DataType, 3>*> conformal_metric,
                  gsl::not_null<Cache*> cache,
                  Tags::ConformalMetric<DataType, 3, Frame::Inertial> /*meta*/)
      const override;
  void operator()(
      gsl::not_null<tnsr::II<DataType, 3>*> inv_conformal_metric,
      gsl::not_null<Cache*> cache,
      Tags::InverseConformalMetric<DataType, 3, Frame::Inertial> /*meta*/)
      const override;
  void operator()(
      gsl::not_null<tnsr::ijj<DataType, 3>*> deriv_conformal_metric,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<Tags::ConformalMetric<DataType, 3, Frame::Inertial>,
                    tmpl::size_t<3>, Frame::Inertial> /*meta*/) const override;
  void operator()(
      gsl::not_null<tnsr::ii<DataType, 3>*> extrinsic_curvature,
      gsl::not_null<Cache*> cache,
      gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataType> /*meta*/)
      const override;
  void operator()(
      gsl::not_null<Scalar<DataType>*> trace_extrinsic_curvature,
      gsl::not_null<Cache*> cache,
      gr::Tags::TraceExtrinsicCurvature<DataType> /*meta*/) const override;
  void operator()(
      gsl::not_null<tnsr::i<DataType, 3>*> deriv_trace_extrinsic_curvature,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataType>,
                    tmpl::size_t<3>, Frame::Inertial> /*meta*/) const override;
  void operator()(
      gsl::not_null<Scalar<DataType>*> dt_trace_extrinsic_curvature,
      gsl::not_null<Cache*> cache,
      ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>> /*meta*/)
      const override;
  void operator()(gsl::not_null<Scalar<DataType>*> conformal_factor,
                  gsl::not_null<Cache*> cache,
                  Tags::ConformalFactor<DataType> /*meta*/) const override;
  void operator()(
      gsl::not_null<tnsr::i<DataType, 3>*> deriv_conformal_factor,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>, tmpl::size_t<3>,
                    Frame::Inertial> /*meta*/) const override;
  void operator()(gsl::not_null<Scalar<DataType>*> lapse,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::Lapse<DataType> /*meta*/) const override;
  void operator()(gsl::not_null<tnsr::i<DataType, 3>*> deriv_lapse,
                  gsl::not_null<Cache*> cache,
                  ::Tags::deriv<gr::Tags::Lapse<DataType>, tmpl::size_t<3>,
                                Frame::Inertial> /*meta*/) const;
  void operator()(
      gsl::not_null<Scalar<DataType>*> lapse_times_conformal_factor,
      gsl::not_null<Cache*> cache,
      Tags::LapseTimesConformalFactor<DataType> /*meta*/) const override;
  void operator()(
      gsl::not_null<tnsr::i<DataType, 3>*> deriv_lapse_times_conformal_factor,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<Tags::LapseTimesConformalFactor<DataType>, tmpl::size_t<3>,
                    Frame::Inertial> /*meta*/) const override;
  void operator()(gsl::not_null<tnsr::I<DataType, 3>*> shift_background,
                  gsl::not_null<Cache*> cache,
                  Tags::ShiftBackground<DataType, 3, Frame::Inertial> /*meta*/)
      const override;
  void operator()(gsl::not_null<tnsr::II<DataType, 3, Frame::Inertial>*>
                      longitudinal_shift_background_minus_dt_conformal_metric,
                  gsl::not_null<Cache*> cache,
                  Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
                      DataType, 3, Frame::Inertial> /*meta*/) const override;
  void operator()(
      gsl::not_null<tnsr::I<DataType, 3>*> shift_excess,
      gsl::not_null<Cache*> cache,
      Tags::ShiftExcess<DataType, 3, Frame::Inertial> /*meta*/) const override;
  void operator()(
      gsl::not_null<tnsr::ii<DataType, 3>*> shift_strain,
      gsl::not_null<Cache*> cache,
      Tags::ShiftStrain<DataType, 3, Frame::Inertial> /*meta*/) const override;
  void operator()(gsl::not_null<Scalar<DataType>*> energy_density,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::Conformal<gr::Tags::EnergyDensity<DataType>,
                                      ConformalMatterScale> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> stress_trace,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::Conformal<gr::Tags::StressTrace<DataType>,
                                      ConformalMatterScale> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::I<DataType, 3>*> momentum_density,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::Conformal<
                      gr::Tags::MomentumDensity<3, Frame::Inertial, DataType>,
                      ConformalMatterScale> /*meta*/) const;

 private:
  template <typename Tag>
  typename Tag::type get_tov_var(Tag /*meta*/) const {
    // Possible optimization: Access the cache of the RelEuler::TovStar solution
    // so its intermediate quantities don't have to be re-computed repeatedly
    return get<Tag>(tov_star.variables(
        x, std::numeric_limits<double>::signaling_NaN(), tmpl::list<Tag>{}));
  }
};

}  // namespace tov_detail

/*!
 * \brief TOV solution to the XCTS equations
 *
 * \see RelativisticEuler::Solutions::TovStar
 * \see gr::Solutions::TovSolution
 */
class TovStar : public elliptic::analytic_data::AnalyticSolution,
                private RelativisticEuler::Solutions::TovStar {
 private:
  using RelEulerTovStar = RelativisticEuler::Solutions::TovStar;

 public:
  using options = RelEulerTovStar::options;
  static constexpr Options::String help = RelEulerTovStar::help;

  TovStar() = default;
  TovStar(const TovStar&) = default;
  TovStar& operator=(const TovStar&) = default;
  TovStar(TovStar&&) = default;
  TovStar& operator=(TovStar&&) = default;
  ~TovStar() = default;

  using RelEulerTovStar::radial_solution;
  using RelEulerTovStar::TovStar;

  /// \cond
  explicit TovStar(CkMigrateMessage* m)
      : elliptic::analytic_data::AnalyticSolution(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(TovStar);
  std::unique_ptr<elliptic::analytic_data::AnalyticSolution> get_clone()
      const override {
    return std::make_unique<TovStar>(*this);
  }
  /// \endcond

  template <typename DataType>
  using tags = typename tov_detail::TovVariablesCache<DataType>::tags_list;

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<RequestedTags...> /*meta*/) const {
    using VarsComputer = tov_detail::TovVariables<DataType>;
    typename VarsComputer::Cache cache{get_size(*x.begin())};
    const DataType radius = get(magnitude(x));
    const VarsComputer computer{std::nullopt, std::nullopt, x, radius, *this};
    return {cache.get_var(computer, RequestedTags{})...};
  }

  template <typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x, const Mesh<3>& mesh,
      const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                            Frame::Inertial>& inv_jacobian,
      tmpl::list<RequestedTags...> /*meta*/) const {
    using VarsComputer = tov_detail::TovVariables<DataVector>;
    typename VarsComputer::Cache cache{get_size(*x.begin())};
    const DataVector radius = get(magnitude(x));
    const VarsComputer computer{mesh, inv_jacobian, x, radius, *this};
    return {cache.get_var(computer, RequestedTags{})...};
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override {
    elliptic::analytic_data::AnalyticSolution::pup(p);
    RelEulerTovStar::pup(p);
  }
};

}  // namespace Xcts::Solutions
