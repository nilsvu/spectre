// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <pup.h>
#include <tuple>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/String.hpp"
#include "ParallelAlgorithms/RayTracer/BackgroundSpacetimes/BackgroundSpacetime.hpp"
#include "PointwiseFunctions/GeneralRelativity/DerivativeSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TupleSlice.hpp"

namespace ray_tracing {

/// Analytic background spacetime from a GR or GRMHD solution.
template <typename SolutionType>
class WrappedGr : public BackgroundSpacetime {
 public:
  WrappedGr() = default;
  WrappedGr(const WrappedGr& /*rhs*/) = default;
  WrappedGr& operator=(const WrappedGr& /*rhs*/) = default;
  WrappedGr(WrappedGr&& /*rhs*/) = default;
  WrappedGr& operator=(WrappedGr&& /*rhs*/) = default;
  ~WrappedGr() override = default;

  template <typename Arg1, typename Arg2, typename... Args>
  WrappedGr(Arg1&& /*arg1*/, Arg2&& arg2, Args&&... args) {
    // Some gymnastics so this works with option parsing: skip the first
    // argument (ParseOptions) and last two arguments (context, Metavars)
    wrapped_solution_ = std::apply(
        [](auto&&... forwarded_args) {
          return SolutionType(
              std::forward<decltype(forwarded_args)>(forwarded_args)...);
        },
        tuple_slice<0, sizeof...(Args) - 2>(std::forward_as_tuple(
            std::forward<Arg2>(arg2), std::forward<Args>(args)...)));
  }

  WrappedGr(SolutionType wrapped_solution)
      : wrapped_solution_(std::move(wrapped_solution)) {}

  auto get_clone() const -> std::unique_ptr<BackgroundSpacetime> override {
    return std::make_unique<WrappedGr<SolutionType>>(*this);
  }

  /// \cond
  explicit WrappedGr(CkMigrateMessage* msg) : BackgroundSpacetime(msg) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(WrappedGr);
  /// \endcond

  using options = typename SolutionType::options;
  static constexpr Options::String help = SolutionType::help;
  static std::string name() { return pretty_type::name<SolutionType>(); }

  tuples::tagged_tuple_from_typelist<tags> variables(
      const tnsr::I<DataType, Dim, Frame>& x, double t,
      const std::optional<gsl::not_null<std::vector<size_t>*>> /*block_order*/ =
          std::nullopt) const override {
    // Tags that the solution can provide
    using available_tags = typename SolutionType::template tags<DataType>;
    static constexpr bool has_hydro =
        tmpl::list_contains_v<available_tags,
                              hydro::Tags::RestMassDensity<DataType>>;
    // Tags that we retrieve from the solution. Includes hydro tags if
    // available. Gets the deriv(SpatialMetric) instead of
    // deriv(InvSpatialMetric) because the latter isn't available.
    using retrieve_tags = tmpl::push_back<
        tmpl::remove<
            tmpl::conditional_t<
                has_hydro, tags,
                tmpl::remove<tags, hydro::Tags::RestMassDensity<DataType>>>,
            DerivInvSpatialMetric>,
        DerivSpatialMetric>;
    tuples::tagged_tuple_from_typelist<tags> result{};
    const auto intermediate_vars =
        wrapped_solution_.variables(x, t, retrieve_tags{});
    // Compute the deriv(InvSpatialMetric) from the deriv(SpatialMetric)
    tmpl::for_each<tmpl::remove<retrieve_tags, DerivSpatialMetric>>(
        [&result, &intermediate_vars](auto tag_v) {
          using tag = tmpl::type_from<decltype(tag_v)>;
          get<tag>(result) = get<tag>(intermediate_vars);
        });
    get<DerivInvSpatialMetric>(result) = gr::deriv_inverse_spatial_metric(
        get<gr::Tags::InverseSpatialMetric<DataType, Dim, Frame>>(
            intermediate_vars),
        get<DerivSpatialMetric>(intermediate_vars));
    return result;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override {
    BackgroundSpacetime::pup(p);
    p | wrapped_solution_;
  }

 private:
  SolutionType wrapped_solution_;
};

template <typename SolutionType>
PUP::able::PUP_ID WrappedGr<SolutionType>::my_PUP_ID = 0;

}  // namespace ray_tracing
