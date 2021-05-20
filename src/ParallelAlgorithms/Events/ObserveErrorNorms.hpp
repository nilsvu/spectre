// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <optional>
#include <pup.h>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Tags.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"  // IWYU pragma: keep
#include "IO/Observer/ReductionActions.hpp"   // IWYU pragma: keep
#include "IO/Observer/TypeOfObservation.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "Options/Options.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

namespace dg {
namespace Events {
template <size_t Dim, typename ObservationValueTag, typename Tensors,
          typename ArraySectionIdTag, typename EventRegistrars>
class ObserveErrorNorms;

namespace Registrars {
template <size_t Dim, typename ObservationValueTag, typename Tensors,
          typename ArraySectionIdTag = void>
struct ObserveErrorNorms {
  template <typename RegistrarList>
  using f = Events::ObserveErrorNorms<Dim, ObservationValueTag, Tensors,
                                      ArraySectionIdTag, RegistrarList>;
};
}  // namespace Registrars

template <size_t Dim, typename ObservationValueTag, typename Tensors,
          typename ArraySectionIdTag = void,
          typename EventRegistrars = tmpl::list<Registrars::ObserveErrorNorms<
              Dim, ObservationValueTag, Tensors, ArraySectionIdTag>>>
class ObserveErrorNorms;  // IWYU pragma: keep

/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief %Observe the RMS errors in the tensors compared to their
 * analytic solution.
 *
 * Writes reduction quantities:
 * - `ObservationValueTag`
 * - `NumberOfPoints` = total number of points in the domain
 * - `Error(*)` = RMS errors in `Tensors` =
 *   \f$\operatorname{RMS}\left(\sqrt{\sum_{\text{independent components}}\left[
 *   \text{value} - \text{analytic solution}\right]^2}\right)\f$
 *   over all points
 */
template <size_t Dim, typename ObservationValueTag, typename... Tensors,
          typename ArraySectionIdTag, typename EventRegistrars>
class ObserveErrorNorms<Dim, ObservationValueTag, tmpl::list<Tensors...>,
                        ArraySectionIdTag, EventRegistrars>
    : public Event<EventRegistrars> {
 private:
  template <typename Tag>
  struct LocalRmsSquareError {
    using type = double;
  };
  template <typename Tag>
  struct LocalLinfError {
    using type = double;
  };
  template <typename Tag>
  struct LocalL2SquareError {
    using type = double;
  };

  using RmsErrorDatum = Parallel::ReductionDatum<double, funcl::Plus<>,
                                                 funcl::Sqrt<funcl::Divides<>>,
                                                 std::index_sequence<1>>;
  using LinfErrorDatum = Parallel::ReductionDatum<double, funcl::Max<>>;
  using L2ErrorDatum = Parallel::ReductionDatum<double, funcl::Plus<>,
                                                funcl::Sqrt<funcl::Divides<>>,
                                                std::index_sequence<2>>;
  using ReductionData = tmpl::wrap<
      tmpl::append<tmpl::list<
                       // Observation value
                       Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
                       // Number of points
                       Parallel::ReductionDatum<size_t, funcl::Plus<>>,
                       // Volume
                       Parallel::ReductionDatum<double, funcl::Plus<>>>,
                   tmpl::filled_list<RmsErrorDatum, sizeof...(Tensors)>,
                   tmpl::filled_list<LinfErrorDatum, sizeof...(Tensors)>,
                   tmpl::filled_list<L2ErrorDatum, sizeof...(Tensors)>>,
      Parallel::ReductionData>;

 public:
  /// The name of the subfile inside the HDF5 file
  struct SubfileName {
    using type = std::string;
    static constexpr Options::String help = {
        "The name of the subfile inside the HDF5 file without an extension and "
        "without a preceding '/'."};
  };

  /// \cond
  explicit ObserveErrorNorms(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ObserveErrorNorms);  // NOLINT
  /// \endcond

  using options = tmpl::list<SubfileName>;
  static constexpr Options::String help =
      "Observe the RMS errors in the tensors compared to their analytic\n"
      "solution (if one is available).\n"
      "\n"
      "Writes reduction quantities:\n"
      " * ObservationValueTag\n"
      " * NumberOfPoints = total number of points in the domain\n"
      " * Error(*) = RMS errors in Tensors (see online help details)\n"
      "\n"
      "Warning: Currently, only one reduction observation event can be\n"
      "triggered at a given observation value.  Causing multiple events to\n"
      "run at once will produce unpredictable results.";

  ObserveErrorNorms() = default;
  explicit ObserveErrorNorms(const std::string& subfile_name) noexcept;

  using observed_reduction_data_tags =
      observers::make_reduction_data_tags<tmpl::list<ReductionData>>;

  using argument_tags = tmpl::flatten<
      tmpl::list<ObservationValueTag,
                 tmpl::conditional_t<
                     std::is_same_v<ArraySectionIdTag, void>, tmpl::list<>,
                     observers::Tags::ObservationKeySuffix<ArraySectionIdTag>>,
                 domain::Tags::Mesh<Dim>,
                 domain::Tags::DetInvJacobian<Frame::Logical, Frame::Inertial>,
                 Tensors..., ::Tags::AnalyticSolutionsBase>>;

  template <typename OptionalAnalyticSolutions, typename Metavariables,
            typename ArrayIndex, typename ParallelComponent>
  void operator()(const typename ObservationValueTag::type& observation_value,
                  const std::optional<std::string>& observation_key_suffix,
                  const Mesh<Dim>& mesh,
                  const Scalar<DataVector>& det_inv_jacobian,
                  const typename Tensors::type&... tensors,
                  const OptionalAnalyticSolutions& optional_analytic_solutions,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ArrayIndex& array_index,
                  const ParallelComponent* const /*meta*/) const noexcept {
    if constexpr (tt::is_a_v<std::optional, OptionalAnalyticSolutions>) {
      if (not optional_analytic_solutions.has_value()) {
        // Nothing to do if we don't have analytic solutions. We may generalize
        // this event to observe norms of other quantities.
        return;
      }
    }
    const auto& analytic_solutions =
        [&optional_analytic_solutions]() noexcept -> decltype(auto) {
      // If we generalize this event to do observations of non-solution
      // quantities then we can return a std::optional<std::reference_wrapper>
      // here (using std::cref).
      if constexpr (tt::is_a_v<std::optional, OptionalAnalyticSolutions>) {
        return *optional_analytic_solutions;
      } else {
        return optional_analytic_solutions;
      }
    }();

    const double local_volume =
        definite_integral(1. / get(det_inv_jacobian), mesh);
    tuples::TaggedTuple<LocalRmsSquareError<Tensors>...,
                        LocalLinfError<Tensors>...,
                        LocalL2SquareError<Tensors>...>
        local_errors;
    const auto record_errors = [&local_errors, &analytic_solutions, &mesh,
                                &det_inv_jacobian](
                                   const auto tensor_tag_v,
                                   const auto& tensor) noexcept {
      using tensor_tag = tmpl::type_from<decltype(tensor_tag_v)>;
      double local_rms_square_error = 0.0;
      double local_linf_error = 0.0;
      double local_l2_square_error = 0.0;
      for (size_t i = 0; i < tensor.size(); ++i) {
        const DataVector error = tensor[i] - get<::Tags::Analytic<tensor_tag>>(
                                                 analytic_solutions)[i];
        local_rms_square_error += alg::accumulate(square(error), 0.0);
        local_linf_error = std::max(max(abs(error)), local_linf_error);
        local_l2_square_error +=
            definite_integral(square(error) / get(det_inv_jacobian), mesh);
      }
      get<LocalRmsSquareError<tensor_tag>>(local_errors) =
          local_rms_square_error;
      get<LocalLinfError<tensor_tag>>(local_errors) = local_linf_error;
      get<LocalL2SquareError<tensor_tag>>(local_errors) = local_l2_square_error;
      return 0;
    };
    expand_pack(record_errors(tmpl::type_<Tensors>{}, tensors)...);
    const size_t num_points = get_first_argument(tensors...).begin()->size();

    // Send data to reduction observer
    auto& local_observer =
        *Parallel::get_parallel_component<observers::Observer<Metavariables>>(
             cache)
             .ckLocalBranch();
    const std::string subfile_path_with_suffix =
        subfile_path_ + observation_key_suffix.value_or("none");
    Parallel::simple_action<observers::Actions::ContributeReductionData>(
        local_observer,
        observers::ObservationId(observation_value,
                                 subfile_path_with_suffix + ".dat"),
        observers::ArrayComponentId{
            std::add_pointer_t<ParallelComponent>{nullptr},
            Parallel::ArrayIndex<ArrayIndex>(array_index)},
        subfile_path_with_suffix,
        std::vector<std::string>{
            db::tag_name<ObservationValueTag>(), "NumberOfPoints", "Volume",
            ("RmsError(" + db::tag_name<Tensors>() + ")")...,
            ("LinfError(" + db::tag_name<Tensors>() + ")")...,
            ("L2Error(" + db::tag_name<Tensors>() + ")")...},
        ReductionData{static_cast<double>(observation_value), num_points,
                      local_volume,
                      get<LocalRmsSquareError<Tensors>>(local_errors)...,
                      get<LocalLinfError<Tensors>>(local_errors)...,
                      get<LocalL2SquareError<Tensors>>(local_errors)...});
  }

  template <typename OptionalAnalyticSolutions, typename Metavariables,
            typename ArrayIndex, typename ParallelComponent>
  void operator()(const typename ObservationValueTag::type& observation_value,
                  const Mesh<Dim>& mesh,
                  const Scalar<DataVector>& det_inv_jacobian,
                  const typename Tensors::type&... tensors,
                  const OptionalAnalyticSolutions& optional_analytic_solutions,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ArrayIndex& array_index,
                  const ParallelComponent* const meta) const noexcept {
    this->operator()(observation_value, std::make_optional(""), mesh,
                     det_inv_jacobian, tensors..., optional_analytic_solutions,
                     cache, array_index, meta);
  }

  using observation_registration_tags = tmpl::conditional_t<
      std::is_same_v<ArraySectionIdTag, void>, tmpl::list<>,
      tmpl::list<observers::Tags::ObservationKeySuffix<ArraySectionIdTag>>>;

  std::pair<observers::TypeOfObservation, observers::ObservationKey>
  get_observation_type_and_key_for_registration(
      const std::optional<std::string>& observation_key_suffix =
          std::make_optional("")) const noexcept {
    return {
        observers::TypeOfObservation::Reduction,
        observers::ObservationKey(
            subfile_path_ + observation_key_suffix.value_or("none") + ".dat")};
  }

  bool needs_evolved_variables() const noexcept override { return true; }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override {
    Event<EventRegistrars>::pup(p);
    p | subfile_path_;
  }

 private:
  std::string subfile_path_;
};

template <size_t Dim, typename ObservationValueTag, typename... Tensors,
          typename ArraySectionIdTag, typename EventRegistrars>
ObserveErrorNorms<Dim, ObservationValueTag, tmpl::list<Tensors...>,
                  ArraySectionIdTag,
                  EventRegistrars>::ObserveErrorNorms(const std::string&
                                                          subfile_name) noexcept
    : subfile_path_("/" + subfile_name) {}

/// \cond
template <size_t Dim, typename ObservationValueTag, typename... Tensors,
          typename ArraySectionIdTag, typename EventRegistrars>
PUP::able::PUP_ID ObserveErrorNorms<Dim, ObservationValueTag,
                                    tmpl::list<Tensors...>, ArraySectionIdTag,
                                    EventRegistrars>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace Events
}  // namespace dg
