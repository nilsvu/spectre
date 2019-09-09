// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Elliptic/Tags.hpp"
#include "IO/Observer/Actions.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/VolumeActions.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Reduction.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Xcts {
namespace Actions {

struct Observe {
 private:
  using observed_reduction_data = Parallel::ReductionData<
      Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
      Parallel::ReductionDatum<size_t, funcl::Plus<>>,
      Parallel::ReductionDatum<double, funcl::Plus<>,
                               funcl::Sqrt<funcl::Divides<>>,
                               std::index_sequence<1>>>;

 public:
  // Compile-time interface for observers
  struct ElementObservationType {};
  template <typename ParallelComponent, typename DbTagsList,
            typename ArrayIndex>
  static std::pair<observers::TypeOfObservation, observers::ObservationId>
  register_info(const db::DataBox<DbTagsList>& box,
                const ArrayIndex& /*array_index*/) noexcept {
    return {observers::TypeOfObservation::ReductionAndVolume,
            observers::ObservationId{db::get<elliptic::Tags::IterationId>(box),
                                     ElementObservationType{}}};
  }
  using observed_reduction_data_tags =
      observers::make_reduction_data_tags<tmpl::list<observed_reduction_data>>;

  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ElementIndex<Dim>& array_index,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const auto& iteration_id = get<elliptic::Tags::IterationId>(box);
    const auto& mesh = get<::Tags::Mesh<Dim>>(box);
    const std::string element_name = MakeString{} << ElementId<Dim>(array_index)
                                                  << '/';

    // Retrieve the current numeric solution
    const auto& conformal_factor =
        get<Xcts::Tags::ConformalFactor<DataVector>>(box);
    const auto& lapse =
        get(get<Xcts::Tags::LapseTimesConformalFactor<DataVector>>(box)) /
        get(get<Xcts::Tags::ConformalFactor<DataVector>>(box));
    // const auto& conformal_factor_correction =
    //     get<NonlinearSolver::Tags::Correction<
    //         Xcts::Tags::ConformalFactor<DataVector>>>(box);

    // Compute the analytic solution
    const auto& inertial_coordinates =
        db::get<::Tags::Coordinates<Dim, Frame::Inertial>>(box);
    const auto conformal_factor_analytic =
        get<::Tags::Analytic<Xcts::Tags::ConformalFactor<DataVector>>>(box);
    const auto lapse_analytic =
        get(get<::Tags::Analytic<
                Xcts::Tags::LapseTimesConformalFactor<DataVector>>>(box)) /
        get(conformal_factor_analytic);

    // Compute error between numeric and analytic solutions
    const DataVector conformal_factor_error =
        get(conformal_factor) - get(conformal_factor_analytic);
    const DataVector lapse_error = lapse - lapse_analytic;

    // Compute l2 error squared over local element
    const double local_l2_error_square =
        alg::accumulate(conformal_factor_error, 0.0,
                        funcl::Plus<funcl::Identity, funcl::Square<>>{});

    // Collect volume data
    // Remove tensor types, only storing individual components
    std::vector<TensorComponent> components;
    components.reserve(Dim + 8);
    components.emplace_back(
        element_name + Xcts::Tags::ConformalFactor<DataVector>::name(),
        get(conformal_factor));
    components.emplace_back(
        element_name + "Error(" +
            Xcts::Tags::ConformalFactor<DataVector>::name() + ")",
        conformal_factor_error);
    components.emplace_back(
        element_name +
            ::Tags::Analytic<Xcts::Tags::ConformalFactor<DataVector>>::name(),
        get(conformal_factor_analytic));
    components.emplace_back(element_name + "Lapse", lapse);
    components.emplace_back(element_name + "Error(Lapse)", lapse_error);
    components.emplace_back(element_name + "Analytic(Lapse)", lapse_analytic);
    components.emplace_back(
        element_name + gr::Tags::EnergyDensity<DataVector>::name(),
        get(get<gr::Tags::EnergyDensity<DataVector>>(box)));
    components.emplace_back(
        element_name + gr::Tags::StressTrace<DataVector>::name(),
        get(get<gr::Tags::StressTrace<DataVector>>(box)));
    components.emplace_back(element_name + "InertialCoordinates_x",
                            get<0>(inertial_coordinates));
    if (Dim >= 2) {
      components.emplace_back(element_name + "InertialCoordinates_y",
                              inertial_coordinates.get(1));
    }
    if (Dim >= 3) {
      components.emplace_back(element_name + "InertialCoordinates_z",
                              inertial_coordinates.get(2));
    }

    // Send data to volume observer
    auto& local_observer =
        *Parallel::get_parallel_component<observers::Observer<Metavariables>>(
             cache)
             .ckLocalBranch();
    Parallel::simple_action<observers::Actions::ContributeVolumeData>(
        local_observer,
        observers::ObservationId(iteration_id, ElementObservationType{}),
        std::string{"/element_data"},
        observers::ArrayComponentId(
            std::add_pointer_t<ParallelComponent>{nullptr},
            Parallel::ArrayIndex<ElementIndex<Dim>>(array_index)),
        std::move(components), mesh.extents());

    // Send data to reduction observer
    Parallel::simple_action<observers::Actions::ContributeReductionData>(
        local_observer,
        observers::ObservationId(iteration_id, ElementObservationType{}),
        std::string{"/element_data"},
        std::vector<std::string>{"Iteration", "NumberOfPoints", "L2Error"},
        observed_reduction_data{iteration_id, mesh.number_of_grid_points(),
                                local_l2_error_square});

    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace Xcts
