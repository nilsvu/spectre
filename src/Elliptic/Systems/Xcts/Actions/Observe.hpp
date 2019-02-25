// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "IO/Observer/Actions.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/VolumeActions.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "NumericalAlgorithms/NonlinearSolver/Tags.hpp"
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
      Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
      Parallel::ReductionDatum<size_t, funcl::Plus<>>,
      Parallel::ReductionDatum<double, funcl::Plus<>,
                               funcl::Sqrt<funcl::Divides<>>,
                               std::index_sequence<1>>>;

 public:
  // Compile-time interface for observers
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
    const auto& iteration_id = get<typename Metavariables::temporal_id>(box);
    const auto& mesh = get<::Tags::Mesh<Dim>>(box);
    const std::string element_name = MakeString{} << ElementId<Dim>(array_index)
                                                  << '/';

    // Retrieve the current numeric solution
    const auto& conformal_factor =
        get<Xcts::Tags::ConformalFactor<DataVector>>(box);
    // const auto& conformal_factor_correction =
    //     get<NonlinearSolver::Tags::Correction<
    //         Xcts::Tags::ConformalFactor<DataVector>>>(box);

    // Compute the analytic solution
    const auto& inertial_coordinates =
        db::get<::Tags::Coordinates<Dim, Frame::Inertial>>(box);
    const auto conformal_factor_analytic =
        get<Xcts::Tags::ConformalFactor<DataVector>>(
            Parallel::get<typename Metavariables::analytic_solution_tag>(cache)
                .variables(
                    inertial_coordinates,
                    tmpl::list<Xcts::Tags::ConformalFactor<DataVector>>{}));

    // Compute error between numeric and analytic solutions
    const DataVector conformal_factor_error =
        get(conformal_factor) - get(conformal_factor_analytic);

    // Compute l2 error squared over local element
    const double local_l2_error_square =
        alg::accumulate(conformal_factor_error, 0.0,
                        funcl::Plus<funcl::Identity, funcl::Square<>>{});

    // Collect volume data
    // Remove tensor types, only storing individual components
    std::vector<TensorComponent> components;
    components.reserve(3 + Dim + 1);
    components.emplace_back(
        element_name + Xcts::Tags::ConformalFactor<DataVector>::name(),
        get(conformal_factor));
    components.emplace_back(
        element_name + Xcts::Tags::ConformalFactor<DataVector>::name() +
            "Analytic",
        get(conformal_factor_analytic));
    components.emplace_back(
        element_name + Xcts::Tags::ConformalFactor<DataVector>::name() +
            "Error",
        conformal_factor_error);
    components.emplace_back(
        element_name + gr::Tags::EnergyDensity<DataVector>::name(),
        get(get<gr::Tags::EnergyDensity<DataVector>>(box)));
    // components.emplace_back(
    //     element_name + Xcts::Tags::ConformalFactor<DataVector>::name() +
    //         "Correction",
    //     get(conformal_factor_correction));
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
        observers::ObservationId(
            iteration_id.value(),
            typename Metavariables::element_observation_type{}),
        std::string{"/element_data"},
        observers::ArrayComponentId(
            std::add_pointer_t<ParallelComponent>{nullptr},
            Parallel::ArrayIndex<ElementIndex<Dim>>(array_index)),
        std::move(components), mesh.extents());

    // Send data to reduction observer
    Parallel::simple_action<observers::Actions::ContributeReductionData>(
        local_observer,
        observers::ObservationId(
            iteration_id.value(),
            typename Metavariables::element_observation_type{}),
        std::string{"/element_data"},
        std::vector<std::string>{"Nonlinear Iteration", "NumberOfPoints",
                                 "L2Error"},
        observed_reduction_data{
            get<NonlinearSolver::Tags::IterationId>(iteration_id),
            mesh.number_of_grid_points(), local_l2_error_square});

    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace Xcts
