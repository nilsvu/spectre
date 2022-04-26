// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <pup.h>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "Options/Options.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf.hpp"
#include "ParallelAlgorithms/Actions/ReceiveBroadcast.hpp"
#include "ParallelAlgorithms/Interpolation/Interpolate.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace SolveXcts {

struct Injection {
  struct Position {
    using type = std::array<double, 3>;
    static constexpr Options::String help = "Position of the injection";
  };
  struct CentralDensity {
    using type = double;
    static constexpr Options::String help =
        "Rest mass density at the center of the star";
  };
  struct MaxRadius {
    using type = double;
    static constexpr Options::String help =
        "Radial limit to which star can extend";
  };
  static constexpr Options::String help = "An injection";
  using options = tmpl::list<Position, CentralDensity, MaxRadius>;
  std::array<double, 3> position;
  double central_density;
  double max_radius;
  void pup(PUP::er& p) {
    p | position;
    p | central_density;
    p | max_radius;
  }
};

namespace OptionTags {

struct Injections : db::SimpleTag {
  using type = std::vector<SolveXcts::Injection>;
  using option_tags = tmpl::list<Injections>;
  static constexpr Options::String help = "Injection energies";
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& value) { return value; }
};

struct EquationOfState : db::SimpleTag {
  using type = EquationsOfState::PolytropicFluid<true>;
  using option_tags = tmpl::list<EquationOfState>;
  static constexpr Options::String help = "Equation of state";
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& value) { return value; }
};

}  // namespace OptionTags

namespace Tags {

/// The injection energy at the star centers
///
/// For TOV stars: $\mathcal{E} = h_c \alpha_c$ conserved throughout the star
struct InjectionEnergies : db::SimpleTag {
  using type = DataVector;
};

struct InterpolationId : db::SimpleTag {
  using type = size_t;
};

}  // namespace Tags

namespace InterpolationTargets {

struct StarCenters {
  using temporal_id = Tags::InterpolationId;
  struct compute_target_points {
    using is_sequential = std::true_type;
    using frame = Frame::Inertial;
    template <typename DbTags, typename Metavariables>
    static tnsr::I<DataVector, 3> points(
        const db::DataBox<DbTags>& box,
        const tmpl::type_<Metavariables>& /*meta*/,
        const size_t& /*temporal_id*/) {
      const auto& injections = get<OptionTags::Injections>(box);
      const size_t num_points = injections.size();
      tnsr::I<DataVector, 3> interpolation_points{num_points};
      for (size_t i = 0; i < num_points; ++i) {
        const auto& injection = injections.at(i);
        get<0>(interpolation_points)[i] = injection.position[0];
        get<1>(interpolation_points)[i] = injection.position[1];
        get<2>(interpolation_points)[i] = injection.position[2];
      }
      return interpolation_points;
    }
  };
  using compute_items_on_source = tmpl::list<>;
  using vars_to_interpolate_to_target =
      tmpl::list<Xcts::Tags::ConformalFactor<DataVector>,
                 Xcts::Tags::LapseTimesConformalFactor<DataVector>>;
  using compute_items_on_target = tmpl::list<>;
  struct post_interpolation_callback {
    template <typename DbTags, typename Metavariables>
    static void apply(const db::DataBox<DbTags>& box,
                      Parallel::GlobalCache<Metavariables>& cache,
                      const size_t& temporal_id) {
      using temporal_id_tag = Tags::InterpolationId;
      const auto& injections = get<OptionTags::Injections>(box);
      const auto& eos = get<OptionTags::EquationOfState>(box);
      const size_t num_points = injections.size();
      DataVector injection_energies(num_points);
      std::vector<std::string> legend{db::tag_name<temporal_id_tag>()};
      legend.reserve(num_points);
      for (size_t i = 0; i < num_points; ++i) {
        const auto& injection = injections.at(i);
        const double lapse_at_center =
            get(get<Xcts::Tags::LapseTimesConformalFactor<DataVector>>(
                box))[i] /
            get(get<Xcts::Tags::ConformalFactor<DataVector>>(box))[i];
        const double specific_enthalpy_at_center =
            get(eos.specific_enthalpy_from_density(
                Scalar<double>(injection.central_density)));
        injection_energies[i] = specific_enthalpy_at_center * lapse_at_center;
        Parallel::printf("lapse at center %zu: %e\n", i, lapse_at_center);
        Parallel::printf("injection energy %zu: %e\n", i,
                         injection_energies.at(i));
        legend.push_back("InjectionEnergy" +
                         (num_points > 1 ? get_output(i) : ""));
      }

      Parallel::receive_data<
          ::Tags::BroadcastInbox<Tags::InjectionEnergies, temporal_id_tag>>(
          Parallel::get_parallel_component<
              typename Metavariables::dg_element_array>(cache),
          temporal_id, injection_energies);

      auto& observer_writer_component = Parallel::get_parallel_component<
          observers::ObserverWriter<Metavariables>>(cache);
      Parallel::threaded_action<
          observers::ThreadedActions::WriteReductionDataRow>(
          // Node 0 is always the writer, so directly call the component on that
          // node
          observer_writer_component[0], std::string{"/StarCenters"},
          std::move(legend), std::make_tuple(temporal_id, injection_energies));
    }
  };
};

}  // namespace InterpolationTargets

namespace Actions {

struct InterpolateToStarCenters {
  using simple_tags = tmpl::list<Tags::InterpolationId>;
  using const_global_cache_tags =
      tmpl::list<OptionTags::Injections, OptionTags::EquationOfState>;

  template <typename DbTagList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagList>&&> apply(
      db::DataBox<DbTagList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    using InterpolationTarget = InterpolationTargets::StarCenters;
    using TemporalIdTag = Tags::InterpolationId;
    db::mutate<TemporalIdTag>(
        make_not_null(&box),
        [](const auto local_temporal_id) { ++(*local_temporal_id); });
    intrp::interpolate<
        InterpolationTarget,
        typename InterpolationTarget::vars_to_interpolate_to_target>(
        get<TemporalIdTag>(box), get<domain::Tags::Mesh<3>>(box), cache,
        array_index, get<Xcts::Tags::ConformalFactor<DataVector>>(box),
        get<Xcts::Tags::LapseTimesConformalFactor<DataVector>>(box));
    return {std::move(box)};
  }
};

}  // namespace Actions

}  // namespace SolveXcts
