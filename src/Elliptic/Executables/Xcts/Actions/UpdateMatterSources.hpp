// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Executables/Xcts/InterpolationTargets/StarCenters.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/TMPL.hpp"

namespace SolveXcts::Actions {

template <int ConformalMatterScale>
struct UpdateMatterSources {
 private:
  using matter_sources_tag = ::Tags::Variables<
      tmpl::list<gr::Tags::Conformal<gr::Tags::EnergyDensity<DataVector>,
                                     ConformalMatterScale>,
                 gr::Tags::Conformal<gr::Tags::StressTrace<DataVector>,
                                     ConformalMatterScale>,
                 gr::Tags::Conformal<
                     gr::Tags::MomentumDensity<3, Frame::Inertial, DataVector>,
                     ConformalMatterScale>>>;
  using argument_tags =
      tmpl::list<Xcts::Tags::ConformalFactor<DataVector>,
                 Xcts::Tags::LapseTimesConformalFactor<DataVector>,
                 SolveXcts::OptionTags::Injections,
                 SolveXcts::Tags::InjectionEnergies,
                 SolveXcts::OptionTags::EquationOfState,
                 domain::Tags::Coordinates<3, Frame::Inertial>>;

 public:
  using simple_tags = tmpl::list<matter_sources_tag>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    db::mutate_apply<tmpl::list<matter_sources_tag>, argument_tags>(
        [](const auto matter_sources,
           const Scalar<DataVector>& conformal_factor,
           const Scalar<DataVector>& lapse_times_conformal_factor,
           const std::vector<SolveXcts::Injection>& injections,
           const DataVector& injection_energies,
           const EquationsOfState::PolytropicFluid<true>& eos,
           const tnsr::I<DataVector, 3>& inertial_coords) {
          static_assert(ConformalMatterScale == 0, "Check this");
          const size_t num_points = get(conformal_factor).size();
          if (UNLIKELY(matter_sources->number_of_grid_points() != num_points)) {
            *matter_sources = typename matter_sources_tag::type{num_points, 0.};
          }
          ASSERT(injections.size() == injection_energies.size(),
                 "Size mismatch");
          if (injections.empty()) {
            return;
          }
          auto& conformal_energy_density =
              get<gr::Tags::Conformal<gr::Tags::EnergyDensity<DataVector>,
                                      ConformalMatterScale>>(*matter_sources);
          auto& conformal_stress_trace =
              get<gr::Tags::Conformal<gr::Tags::StressTrace<DataVector>,
                                      ConformalMatterScale>>(*matter_sources);
          auto& conformal_momentum_density = get<gr::Tags::Conformal<
              gr::Tags::MomentumDensity<3, Frame::Inertial, DataVector>,
              ConformalMatterScale>>(*matter_sources);
          Scalar<DataVector> specific_enthalpy{num_points, 1.};
          const auto closest_injection_energy =
              [&injections,
               &injection_energies](const std::array<double, 3>& x) {
                double energy = 0.;
                double closest_r = std::numeric_limits<double>::infinity();
                for (size_t i = 0; i < injections.size(); ++i) {
                  const auto& injection = injections.at(i);
                  const double r = sqrt(square(injection.position[0] - x[0]) +
                                        square(injection.position[1] - x[1]) +
                                        square(injection.position[2] - x[2]));
                  if (r < closest_r and r <= injection.max_radius) {
                    closest_r = r;
                    energy = injection_energies[i];
                  }
                }
                return energy;
              };
          for (size_t i = 0; i < num_points; ++i) {
            const double injection_energy = closest_injection_energy(
                {{get<0>(inertial_coords)[i], get<1>(inertial_coords)[i],
                  get<2>(inertial_coords)[i]}});
            const double lapse =
                get(lapse_times_conformal_factor)[i] / get(conformal_factor)[i];
            if (injection_energy > lapse) {
              get(specific_enthalpy)[i] = injection_energy / lapse;
            }
          }

          // rho_0
          conformal_energy_density =
              eos.rest_mass_density_from_enthalpy(specific_enthalpy);
          // P
          conformal_stress_trace =
              eos.pressure_from_density(conformal_energy_density);
          // rho = rho_0 * h - P
          get(conformal_energy_density) *= get(specific_enthalpy);
          get(conformal_energy_density) -= get(conformal_stress_trace);
          // S = 3 * P
          get(conformal_stress_trace) *= 3.;
          // S^i = 0
          std::fill(conformal_momentum_density.begin(),
                    conformal_momentum_density.end(), 0.);
        },
        make_not_null(&box));
    return {std::move(box)};
  }
};

}  // namespace SolveXcts::Actions
