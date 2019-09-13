// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/preprocessor/list/for_each.hpp>
#include <boost/preprocessor/tuple/to_list.hpp>
#include <limits>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/TovStar.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Vacuum.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/Tags.hpp"  // IWYU pragma: keep
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;  // IWYU pragma: keep
}  // namespace PUP
/// \endcond

namespace Xcts {
namespace AnalyticData {

class NeutronStarBinary {
 public:
  using equation_of_state_type = EquationsOfState::PolytropicFluid<true>;

  struct CentralDensities {
    using type = std::array<double, 2>;
    static constexpr OptionString help = {"The central densities of the stars"};
  };

  struct AngularVelocity {
    using type = double;
    static constexpr OptionString help = {
        "The angular velocity of the binary."};
  };

  struct PolytropicConstant {
    using type = double;
    static constexpr OptionString help = {
        "The polytropic constant of the fluid."};
    static type lower_bound() noexcept { return 0.; }
  };

  struct PolytropicExponent {
    using type = double;
    static constexpr OptionString help = {
        "The polytropic exponent of the fluid."};
    static type lower_bound() noexcept { return 1.; }
  };

  using options = tmpl::list<CentralDensities, AngularVelocity,
                             PolytropicConstant, PolytropicExponent>;

  static constexpr OptionString help = {
      "Two neutron stars with a polytropic EOS and given central densities "
      "that orbit around each other."};

  NeutronStarBinary() = default;
  NeutronStarBinary(const NeutronStarBinary& /*rhs*/) = delete;
  NeutronStarBinary& operator=(const NeutronStarBinary& /*rhs*/) = delete;
  NeutronStarBinary(NeutronStarBinary&& /*rhs*/) noexcept = default;
  NeutronStarBinary& operator=(NeutronStarBinary&& /*rhs*/) noexcept = default;
  ~NeutronStarBinary() = default;

  NeutronStarBinary(std::array<double, 2> central_densities,
                    double angular_velocity, double polytropic_constant,
                    double polytropic_exponent) noexcept;

  template <typename DataType>
  using RadialVariables = DataType;  // placeholder

  /// Retrieve a collection of variables at coordinates `x`
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataType, 3>& x,
                                         tmpl::list<Tags...> /*meta*/) const
      noexcept {
    auto x_star_left = x;
    auto x_star_right = x;
    get<0>(x_star_left) += 50.;
    get<0>(x_star_right) -= 50.;
    const Variables<tmpl::list<Tags...>> superposed_vars =
        variables_from_tagged_tuple(
            stars_[0].variables(x_star_left, tmpl::list<Tags...>{})) +
        variables_from_tagged_tuple(
            stars_[1].variables(x_star_right, tmpl::list<Tags...>{})) -
        variables_from_tagged_tuple(
            vacuum_.variables(x, tmpl::list<Tags...>{}));
    tuples::TaggedTuple<Tags...> superposed_tuple{};
    tmpl::for_each<tmpl::list<Tags...>>(
        [&superposed_tuple, &superposed_vars ](auto tag_v) noexcept {
          using tag = tmpl::type_from<decltype(tag_v)>;
          get<tag>(superposed_tuple) = get<tag>(superposed_vars);
        });
    return superposed_tuple;
  }

  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept;  //  NOLINT

  const EquationsOfState::PolytropicFluid<true>& equation_of_state() const
      noexcept {
    return equation_of_state_;
  }

  struct BackgroundFieldsCompute
      : ::Tags::Variables<tmpl::list<gr::Tags::EnergyDensity<DataVector>,
                                     gr::Tags::StressTrace<DataVector>>>,
        db::ComputeTag {
    using base =
        ::Tags::Variables<tmpl::list<gr::Tags::EnergyDensity<DataVector>,
                                     gr::Tags::StressTrace<DataVector>>>;
    using argument_tags =
        tmpl::list<::Tags::AnalyticSolutionComputer<NeutronStarBinary>,
                   Xcts::Tags::LapseAtStarCenters,
                   ::Tags::Coordinates<3, Frame::Inertial>,
                   Xcts::Tags::ConformalFactor<DataVector>,
                   Xcts::Tags::LapseTimesConformalFactor<DataVector>>;
    static db::item_type<base> function(
        const NeutronStarBinary& solution,
        const std::array<double, 2>& lapse_at_star_centers,
        const tnsr::I<DataVector, 3, Frame::Inertial>& x,
        const Scalar<DataVector>& conformal_factor,
        const Scalar<DataVector>& lapse_times_conformal_factor) noexcept {
      // From the solution, retrieve the central enthalpy times the lapse, which
      // we use to pick a physical situation.
      //   const tnsr::I<double, 3> origin{{{0., 0., 0.}}};
      //   const auto specific_enthalpy_and_lapse_at_origin =
      //   solution.variables(
      //       origin, tmpl::list<hydro::Tags::SpecificEnthalpy<double>,
      //                          gr::Tags::Lapse<double>>{});
      //   const double specific_enthalpy_times_lapse_at_origin =
      //       get(get<hydro::Tags::SpecificEnthalpy<double>>(
      //           specific_enthalpy_and_lapse_at_origin)) *
      //       get(get<gr::Tags::Lapse<double>>(
      //           specific_enthalpy_and_lapse_at_origin));
      //   const double specific_enthalpy_at_origin =
      //      get(get<hydro::Tags::SpecificEnthalpy<double>>(solution.variables(
      //           origin,
      //           tmpl::list<hydro::Tags::SpecificEnthalpy<double>>{})));
      const auto& eos = solution.equation_of_state();
      const std::array<double, 2> specific_enthalpy_at_star_centers{
          {get(eos.specific_enthalpy_from_density(
               Scalar<double>(solution.central_rest_mass_densities_[0]))),
           get(eos.specific_enthalpy_from_density(
               Scalar<double>(solution.central_rest_mass_densities_[1])))}};
      Parallel::printf("h at star centers: %s\n",
                       specific_enthalpy_at_star_centers);
      const std::array<double, 2> specific_enthalpy_times_lapse_at_star_centers{
          {specific_enthalpy_at_star_centers[0] * lapse_at_star_centers[0],
           specific_enthalpy_at_star_centers[1] * lapse_at_star_centers[1]}};
      auto coord_separation_from_left_star = x;
      auto coord_separation_from_right_star = x;
      get<0>(coord_separation_from_left_star) += 50.;
      get<0>(coord_separation_from_right_star) -= 50.;
      const auto coord_dist_from_left_star =
          get(magnitude(coord_separation_from_left_star));
      const auto coord_dist_from_right_star =
          get(magnitude(coord_separation_from_right_star));
      // From the variable data, compute the lapse and then the specific
      // enthalpy.
      const auto lapse =
          get(lapse_times_conformal_factor) / get(conformal_factor);
      auto specific_enthalpy =
          make_with_value<Scalar<DataVector>>(conformal_factor, 1.);
      for (size_t i = 0; i < get_size(get(specific_enthalpy)); i++) {
        if (coord_dist_from_left_star[i] < coord_dist_from_right_star[i]) {
          if (lapse[i] < specific_enthalpy_times_lapse_at_star_centers[0]) {
            get(specific_enthalpy)[i] =
                specific_enthalpy_times_lapse_at_star_centers[0] / lapse[i];
          }
        } else {
          if (lapse[i] < specific_enthalpy_times_lapse_at_star_centers[1]) {
            get(specific_enthalpy)[i] =
                specific_enthalpy_times_lapse_at_star_centers[1] / lapse[i];
          }
        }
      }
      // Use the specific enthalpy to compute the quantities that feed back into
      // the XCTS equations, i.e. the energy density and the stress trace.
      auto background_fields =
          make_with_value<db::item_type<base>>(conformal_factor, 0.);
      const auto rest_mass_density =
          eos.rest_mass_density_from_enthalpy(specific_enthalpy);
      const auto pressure = eos.pressure_from_density(rest_mass_density);
      get(get<gr::Tags::EnergyDensity<DataVector>>(background_fields)) =
          get(specific_enthalpy) * get(rest_mass_density) - get(pressure);
      get(get<gr::Tags::StressTrace<DataVector>>(background_fields)) =
          3. * get(pressure);
      return background_fields;
    }
  };

  using compute_tags = tmpl::list<BackgroundFieldsCompute>;

 private:
  friend bool operator==(const NeutronStarBinary& lhs,
                         const NeutronStarBinary& rhs) noexcept;

  std::array<double, 2> central_rest_mass_densities_{
      {std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN()}};
  double angular_velocity_ = std::numeric_limits<double>::signaling_NaN();
  double polytropic_constant_ = std::numeric_limits<double>::signaling_NaN();
  double polytropic_exponent_ = std::numeric_limits<double>::signaling_NaN();
  EquationsOfState::PolytropicFluid<true> equation_of_state_{};
  std::array<Xcts::Solutions::TovStar, 2> stars_{};
  Xcts::Solutions::Vacuum vacuum_{};
};

bool operator!=(const NeutronStarBinary& lhs,
                const NeutronStarBinary& rhs) noexcept;

}  // namespace AnalyticData
}  // namespace Xcts
