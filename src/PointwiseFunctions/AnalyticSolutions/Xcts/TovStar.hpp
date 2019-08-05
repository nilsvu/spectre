// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/preprocessor/list/for_each.hpp>
#include <boost/preprocessor/tuple/to_list.hpp>
#include <limits>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/TovIsotropic.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
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
namespace Solutions {

/*!
 * \brief A static spherically symmetric star
 *
 * An analytic solution for a static, spherically-symmetric star found by
 * solving the Tolman-Oppenheimer-Volkoff (TOV) equations.  The equation of
 * state is assumed to be that of a polytropic fluid.
 */
class TovStar {
 public:
  using equation_of_state_type = EquationsOfState::PolytropicFluid<true>;

  /// The central density of the star.
  struct CentralDensity {
    using type = double;
    static constexpr OptionString help = {"The central density of the star."};
    static type lower_bound() noexcept { return 0.; }
  };

  /// The polytropic constant of the polytropic fluid.
  struct PolytropicConstant {
    using type = double;
    static constexpr OptionString help = {
        "The polytropic constant of the fluid."};
    static type lower_bound() noexcept { return 0.; }
  };

  /// The polytropic exponent of the polytropic fluid.
  struct PolytropicExponent {
    using type = double;
    static constexpr OptionString help = {
        "The polytropic exponent of the fluid."};
    static type lower_bound() noexcept { return 1.; }
  };

  using options =
      tmpl::list<CentralDensity, PolytropicConstant, PolytropicExponent>;

  static constexpr OptionString help = {
      "A static, spherically-symmetric star found by solving the \n"
      "Tolman-Oppenheimer-Volkoff (TOV) equations, with a given central \n"
      "density and polytropic fluid."};

  TovStar() = default;
  TovStar(const TovStar& /*rhs*/) = delete;
  TovStar& operator=(const TovStar& /*rhs*/) = delete;
  TovStar(TovStar&& /*rhs*/) noexcept = default;
  TovStar& operator=(TovStar&& /*rhs*/) noexcept = default;
  ~TovStar() = default;

  TovStar(double central_rest_mass_density, double polytropic_constant,
          double polytropic_exponent) noexcept;

  const gr::Solutions::TovIsotropic& radial_solution() const noexcept {
    return radial_solution_;
  }

  template <typename DataType>
  using RadialVariables =
      gr::Solutions::TovIsotropic::RadialVariables<DataType>;

  /// Retrieve a collection of variables at coordinates `x`
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataType, 3>& x,
                                         tmpl::list<Tags...> /*meta*/) const
      noexcept {
    auto radial_vars =
        radial_solution().radial_variables(equation_of_state_, x);
    return {get<Tags>(variables(x, tmpl::list<Tags>{}, radial_vars))...};
  }

  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept;  //  NOLINT

  const EquationsOfState::PolytropicFluid<true>& equation_of_state() const
      noexcept {
    return equation_of_state_;
  }

  double central_rest_mass_density() const noexcept {
    return central_rest_mass_density_;
  }

  struct BackgroundFieldsComputeAnalytic
      : ::Tags::Variables<tmpl::list<
            gr::Tags::EnergyDensity<DataVector>,
            gr::Tags::StressTrace<DataVector>,
            gr::Tags::MomentumDensity<3, Frame::Inertial, DataVector>>>,
        db::ComputeTag {
    using base = ::Tags::Variables<tmpl::list<
        gr::Tags::EnergyDensity<DataVector>, gr::Tags::StressTrace<DataVector>,
        gr::Tags::MomentumDensity<3, Frame::Inertial, DataVector>>>;
    using argument_tags = tmpl::list<::Tags::AnalyticSolution<TovStar>,
                                     ::Tags::Coordinates<3, Frame::Inertial>>;
    static db::item_type<base> function(
        const TovStar& solution, const tnsr::I<DataVector, 3, Frame::Inertial>&
                                     inertial_coords) noexcept {
      return variables_from_tagged_tuple(solution.variables(
          inertial_coords, db::get_variables_tags_list<base>{}));
    }
  };

  struct BackgroundFieldsComputeDynamic
      : ::Tags::Variables<tmpl::list<
            gr::Tags::EnergyDensity<DataVector>,
            gr::Tags::StressTrace<DataVector>,
            gr::Tags::MomentumDensity<3, Frame::Inertial, DataVector>>>,
        db::ComputeTag {
    using base = ::Tags::Variables<tmpl::list<
        gr::Tags::EnergyDensity<DataVector>, gr::Tags::StressTrace<DataVector>,
        gr::Tags::MomentumDensity<3, Frame::Inertial, DataVector>>>;
    using argument_tags =
        tmpl::list<::Tags::AnalyticSolution<TovStar>,
                   Xcts::Tags::ConformalFactor<DataVector>,
                   Xcts::Tags::LapseTimesConformalFactor<DataVector>,
                   hydro::Tags::InjectionEnergy>;
    static db::item_type<base> function(
        const TovStar& solution, const Scalar<DataVector>& conformal_factor,
        const Scalar<DataVector>& lapse_times_conformal_factor,
        // const Scalar<DataVector>& conformal_factor_at_center,
        // const Scalar<DataVector>&
        //     lapse_times_conformal_factor_at_center,
        const double& injection_energy) noexcept {
      // From the variable data, compute the lapse and then the specific
      // enthalpy.
      const auto lapse =
          get(lapse_times_conformal_factor) / get(conformal_factor);
      auto specific_enthalpy =
          make_with_value<Scalar<DataVector>>(conformal_factor, 1.);
      for (size_t i = 0; i < get_size(get(specific_enthalpy)); i++) {
        if (lapse[i] < injection_energy) {
          get(specific_enthalpy)[i] = injection_energy / lapse[i];
        }
      }
      // Use the specific enthalpy to compute the quantities that feed back into
      // the XCTS equations, i.e. the energy density and the stress trace.
      const auto& eos = solution.equation_of_state();
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

  using compute_tags = tmpl::list<BackgroundFieldsComputeDynamic>;

  using observe_fields = tmpl::list<gr::Tags::EnergyDensity<DataVector>,
                                    gr::Tags::StressTrace<DataVector>>;

 private:
  friend bool operator==(const TovStar& lhs, const TovStar& rhs) noexcept;

  template <typename DataType>
  using ConformalFactorGradient =
      ::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>, tmpl::size_t<3>,
                    Frame::Inertial>;
  template <typename DataType>
  using LapseTimesConformalFactorGradient =
      ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
                    tmpl::size_t<3>, Frame::Inertial>;
  template <typename DataType>
  using Shift = gr::Tags::Shift<3, Frame::Inertial, DataType>;
  template <typename DataType>
  using ShiftStrain = Xcts::Tags::ShiftStrain<3, Frame::Inertial, DataType>;
  template <typename DataType>
  using MomentumDensity =
      gr::Tags::MomentumDensity<3, Frame::Inertial, DataType>;

#define FUNC_DECL(r, data, elem)                                    \
  template <typename DataType>                                      \
  tuples::TaggedTuple<elem> variables(                              \
      const tnsr::I<DataType, 3>& x, tmpl::list<elem> /*meta*/,     \
      const RadialVariables<DataType>& radial_vars) const noexcept; \
  template <typename DataType>                                      \
  tuples::TaggedTuple<::Tags::Initial<elem>> variables(             \
      const tnsr::I<DataType, 3>& x,                                \
      tmpl::list<::Tags::Initial<elem>> /*meta*/,                   \
      const RadialVariables<DataType>& radial_vars) const noexcept; \
  template <typename DataType>                                      \
  tuples::TaggedTuple<::Tags::FixedSource<elem>> variables(         \
      const tnsr::I<DataType, 3>& x,                                \
      tmpl::list<::Tags::FixedSource<elem>> /*meta*/,               \
      const RadialVariables<DataType>& radial_vars) const noexcept;

#define MY_LIST                                                         \
  BOOST_PP_TUPLE_TO_LIST(                                               \
      9, (Xcts::Tags::ConformalFactor<DataType>,                        \
          ConformalFactorGradient<DataType>,                            \
          Xcts::Tags::LapseTimesConformalFactor<DataType>,              \
          LapseTimesConformalFactorGradient<DataType>, Shift<DataType>, \
          ShiftStrain<DataType>, gr::Tags::EnergyDensity<DataType>,     \
          gr::Tags::StressTrace<DataType>, MomentumDensity<DataType>))

  BOOST_PP_LIST_FOR_EACH(FUNC_DECL, _, MY_LIST)
#undef MY_LIST
#undef FUNC_DECL

  double central_rest_mass_density_ =
      std::numeric_limits<double>::signaling_NaN();
  double polytropic_constant_ = std::numeric_limits<double>::signaling_NaN();
  double polytropic_exponent_ = std::numeric_limits<double>::signaling_NaN();
  EquationsOfState::PolytropicFluid<true> equation_of_state_{};
  gr::Solutions::TovIsotropic radial_solution_{};
};

bool operator!=(const TovStar& lhs, const TovStar& rhs) noexcept;

}  // namespace Solutions
}  // namespace Xcts
