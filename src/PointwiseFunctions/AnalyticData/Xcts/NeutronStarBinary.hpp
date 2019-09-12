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
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/TovStar.hpp"
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
    return {get<Tags>(
        variables(x, tmpl::list<Tags>{}, RadialVariables<DataType>{}))...};
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
                   Xcts::Tags::ConformalFactor<DataVector>,
                   Xcts::Tags::LapseTimesConformalFactor<DataVector>>;
    static db::item_type<base> function(
        const NeutronStarBinary& /*solution*/,
        const Scalar<DataVector>& conformal_factor,
        const Scalar<DataVector>& /*lapse_times_conformal_factor*/) noexcept {
      auto background_fields =
          make_with_value<db::item_type<base>>(conformal_factor, 0.);
      return background_fields;
    }
  };

  using compute_tags = tmpl::list<BackgroundFieldsCompute>;

 private:
  friend bool operator==(const NeutronStarBinary& lhs,
                         const NeutronStarBinary& rhs) noexcept;

  template <typename DataType>
  using ConformalFactorGradient =
      Xcts::Tags::ConformalFactorGradient<3, Frame::Inertial, DataType>;

  template <typename DataType>
  using LapseTimesConformalFactorGradient =
      Xcts::Tags::LapseTimesConformalFactorGradient<3, Frame::Inertial,
                                                    DataType>;

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

#define MY_LIST                                                               \
  BOOST_PP_TUPLE_TO_LIST(4, (Xcts::Tags::ConformalFactor<DataType>,           \
                             ConformalFactorGradient<DataType>,               \
                             Xcts::Tags::LapseTimesConformalFactor<DataType>, \
                             LapseTimesConformalFactorGradient<DataType>))

  BOOST_PP_LIST_FOR_EACH(FUNC_DECL, _, MY_LIST)
#undef MY_LIST
#undef FUNC_DECL

  std::array<double, 2> central_rest_mass_densities_{
      {std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN()}};
  double angular_velocity_ = std::numeric_limits<double>::signaling_NaN();
  double polytropic_constant_ = std::numeric_limits<double>::signaling_NaN();
  double polytropic_exponent_ = std::numeric_limits<double>::signaling_NaN();
  EquationsOfState::PolytropicFluid<true> equation_of_state_{};
  std::array<Xcts::Solutions::TovStar, 2> stars_{};
};

bool operator!=(const NeutronStarBinary& lhs,
                const NeutronStarBinary& rhs) noexcept;

}  // namespace AnalyticData
}  // namespace Xcts
