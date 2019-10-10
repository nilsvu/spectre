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
#include "PointwiseFunctions/OrbitalDynamics/Tags.hpp"
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

  struct Separation {
    using type = double;
    static constexpr OptionString help = {
        "The coordinate separation of the binary at apoapsis."};
    static type lower_bound() noexcept { return 0.; }
  };

  struct Eccentricity {
    using type = double;
    static constexpr OptionString help = {
        "The Newtonian eccentricity of the binary."};
    static type lower_bound() noexcept { return 0.; }
    static type upper_bound() noexcept { return 1.; }
  };

  struct CentralDensities {
    using type = std::array<double, 2>;
    static constexpr OptionString help = {"The central densities of the stars"};
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

  using options = tmpl::list<Separation, Eccentricity, CentralDensities,
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

  NeutronStarBinary(double separation, double eccentricity,
                    std::array<double, 2> central_densities,
                    double polytropic_constant,
                    double polytropic_exponent) noexcept;

  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept;  //  NOLINT

  double separation() const noexcept;
  double eccentricity() const noexcept;
  const std::array<double, 2>& central_rest_mass_densities() const noexcept;
  const std::array<double, 2>& star_centers() const noexcept;
  const EquationsOfState::PolytropicFluid<true>& equation_of_state() const
      noexcept;

  double center_of_mass_estimate() const noexcept;
  double angular_velocity_estimate() const noexcept;

  std::array<tnsr::I<DataVector, 3, Frame::Inertial>, 2>
  star_centered_coordinates(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x) const noexcept;

  template <typename Tag, typename DataType>
  db::item_type<Tag> superposed_tov(const tnsr::I<DataType, 3>& x) const
      noexcept {
    const auto star_centered_coords = star_centered_coordinates(x);
    const Variables<tmpl::list<Tag>> superposed_vars =
        variables_from_tagged_tuple(
            stars_[0].variables(star_centered_coords[0], tmpl::list<Tag>{})) +
        variables_from_tagged_tuple(
            stars_[1].variables(star_centered_coords[1], tmpl::list<Tag>{})) -
        variables_from_tagged_tuple(vacuum_.variables(x, tmpl::list<Tag>{}));
    return get<Tag>(superposed_vars);
  }

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

#define FUNC_DECL(r, data, elem)                                               \
  template <typename DataType>                                                 \
  tuples::TaggedTuple<elem> variables(const tnsr::I<DataType, 3>& x,           \
                                      tmpl::list<elem> /*meta*/)               \
      const noexcept {                                                         \
    return {get<::Tags::Initial<elem>>(                                        \
        variables(x, tmpl::list<::Tags::Initial<elem>>{}))};                   \
  }                                                                            \
  template <typename DataType>                                                 \
  tuples::TaggedTuple<::Tags::Initial<elem>> variables(                        \
      const tnsr::I<DataType, 3>& x,                                           \
      tmpl::list<::Tags::Initial<elem>> /*meta*/) const noexcept {             \
    return {superposed_tov<elem>(x)};                                          \
  }                                                                            \
  template <typename DataType>                                                 \
  tuples::TaggedTuple<::Tags::FixedSource<elem>> variables(                    \
      const tnsr::I<DataType, 3>& x,                                           \
      tmpl::list<::Tags::FixedSource<elem>> /*meta*/) const noexcept {         \
    return {make_with_value<db::item_type<::Tags::FixedSource<elem>>>(x, 0.)}; \
  }

#define MY_LIST                                                               \
  BOOST_PP_TUPLE_TO_LIST(6, (Xcts::Tags::ConformalFactor<DataType>,           \
                             ConformalFactorGradient<DataType>,               \
                             Xcts::Tags::LapseTimesConformalFactor<DataType>, \
                             LapseTimesConformalFactorGradient<DataType>,     \
                             Shift<DataType>, ShiftStrain<DataType>))

  BOOST_PP_LIST_FOR_EACH(FUNC_DECL, _, MY_LIST)
#undef MY_LIST
#undef FUNC_DECL

  /// Retrieve a collection of variables at coordinates `x`
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataType, 3>& x,
                                         tmpl::list<Tags...> /*meta*/) const
      noexcept {
    return {get<Tags>(variables(x, tmpl::list<Tags>{}))...};
  }

  Variables<tmpl::list<
      gr::Tags::EnergyDensity<DataVector>, gr::Tags::StressTrace<DataVector>,
      gr::Tags::MomentumDensity<3, Frame::Inertial, DataVector>>>
  matter_sources(const Scalar<DataVector>& conformal_factor,
                 const Scalar<DataVector>& lapse_times_conformal_factor,
                 const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
                 const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coords,
                 const double& center_of_mass, const double& angular_velocity,
                 const std::array<double, 2>& injection_energy) const noexcept;

  struct BackgroundFieldsCompute
      : ::Tags::Variables<tmpl::list<
            gr::Tags::EnergyDensity<DataVector>,
            gr::Tags::StressTrace<DataVector>,
            gr::Tags::MomentumDensity<3, Frame::Inertial, DataVector>>>,
        db::ComputeTag {
    using base = ::Tags::Variables<tmpl::list<
        gr::Tags::EnergyDensity<DataVector>, gr::Tags::StressTrace<DataVector>,
        gr::Tags::MomentumDensity<3, Frame::Inertial, DataVector>>>;
    using argument_tags =
        tmpl::list<::Tags::AnalyticSolution<NeutronStarBinary>,
                   Xcts::Tags::ConformalFactor<DataVector>,
                   Xcts::Tags::LapseTimesConformalFactor<DataVector>,
                   gr::Tags::Shift<3, Frame::Inertial, DataVector>,
                   ::Tags::Coordinates<3, Frame::Inertial>,
                   orbital::Tags::CenterOfMass, orbital::Tags::AngularVelocity,
                   hydro::Tags::BinaryInjectionEnergy>;
    static db::item_type<base> function(
        const NeutronStarBinary& solution,
        const Scalar<DataVector>& conformal_factor,
        const Scalar<DataVector>& lapse_times_conformal_factor,
        const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
        const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coords,
        const double& center_of_mass, const double& angular_velocity,
        const std::array<double, 2>& injection_energy) noexcept {
      return solution.matter_sources(
          conformal_factor, lapse_times_conformal_factor, shift,
          inertial_coords, center_of_mass, angular_velocity, injection_energy);
    }
  };

  using inv_jacobian_tag =
      ::Tags::InverseJacobian<::Tags::ElementMap<3>,
                              ::Tags::Coordinates<3, Frame::Logical>>;
  using compute_tags = tmpl::list<
      ::Tags::DerivTensorCompute<
          gr::Tags::Shift<3, Frame::Inertial, DataVector>, inv_jacobian_tag>,
      BackgroundFieldsCompute>;

  using observe_fields =
      tmpl::list<gr::Tags::EnergyDensity<DataVector>,
                 gr::Tags::StressTrace<DataVector>,
                 gr::Tags::MomentumDensity<3, Frame::Inertial, DataVector>>;

 private:
  friend bool operator==(const NeutronStarBinary& lhs,
                         const NeutronStarBinary& rhs) noexcept;

  double separation_ = std::numeric_limits<double>::signaling_NaN();
  double eccentricity_ = std::numeric_limits<double>::signaling_NaN();
  std::array<double, 2> central_rest_mass_densities_{
      {std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN()}};
  double polytropic_constant_ = std::numeric_limits<double>::signaling_NaN();
  double polytropic_exponent_ = std::numeric_limits<double>::signaling_NaN();
  EquationsOfState::PolytropicFluid<true> equation_of_state_{};
  std::array<Xcts::Solutions::TovStar, 2> stars_{};
  Xcts::Solutions::Vacuum vacuum_{};
  std::array<double, 2> star_centers_{};
};

bool operator!=(const NeutronStarBinary& lhs,
                const NeutronStarBinary& rhs) noexcept;

}  // namespace AnalyticData
}  // namespace Xcts
