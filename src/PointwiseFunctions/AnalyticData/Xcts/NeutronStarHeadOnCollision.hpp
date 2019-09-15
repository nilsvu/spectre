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

class NeutronStarHeadOnCollision {
 public:
  using equation_of_state_type = EquationsOfState::PolytropicFluid<true>;

  struct Separation {
    using type = double;
    static constexpr OptionString help = {
        "The coordinate separation of the binary at apoapsis."};
    static type lower_bound() noexcept { return 0.; }
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

  using options = tmpl::list<Separation, CentralDensities, PolytropicConstant,
                             PolytropicExponent>;
  static std::string name() noexcept { return "HeadOnCollision"; }

  static constexpr OptionString help = {
      "Two neutron stars with a polytropic EOS and given central densities "
      "that collide head-on."};

  NeutronStarHeadOnCollision() = default;
  NeutronStarHeadOnCollision(const NeutronStarHeadOnCollision& /*rhs*/) =
      delete;
  NeutronStarHeadOnCollision& operator=(
      const NeutronStarHeadOnCollision& /*rhs*/) = delete;
  NeutronStarHeadOnCollision(NeutronStarHeadOnCollision&& /*rhs*/) noexcept =
      default;
  NeutronStarHeadOnCollision& operator=(
      NeutronStarHeadOnCollision&& /*rhs*/) noexcept = default;
  ~NeutronStarHeadOnCollision() = default;

  NeutronStarHeadOnCollision(double separation,
                             std::array<double, 2> central_densities,
                             double polytropic_constant,
                             double polytropic_exponent) noexcept;

  template <typename DataType>
  using RadialVariables = DataType;  // placeholder

  const std::array<double, 2>& star_centers() const noexcept {
    return star_centers_;
  }

  /// Retrieve a collection of variables at coordinates `x`
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataType, 3>& x,
                                         tmpl::list<Tags...> /*meta*/) const
      noexcept {
    return {get<Tags>(variables(x, tmpl::list<Tags>{}))...};
  }

  template <typename Tag, typename DataType>
  db::item_type<Tag> superposed_tov(const tnsr::I<DataType, 3>& x) const
      noexcept {
    std::array<tnsr::I<DataType, 3, Frame::Inertial>, 2> x_centered_on_stars{
        {x, x}};
    get<0>(x_centered_on_stars[0]) -= star_centers_[0];
    get<0>(x_centered_on_stars[1]) -= star_centers_[1];
    const Variables<tmpl::list<Tag>> superposed_vars =
        variables_from_tagged_tuple(
            stars_[0].variables(x_centered_on_stars[0], tmpl::list<Tag>{})) +
        variables_from_tagged_tuple(
            stars_[1].variables(x_centered_on_stars[1], tmpl::list<Tag>{})) -
        variables_from_tagged_tuple(vacuum_.variables(x, tmpl::list<Tag>{}));
    // // Convert back to TaggedTuple
    // tuples::TaggedTuple<Tag> superposed_tuple{};
    // tmpl::for_each<tmpl::list<Tag>>(
    //     [&superposed_tuple, &superposed_vars ](auto tag_v) noexcept {
    //       using tag = tmpl::type_from<decltype(tag_v)>;
    //       get<tag>(superposed_tuple) = get<tag>(superposed_vars);
    //     });
    return get<Tag>(superposed_vars);
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
        tmpl::list<::Tags::AnalyticSolutionComputer<NeutronStarHeadOnCollision>,
                   ::Tags::Coordinates<3, Frame::Inertial>>;
    static db::item_type<base> function(
        const NeutronStarHeadOnCollision& solution,
        const tnsr::I<DataVector, 3, Frame::Inertial>&
            inertial_coords) noexcept {
      return variables_from_tagged_tuple(solution.variables(
          inertial_coords, db::get_variables_tags_list<base>{}));
    }
  };

  using compute_tags = tmpl::list<BackgroundFieldsCompute>;

 private:
  friend bool operator==(const NeutronStarHeadOnCollision& lhs,
                         const NeutronStarHeadOnCollision& rhs) noexcept;

  template <typename DataType>
  using ConformalFactorGradient =
      Xcts::Tags::ConformalFactorGradient<3, Frame::Inertial, DataType>;

  template <typename DataType>
  using LapseTimesConformalFactorGradient =
      Xcts::Tags::LapseTimesConformalFactorGradient<3, Frame::Inertial,
                                                    DataType>;

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

#define MY_LIST                                            \
  BOOST_PP_TUPLE_TO_LIST(                                  \
      6, (Xcts::Tags::ConformalFactor<DataType>,           \
          ConformalFactorGradient<DataType>,               \
          Xcts::Tags::LapseTimesConformalFactor<DataType>, \
          LapseTimesConformalFactorGradient<DataType>,     \
          gr::Tags::EnergyDensity<DataType>, gr::Tags::StressTrace<DataType>))

  BOOST_PP_LIST_FOR_EACH(FUNC_DECL, _, MY_LIST)
#undef MY_LIST
#undef FUNC_DECL

  double separation_ = std::numeric_limits<double>::signaling_NaN();
  std::array<double, 2> central_rest_mass_densities_{
      {std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN()}};
  double polytropic_constant_ = std::numeric_limits<double>::signaling_NaN();
  double polytropic_exponent_ = std::numeric_limits<double>::signaling_NaN();
  EquationsOfState::PolytropicFluid<true> equation_of_state_{};
  std::array<Xcts::Solutions::TovStar, 2> stars_{};
  Xcts::Solutions::Vacuum vacuum_{};
  std::array<double, 2> star_centers_{
      {std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN()}};
};

bool operator!=(const NeutronStarHeadOnCollision& lhs,
                const NeutronStarHeadOnCollision& rhs) noexcept;

}  // namespace AnalyticData
}  // namespace Xcts
