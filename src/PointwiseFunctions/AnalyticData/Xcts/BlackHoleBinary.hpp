// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <ostream>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Elliptic/BoundaryConditions.hpp"
#include "Elliptic/Protocols.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Schwarzschild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Vacuum.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Xcts::AnalyticData {

enum class BackgroundSpacetime {
  FlatMaximallySliced,
  FlatKerrSchildIsotropicSliced,
  SuperposedKerrSchild,
  SuperposedHarmonic
};

std::ostream& operator<<(std::ostream& os,
                         const BackgroundSpacetime& coords) noexcept;

template <BackgroundSpacetime Background>
class BlackHoleBinary {
 public:
  struct MassRatio {
    using type = double;
    static constexpr Options::String help = "Mass ratio";
    static double lower_bound() noexcept { return 1.; }
  };
  // TODO: Spins
  struct Separation {
    using type = double;
    static constexpr Options::String help =
        "Coordinate distance between the black hole centers";
    static double lower_bound() noexcept { return 0.; }
  };
  struct AngularVelocity {
    using type = double;
    static constexpr Options::String help = "Orbital angular velocity";
  };

  using options = tmpl::list<MassRatio, Separation, AngularVelocity>;
  static constexpr Options::String help{
      "Black hole binary initial data in general relativity"};

  BlackHoleBinary() = default;
  BlackHoleBinary(const BlackHoleBinary&) noexcept = default;
  BlackHoleBinary& operator=(const BlackHoleBinary&) noexcept = default;
  BlackHoleBinary(BlackHoleBinary&&) noexcept = default;
  BlackHoleBinary& operator=(BlackHoleBinary&&) noexcept = default;
  ~BlackHoleBinary() noexcept = default;

  BlackHoleBinary(double mass_ratio, double separation,
                  double angular_velocity) noexcept;

  // @{
  /// Retrieve variable at coordinates `x`
  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<gr::Tags::TraceExtrinsicCurvature<DataType>> /*meta*/)
      const noexcept
      -> tuples::TaggedTuple<gr::Tags::TraceExtrinsicCurvature<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataType>,
                               tmpl::size_t<3>, Frame::Inertial>> /*meta*/)
      const noexcept -> tuples::TaggedTuple<
          ::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataType>,
                        tmpl::size_t<3>, Frame::Inertial>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<Xcts::Tags::ShiftBackground<
                     DataType, 3, Frame::Inertial>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<
          Xcts::Tags::ShiftBackground<DataType, 3, Frame::Inertial>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<
          ::Tags::FixedSource<Xcts::Tags::ConformalFactor<DataType>>> /*meta*/)
      const noexcept -> tuples::TaggedTuple<
          ::Tags::FixedSource<Xcts::Tags::ConformalFactor<DataType>>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<::Tags::FixedSource<
                     Xcts::Tags::LapseTimesConformalFactor<DataType>>> /*meta*/)
      const noexcept -> tuples::TaggedTuple<
          ::Tags::FixedSource<Xcts::Tags::LapseTimesConformalFactor<DataType>>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<::Tags::FixedSource<Xcts::Tags::ShiftExcess<
                     DataType, 3, Frame::Inertial>>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<::Tags::FixedSource<
          Xcts::Tags::ShiftExcess<DataType, 3, Frame::Inertial>>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<
          ::Tags::Initial<Xcts::Tags::ConformalFactor<DataType>>> /*meta*/)
      const noexcept -> tuples::TaggedTuple<
          ::Tags::Initial<Xcts::Tags::ConformalFactor<DataType>>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<::Tags::Initial<
                     ::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>,
                                   tmpl::size_t<3>, Frame::Inertial>>> /*meta*/)
      const noexcept -> tuples::TaggedTuple<
          ::Tags::Initial<::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>,
                                        tmpl::size_t<3>, Frame::Inertial>>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<::Tags::Initial<
                     Xcts::Tags::LapseTimesConformalFactor<DataType>>> /*meta*/)
      const noexcept -> tuples::TaggedTuple<
          ::Tags::Initial<Xcts::Tags::LapseTimesConformalFactor<DataType>>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<::Tags::Initial<
          ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
                        tmpl::size_t<3>, Frame::Inertial>>> /*meta*/)
      const noexcept -> tuples::TaggedTuple<::Tags::Initial<
          ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
                        tmpl::size_t<3>, Frame::Inertial>>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<::Tags::Initial<Xcts::Tags::ShiftExcess<
                     DataType, 3, Frame::Inertial>>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<::Tags::Initial<
          Xcts::Tags::ShiftExcess<DataType, 3, Frame::Inertial>>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<::Tags::Initial<Xcts::Tags::ShiftStrain<
                     3, Frame::Inertial, DataType>>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<::Tags::Initial<
          Xcts::Tags::ShiftStrain<3, Frame::Inertial, DataType>>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<gr::Tags::EnergyDensity<DataType>> /*meta*/)
      const noexcept -> tuples::TaggedTuple<gr::Tags::EnergyDensity<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<gr::Tags::StressTrace<DataType>> /*meta*/)
      const noexcept -> tuples::TaggedTuple<gr::Tags::StressTrace<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<gr::Tags::MomentumDensity<
                     3, Frame::Inertial, DataType>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<
          gr::Tags::MomentumDensity<3, Frame::Inertial, DataType>>;
  // @}

  /// Retrieve a collection of variables at coordinates `x`
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<Tags...> /*meta*/) const noexcept {
    static_assert(sizeof...(Tags) > 1, "The requested tag is not implemented.");
    return {tuples::get<Tags>(variables(x, tmpl::list<Tags>{}))...};
  }

  void pup(PUP::er& p) noexcept {  // NOLINT
    p | mass_ratio_;
    p | separation_;
    p | angular_velocity_;
    p | isolated_solutions_;
  }

  double mass_ratio() const noexcept { return mass_ratio_; }
  double separation() const noexcept { return separation_; }
  double angular_velocity() const noexcept { return angular_velocity_; }

 private:
  double mass_ratio_;
  double separation_;
  double angular_velocity_;

  std::array<
      tmpl::conditional_t<
          Background == BackgroundSpacetime::FlatMaximallySliced,
          Xcts::Solutions::Schwarzschild<
              Xcts::Solutions::SchwarzschildCoordinates::Isotropic>,
          Xcts::Solutions::Schwarzschild<
              Xcts::Solutions::SchwarzschildCoordinates::KerrSchildIsotropic>>,
      2>
      isolated_solutions_;
  Xcts::Solutions::Vacuum vacuum_{};

  template <typename DataType>
  std::array<tnsr::I<DataType, 3, Frame::Inertial>, 2>
  isolated_solution_centered_coordinates(
      const tnsr::I<DataType, 3, Frame::Inertial>& x) const noexcept {
    auto x_centered_left = x;
    auto x_centered_right = x;
    get<0>(x_centered_left) += separation_ / 2.;
    get<0>(x_centered_right) -= separation_ / 2.;
    return {{std::move(x_centered_left), std::move(x_centered_right)}};
  }

  template <typename Tag, bool RadialDamping = false, typename DataType>
  typename Tag::type superposition(
      const tnsr::I<DataType, 3, Frame::Inertial>& x) const noexcept {
    const auto x_centered = isolated_solution_centered_coordinates(x);
    const auto left_data = get<Tag>(
        isolated_solutions_[0].variables(x_centered[0], tmpl::list<Tag>{}));
    const auto right_data = get<Tag>(
        isolated_solutions_[1].variables(x_centered[1], tmpl::list<Tag>{}));
    const auto vacuum_data = get<Tag>(vacuum_.variables(x, tmpl::list<Tag>{}));
    auto superposition = make_with_value<typename Tag::type>(x, 0.);
    auto screen = make_with_value<DataType>(*x.begin(), 1.);
    // if constexpr (RadialDamping) {
    //   const auto r = get(magnitude(x));
    //   screen -= smoothstep<1>(3. * separation_, 5. * separation_, r);
    // }
    // TODO: When adding a gaussian, also make sure to correctly superpose
    // derivatives
    for (size_t i = 0; i < superposition.size(); ++i) {
      superposition[i] = vacuum_data[i] +
                         screen * (left_data[i] - vacuum_data[i]) +
                         screen * (right_data[i] - vacuum_data[i]);
    }
    return superposition;
  }
};

template <BackgroundSpacetime Background>
SPECTRE_ALWAYS_INLINE bool operator==(
    const BlackHoleBinary<Background>& lhs,
    const BlackHoleBinary<Background>& rhs) noexcept {
  return lhs.mass_ratio() == rhs.mass_ratio() and
         lhs.separation() == rhs.separation() and
         lhs.angular_velocity() == rhs.angular_velocity();
}

template <BackgroundSpacetime Background>
SPECTRE_ALWAYS_INLINE bool operator!=(
    const BlackHoleBinary<Background>& lhs,
    const BlackHoleBinary<Background>& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace Xcts::AnalyticData
