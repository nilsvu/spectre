// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <ostream>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Xcts {
namespace Solutions {

enum class SchwarzschildCoordinates {
  /*!
   * \brief Isotropic Schwarzschild coordinates
   *
   * Eq. 1.60 in \cite BaumgarteShapiro
   *
   * Horizon at r = M / 2
   *
   * Maximally sliced (same as Schwarzschild), singularity-avoiding,
   */
  Isotropic,

  /*!
   * \brief Kerr-Schild (or ingoing Eddington-Finkelstein) coordinates with an
   * isotropic radial transformation
   */
  KerrSchildIsotropic
};

std::ostream& operator<<(std::ostream& os,
                         const SchwarzschildCoordinates& coords) noexcept;

template <SchwarzschildCoordinates Coords>
class Schwarzschild {
 public:
  using options = tmpl::list<>;
  static constexpr OptionString help{
      "A Schwarzschild spacetime in general relativity"};

  Schwarzschild() = default;
  Schwarzschild(const Schwarzschild&) noexcept = delete;
  Schwarzschild& operator=(const Schwarzschild&) noexcept = delete;
  Schwarzschild(Schwarzschild&&) noexcept = default;
  Schwarzschild& operator=(Schwarzschild&&) noexcept = default;
  ~Schwarzschild() noexcept = default;

  double radius_at_horizon() const noexcept;

  // @{
  /// Retrieve variable at coordinates `x`
  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<Xcts::Tags::ConformalFactor<DataType>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<Xcts::Tags::ConformalFactor<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>,
                               tmpl::size_t<3>, Frame::Inertial>> /*meta*/)
      const noexcept -> tuples::TaggedTuple<
          ::Tags::deriv<Xcts::Tags::ConformalFactor<DataType>, tmpl::size_t<3>,
                        Frame::Inertial>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<Xcts::Tags::LapseTimesConformalFactor<DataType>> /*meta*/)
      const noexcept
      -> tuples::TaggedTuple<Xcts::Tags::LapseTimesConformalFactor<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
                               tmpl::size_t<3>, Frame::Inertial>> /*meta*/)
      const noexcept -> tuples::TaggedTuple<
          ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactor<DataType>,
                        tmpl::size_t<3>, Frame::Inertial>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<gr::Tags::Shift<3, Frame::Inertial, DataType>> /*meta*/) const
      noexcept
      -> tuples::TaggedTuple<gr::Tags::Shift<3, Frame::Inertial, DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<Xcts::Tags::ShiftStrain<3, Frame::Inertial,
                                                    DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<
          Xcts::Tags::ShiftStrain<3, Frame::Inertial, DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<::Tags::FixedSource<
                     Xcts::Tags::ConformalFactor<DataType>>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<
          ::Tags::FixedSource<Xcts::Tags::ConformalFactor<DataType>>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<::Tags::FixedSource<
                     Xcts::Tags::LapseTimesConformalFactor<DataType>>> /*meta*/)
      const noexcept -> tuples::TaggedTuple<
          ::Tags::FixedSource<Xcts::Tags::LapseTimesConformalFactor<DataType>>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<::Tags::FixedSource<
                     gr::Tags::Shift<3, Frame::Inertial, DataType>>> /*meta*/)
      const noexcept -> tuples::TaggedTuple<
          ::Tags::FixedSource<gr::Tags::Shift<3, Frame::Inertial, DataType>>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<gr::Tags::EnergyDensity<DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<gr::Tags::EnergyDensity<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<gr::Tags::StressTrace<DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<gr::Tags::StressTrace<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                 tmpl::list<gr::Tags::MomentumDensity<3, Frame::Inertial,
                                                      DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<
          gr::Tags::MomentumDensity<3, Frame::Inertial, DataType>>;
  // @}

  /// Retrieve a collection of variables at coordinates `x`
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<Tags...> /*meta*/) const noexcept {
    static_assert(sizeof...(Tags) > 1,
                  "The generic template will recurse infinitely if only one "
                  "tag is being retrieved.");
    return {tuples::get<Tags>(variables(x, tmpl::list<Tags>{}))...};
  }

  // clang-tidy: no pass by reference
  void pup(PUP::er& /* p */) noexcept {}  // NOLINT
};

template <SchwarzschildCoordinates Coords>
SPECTRE_ALWAYS_INLINE bool operator==(
    const Schwarzschild<Coords>& /*lhs*/,
    const Schwarzschild<Coords>& /*rhs*/) noexcept {
  return true;
}

template <SchwarzschildCoordinates Coords>
SPECTRE_ALWAYS_INLINE bool operator!=(
    const Schwarzschild<Coords>& /*lhs*/,
    const Schwarzschild<Coords>& /*rhs*/) noexcept {
  return false;
}

}  // namespace Solutions
}  // namespace Xcts
