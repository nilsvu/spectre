// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Elliptic/BoundaryConditions.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"
#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
class DataVector;
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Elasticity::AnalyticData {
class Mirror {
 public:
  static constexpr size_t volume_dim = 3;

  struct BeamWidth {
    using type = double;
    static constexpr Options::String help{"The lasers beam width"};
    static type lower_bound() noexcept { return 0.0; }
  };

  using options = tmpl::list<BeamWidth>;
  static constexpr Options::String help{
      "A mirror on which a laser introduces stress perpendicular to the "
      "surface."};

  Mirror() = default;
  Mirror(const Mirror&) noexcept = default;
  Mirror& operator=(const Mirror&) noexcept = default;
  Mirror(Mirror&&) noexcept = default;
  Mirror& operator=(Mirror&&) noexcept = default;
  ~Mirror() noexcept = default;

  Mirror(double beam_width) noexcept;

  // @{
  /// Retrieve variable at coordinates `x`
  auto boundary_variables(
      const tnsr::I<DataVector, 3>& x, const Direction<3>& direction,
      const tnsr::i<DataVector, 3>& face_normal,
      tmpl::list<Tags::MinusNormalDotStress<3>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<Tags::MinusNormalDotStress<3>>;

  auto variables(const tnsr::I<DataVector, 3>& x,
                 tmpl::list<Tags::Displacement<3>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<Tags::Displacement<3>>;

  static auto variables(
      const tnsr::I<DataVector, 3>& x,
      tmpl::list<::Tags::FixedSource<Tags::Displacement<3>>> /*meta*/) noexcept
      -> tuples::TaggedTuple<::Tags::FixedSource<Tags::Displacement<3>>>;
  // @}

  /// Retrieve a collection of variables at coordinates `x`
  template <typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataVector, 3>& x,
                                         tmpl::list<Tags...> /*meta*/) const
      noexcept {
    static_assert(sizeof...(Tags) > 1, "An unsupported Tag was requested.");
    return {tuples::get<Tags>(variables(x, tmpl::list<Tags>{}))...};
  }

  template <typename... Tags>
  tuples::TaggedTuple<Tags...> boundary_variables(
      const tnsr::I<DataVector, 3>& x, const Direction<3>& direction,
      const tnsr::i<DataVector, 3>& face_normal,
      tmpl::list<Tags...> /*meta*/) const noexcept {
    static_assert(sizeof...(Tags) > 1, "An unsupported Tag was requested.");
    return {tuples::get<Tags>(
        boundary_variables(x, direction, face_normal, tmpl::list<Tags>{}))...};
  }

  static elliptic::BoundaryCondition boundary_condition_type(
      const tnsr::I<DataVector, 3>& /*x*/, const Direction<3>& direction) {
    // Impose Dirichlet conditions on the side of the mirror that faces away
    // from the laser
    if (direction == Direction<3>::upper_zeta()) {
      return elliptic::BoundaryCondition::Dirichlet;
    } else {
      return elliptic::BoundaryCondition::Neumann;
    }
  }

  // clang-tidy: no pass by reference
  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  friend bool operator==(const Mirror& lhs, const Mirror& rhs) noexcept;

  double beam_width_{std::numeric_limits<double>::signaling_NaN()};
};

bool operator!=(const Mirror& lhs, const Mirror& rhs) noexcept;

}  // namespace Elasticity::AnalyticData
