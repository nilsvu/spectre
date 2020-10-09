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
#include "Elliptic/Systems/Xcts/BoundaryConditions/ApparentHorizon.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "ErrorHandling/Assert.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Xcts::BoundaryConditions {

template <typename Component, bool IsLinearization>
struct Binary;

namespace detail {
template <typename Component, typename = std::void_t<>>
struct LinearizedBinary {
  using type = void;
};
template <typename Component>
struct LinearizedBinary<Component,
                        std::void_t<typename Component::linearization>> {
  using type = Binary<typename Component::linearization, true>;
};
}  // namespace detail

template <typename Component, bool IsLinearization = false>
struct Binary {
 public:
  struct Positions {
    using type = std::array<double, 2>;
    static constexpr Options::String help{"Positions on the x-axis"};
  };

  struct Components {
    using type = std::array<Component, 2>;
    static constexpr Options::String help{"The two components of the binary"};
  };

  using options = tmpl::list<Positions, Components>;
  static constexpr Options::String help{
      "Binary initial data in general relativity"};
  static std::string name() noexcept {
    return "Binary" + Options::name<Component>();
  }

  using linearization = typename detail::LinearizedBinary<Component>::type;

  Binary() = default;
  Binary(const Binary&) noexcept = default;
  Binary& operator=(const Binary&) noexcept = default;
  Binary(Binary&&) noexcept = default;
  Binary& operator=(Binary&&) noexcept = default;
  ~Binary() noexcept = default;

  Binary(std::array<double, 2> positions,
         typename Components::type components) noexcept
      : positions_(std::move(positions)), components_(std::move(components)) {
    ASSERT(
        positions_[0] < 0. and positions_[1] > 0.,
        "Left position must be negative and right position must be positive");
  }

  template <typename Tag>
  elliptic::BoundaryCondition boundary_condition_type(
      const tnsr::I<DataVector, 3>& x, const Direction<3>& direction,
      Tag meta) const noexcept {
    const auto& component = gsl::at(components_, get<0>(x)[0] < 0 ? 0 : 1);
    auto x_centered = x;
    if (get<0>(x)[0] > 0) {
      get<0>(x_centered) -= positions_[1];
    } else {
      get<0>(x_centered) -= positions_[0];
    }
    return component.boundary_condition_type(x_centered, direction, meta);
  }

  using argument_tags = typename Component::argument_tags;
  using volume_tags = typename Component::volume_tags;

  template <typename... Args>
  void apply(
      const gsl::not_null<Scalar<DataVector>*> conformal_factor,
      const gsl::not_null<Scalar<DataVector>*> lapse_times_conformal_factor,
      const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
      const gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
      const gsl::not_null<Scalar<DataVector>*>
          n_dot_lapse_times_conformal_factor_gradient,
      const gsl::not_null<tnsr::I<DataVector, 3>*>
          n_dot_longitudinal_shift_excess,
      const tnsr::i<DataVector, 3>& inward_pointing_face_normal,
      const tnsr::I<DataVector, 3>& x, const Args&... args)
      const noexcept {
    const auto& component = gsl::at(components_, get<0>(x)[0] < 0 ? 0 : 1);
    auto x_centered = x;
    if (get<0>(x)[0] > 0) {
      get<0>(x_centered) -= positions_[1];
    } else {
      get<0>(x_centered) -= positions_[0];
    }
    component.apply(conformal_factor, lapse_times_conformal_factor,
                    shift_excess, n_dot_conformal_factor_gradient,
                    n_dot_lapse_times_conformal_factor_gradient,
                    n_dot_longitudinal_shift_excess,
                    inward_pointing_face_normal, x_centered, args...);
    // TODO: add COM correction to outer boundary shift
  }

  void pup(PUP::er& p) noexcept {
    p | positions_;
    p | components_;
  }

 private:
  std::array<double, 2> positions_;
  std::array<Component, 2> components_;
};

}  // namespace Xcts::BoundaryConditions
