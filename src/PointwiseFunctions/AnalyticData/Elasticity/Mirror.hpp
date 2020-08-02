// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
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

  using constitutive_relation_type =
      Elasticity::ConstitutiveRelations::IsotropicHomogeneous<3>;

  struct Material {
    using type = constitutive_relation_type;
    static constexpr Options::String help{
        "The material properties of the beam"};
  };

  using options = tmpl::list<Material>;
  static constexpr Options::String help{
      "A mirror on which a laser introduces stress perpendicular to the "
      "surface."};

  Mirror() = default;
  Mirror(const Mirror&) noexcept = default;
  Mirror& operator=(const Mirror&) noexcept = default;
  Mirror(Mirror&&) noexcept = default;
  Mirror& operator=(Mirror&&) noexcept = default;
  ~Mirror() noexcept = default;

  Mirror(constitutive_relation_type material) noexcept;

  const constitutive_relation_type& constitutive_relation() const noexcept {
    return constitutive_relation_;
  }

  // @{
  /// Retrieve variable at coordinates `x`
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

  // clang-tidy: no pass by reference
  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  friend bool operator==(const Mirror& lhs, const Mirror& rhs) noexcept;

  constitutive_relation_type constitutive_relation_{};
};

bool operator!=(const Mirror& lhs, const Mirror& rhs) noexcept;

}  // namespace Elasticity::AnalyticData
