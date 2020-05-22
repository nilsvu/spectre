// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Direction.hpp"
#include "Elliptic/BoundaryConditions.hpp"
#include "Elliptic/Systems/Elasticity/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
class DataVector;
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Elasticity::Solutions {
/*!
 * \brief The solution for a half-space mirror deformed by a laser beam.
 *
 * \details This solution is mapping (via the fluctuation dissipation theorem)
 * thermal noise to an elasticity problem where a normally incident and
 * axisymmetric laser beam with a Gaussian beam profile acts on the face of a
 * semi-infinite mirror. Here we assume the face to be at \f$z = 0\f$ and the
 * material to extend to \f$+\infty\f$ in the z-direction as well as for the
 * mirror diameter to be comparetively large to the `beam width`. The mirror
 * material is characterized by an isotropic homogeneous constitutive relation
 * \f$Y^{ijkl}\f$ (see
 * `Elasticity::ConstitutiveRelations::IsotropicHomogeneous`). In this scenario,
 * the auxiliary elastic problem has an applied pressure distribution equal to
 * the laser beam intensity profile (see Eq. (11.94) and Eq. (11.95) in
 * \cite ThorneBlandford2017 with F = 1 and the time dependency dropped)
 *
 * \f{align}
 * T^{zr} &= T^{rz} = 0 \\
 * T^{zz} &= \frac{e^{-\frac{r^2}{r_0^2}}}{\pi r_0^2} \text{.}
 * \f}
 *
 * in the form of a Neumann boundary condition to the face of the mirror. We
 * find that this stress in cylinder coordinates is produced by the displacement
 * field
 *
 * \f{align}
 * \xi_{r} &= \frac{1}{2 \mu} \int_0^{\infty} dk J_1(kr)e^{(-kz)}\left(1 -
 * \frac{\lambda + 2\mu}{\lambda + \mu} + kz \right) \tilde{p}(k) \\
 * \xi_{\phi} &= 0 \\
 * \xi_{z} &=  \frac{1}{2 \mu} \int_0^{\infty} dk J_0(kr)e^{(-kz)}\left(1 +
 * \frac{\mu}{\lambda + \mu} + kz \right) \tilde{p}(k)
 * \f}
 *
 * and the strain
 *
 * \f{align}
 * \Theta &= \frac{1}{2 \mu} \int_0^{\infty} dk
 * J_0(kr) k e^{(-kz)}\left(\frac{-2\mu}{\lambda + \mu}\right) \tilde{p}(k) \\
 * S_{rr} &= \Theta - S_{\phi\phi} - S_{zz} \\
 * S_{\phi\phi} &= \frac{\xi_{r}}{r} \\
 * S_{(rz)} &= -\frac{1}{2 \mu} \int_0^{\infty} dk J_1(kr) k e^{(-kz)}\left(kz
 * \right) \tilde{p}(k) \\
 * S_{zz} &= \frac{1}{2 \mu} \int_0^{\infty} dk
 * J_0(kr) k e^{(-kz)}\left(-\frac{\mu}{\lambda + \mu} - kz \right) \tilde{p}(k)
 * \f}
 *
 * (see Eqs. (11 a) - (11 c) and (13 a) - (13 e), with (13 c) swapped in favor
 * of (12 c) in \cite Lovelace2007tn), where \f$\tilde{p}(k)= \frac{1}{2\pi}
 * e^{(\frac{kr}{2})^2}\f$ is the Hankel-Transform of the lasers intensity
 * profile and \f$ \Theta = \mathrm{Tr}(S)\f$ the materials expansion.
 *
 */
class HalfSpaceMirror {
 public:
  static constexpr size_t volume_dim = 3;

  using constitutive_relation_type =
      Elasticity::ConstitutiveRelations::IsotropicHomogeneous<3>;

  struct BeamWidth {
    using type = double;
    static constexpr OptionString help{"The lasers beam width"};
    static type lower_bound() noexcept { return 0.0; }
  };

  struct Material {
    using type = constitutive_relation_type;
    static constexpr OptionString help{"The material properties of the beam"};
  };

  struct IntegrationIntervals {
    using type = size_t;
    static constexpr OptionString help{
        "Workspace size for numerical integrals. Increase if integrals fail to "
        "reach the prescribed tolerance at large distances relative to the "
        "beam width. The default values for workspace size and tolerances "
        "should accommodate distances of up to ~100 beam widths."};
    static type lower_bound() noexcept { return 1; }
    static type default_value() noexcept { return 350; }
  };

  struct AbsoluteTolerance {
    using type = double;
    static constexpr OptionString help{
        "Absolute tolerance for numerical integrals"};
    static type lower_bound() noexcept { return 0.; }
    static type default_value() noexcept { return 1e-12; }
  };

  struct RelativeTolerance {
    using type = double;
    static constexpr OptionString help{
        "Relative tolerance for numerical integrals"};
    static type lower_bound() noexcept { return 0.; }
    static type upper_bound() noexcept { return 1.; }
    static type default_value() noexcept { return 1e-10; }
  };

  using options = tmpl::list<BeamWidth, Material, IntegrationIntervals,
                             AbsoluteTolerance, RelativeTolerance>;
  static constexpr OptionString help{
      "A semi-infinite mirror on which a laser introduces stress perpendicular "
      "to the mirrors surface. The displacement then is the Hankel-Transform "
      "of the general solution multiplied by the beam profile"};

  HalfSpaceMirror() = default;
  HalfSpaceMirror(const HalfSpaceMirror&) noexcept = delete;
  HalfSpaceMirror& operator=(const HalfSpaceMirror&) noexcept = delete;
  HalfSpaceMirror(HalfSpaceMirror&&) noexcept = default;
  HalfSpaceMirror& operator=(HalfSpaceMirror&&) noexcept = default;
  ~HalfSpaceMirror() noexcept = default;

  HalfSpaceMirror(double beam_width,
                  constitutive_relation_type constitutive_relation,
                  size_t integration_intervals = 350,
                  double absolute_tolerance = 1e-12,
                  double relative_tolerance = 1e-10) noexcept;

  const constitutive_relation_type& constitutive_relation() const noexcept {
    return constitutive_relation_;
  }

  // @{
  /// Retrieve variable at coordinates `x`
  auto variables(const tnsr::I<DataVector, 3>& x,
                 tmpl::list<Tags::Displacement<3>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<Tags::Displacement<3>>;

  auto variables(const tnsr::I<DataVector, 3>& x,
                 tmpl::list<Tags::Strain<3>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<Tags::Strain<3>>;

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
  friend bool operator==(const HalfSpaceMirror& lhs,
                         const HalfSpaceMirror& rhs) noexcept;

  double beam_width_{std::numeric_limits<double>::signaling_NaN()};
  constitutive_relation_type constitutive_relation_{};
  size_t integration_intervals_{};
  double absolute_tolerance_{};
  double relative_tolerance_{};
};

bool operator!=(const HalfSpaceMirror& lhs,
                const HalfSpaceMirror& rhs) noexcept;

}  // namespace Elasticity::Solutions
