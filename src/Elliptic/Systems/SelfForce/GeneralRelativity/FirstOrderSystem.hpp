// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/Protocols/FirstOrderSystem.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/Equations.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/Tags.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace GrSelfForce {

/*!
 * \brief Self-force of a mass on a Kerr background.
 *
 * Currently specialized to circular equatorial motion, so the charge remains
 * fixed at $r=r_0$, $\theta=\pi/2$, and $\phi=\Omega t$ with
 *
 * \begin{equation}
 * \Omega &= \frac{1}{a + \sqrt{r_0^3/M}
 * \text{.}
 * \end{equation}
 */
struct FirstOrderSystem
    : tt::ConformsTo<elliptic::protocols::FirstOrderSystem> {
  static constexpr size_t volume_dim = 3;

  using primal_fields = tmpl::list<Tags::MModeRe, Tags::MModeIm>;
  using primal_fluxes =
      tmpl::list<::Tags::Flux<Tags::MModeRe, tmpl::size_t<3>, Frame::Inertial>,
                 ::Tags::Flux<Tags::MModeIm, tmpl::size_t<3>, Frame::Inertial>>;

  using background_fields =
      tmpl::list<Tags::Alpha, Tags::Beta, Tags::GammaRstar, Tags::GammaTheta>;
  using inv_metric_tag = void;

  using fluxes_computer = Fluxes;
  using sources_computer = Sources;
  using modify_boundary_data = ModifyBoundaryData;

  using boundary_conditions_base =
      elliptic::BoundaryConditions::BoundaryCondition<3>;
};

}  // namespace GrSelfForce
