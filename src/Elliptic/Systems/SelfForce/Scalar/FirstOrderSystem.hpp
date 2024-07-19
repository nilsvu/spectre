// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/Protocols/FirstOrderSystem.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/Equations.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/Tags.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarSelfForce {

/*!
 * \brief Self-force of a scalar charge on a Kerr background.
 *
 * In this formulation we solve a 2D elliptic equation for the m-mode field
 * $\Psi_m$, following \cite Osburn2022 . The two dimensions are a radial and
 * angular coordinate, specifically the tortoise radius $r_\star$ and
 * $\cos\theta$ (note that \cite Osburn2022 uses $\theta$).
 *
 * We currently specialize to circular equatorial motion, so the scalar charge
 * remains fixed at $r=r_0$, $\theta=\pi/2$, and $\phi=\Omega t$ with
 * \begin{equation}
 * \Omega &= \frac{1}{a + \sqrt{r_0^3/M}
 * \text{.}
 * \end{equation}
 *
 * We write the equation (2.9) in \cite Osburn2022 in first-order flux form
 * \begin{equation}
 * -\partial_i F^i + \beta \Psi_m + \gamma_i F^i = S_m
 * \end{equation}
 * with the flux
 * \begin{equation}
 * F^i = \{\partial_{r_\star}, \alpha \partial_{\cos\theta}\} \Psi_m
 * \end{equation}
 * and coefficients $\alpha$, $\beta$, and $\gamma^i$.
 */
struct FirstOrderSystem
    : tt::ConformsTo<elliptic::protocols::FirstOrderSystem> {
  static constexpr size_t volume_dim = 2;

  using primal_fields = tmpl::list<Tags::MMode>;
  using primal_fluxes =
      tmpl::list<::Tags::Flux<Tags::MMode, tmpl::size_t<2>, Frame::Inertial>>;

  using background_fields = tmpl::list<Tags::Alpha, Tags::Beta, Tags::Gamma>;
  using inv_metric_tag = void;

  using fluxes_computer = Fluxes;
  using sources_computer = Sources;
  using modify_boundary_data = ModifyBoundaryData;

  using boundary_conditions_base =
      elliptic::BoundaryConditions::BoundaryCondition<2>;
};

}  // namespace ScalarSelfForce
