// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class Poisson::FirstOrderSystem

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Elliptic/BoundaryConditions/AnalyticSolution.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/Systems/Poisson/BoundaryConditions/Robin.hpp"
#include "Elliptic/Systems/Poisson/Equations.hpp"
#include "Elliptic/Systems/Poisson/Geometry.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/TMPL.hpp"

namespace Poisson {

/*!
 * \brief The Poisson equation formulated as a set of coupled first-order PDEs.
 *
 * \details This system formulates the Poisson equation \f$-\Delta_\gamma u(x) =
 * f(x)\f$ on a background metric \f$\gamma_{ij}\f$ as the set of coupled
 * first-order PDEs
 *
 * \f[
 * -\partial_i \gamma^{ij} v_j(x) - \Gamma^i_{ij}\gamma^{jk}v_k = f(x) \\
 * -\partial_i u(x) + v_i(x) = 0
 * \f]
 *
 * where we have chosen the field gradient as an auxiliary variable \f$v_i\f$
 * and where \f$\Gamma^i_{jk}=\frac{1}{2}\gamma^{il}\left(\partial_j\gamma_{kl}
 * +\partial_k\gamma_{jl}-\partial_l\gamma_{jk}\right)\f$ are the Christoffel
 * symbols of the second kind of the background metric \f$\gamma_{ij}\f$. The
 * background metric \f$\gamma_{ij}\f$ and the Christoffel symbols derived from
 * it are assumed to be independent of the variables \f$u\f$ and \f$v_i\f$, i.e.
 * constant throughout an iterative elliptic solve.
 *
 * This scheme also goes by the name of _mixed_ or _flux_ formulation (see e.g.
 * \cite Arnold2002). The reason for the latter name is that we can write the
 * set of coupled first-order PDEs in flux-form
 *
 * \f[
 * -\partial_i F^i_A + S_A = f_A(x)
 * \f]
 *
 * by choosing the fluxes and sources in terms of the system variables
 * \f$u(x)\f$ and \f$v_i(x)\f$ as
 *
 * \f{align*}
 * F^i_u &= \gamma^{ij} v_j(x) \\
 * S_u &= -\Gamma^i_{ij}\gamma^{jk}v_k \\
 * f_u &= f(x) \\
 * F^i_{v_j} &= u \delta^i_j \\
 * S_{v_j} &= v_j \\
 * f_{v_j} &= 0 \text{.}
 * \f}
 *
 * Note that we use the system variables to index the fluxes and sources, which
 * we also do in the code by using DataBox tags.
 * Also note that we have defined the _fixed sources_ \f$f_A\f$ as those source
 * terms that are independent of the system variables.
 *
 * The fluxes and sources simplify significantly when the background metric is
 * flat and we employ Cartesian coordinates so \f$\gamma_{ij} = \delta_{ij}\f$
 * and \f$\Gamma^i_{jk} = 0\f$. Set the template parameter `BackgroundGeometry`
 * to `Poisson::Geometry::FlatCartesian` to specialise the system for this case.
 * Set it to `Poisson::Geometry::Curved` for the general case.
 */
template <size_t Dim, typename InvMetricTag, typename ChristoffelContractedTag>
struct FirstOrderSystem {
 private:
  using field = Tags::Field;
  using field_gradient =
      ::Tags::deriv<field, tmpl::size_t<Dim>, Frame::Inertial>;

 public:
  static constexpr size_t volume_dim = Dim;

  // The physical fields to solve for
  using primal_fields = tmpl::list<field>;
  using auxiliary_fields = tmpl::list<field_gradient>;

  // Tags for the first-order fluxes. We just use the standard `Flux` prefix
  // because the fluxes don't have symmetries and we don't need to give them a
  // particular meaning.
  using primal_fluxes =
      tmpl::list<::Tags::Flux<field, tmpl::size_t<Dim>, Frame::Inertial>>;

  // The variable-independent background fields in the equations
  using background_fields =
      tmpl::conditional_t<std::is_same_v<InvMetricTag, void>, tmpl::list<>,
                          tmpl::list<InvMetricTag, ChristoffelContractedTag>>;
  using inv_metric_tag = InvMetricTag;

  // The system equations formulated as fluxes and sources
  using fluxes_computer = Fluxes<Dim, InvMetricTag>;
  using sources_computer = Sources<Dim, ChristoffelContractedTag>;

  // The supported boundary conditions. Boundary conditions can be
  // factory-created from this base class.
  using boundary_conditions_base =
      elliptic::BoundaryConditions::BoundaryCondition<
          Dim, tmpl::list<elliptic::BoundaryConditions::Registrars::
                              AnalyticSolution<FirstOrderSystem>,
                          Poisson::BoundaryConditions::Registrars::Robin<Dim>>>;
};
}  // namespace Poisson
