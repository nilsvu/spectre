// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class Poisson::FirstOrderSystem

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Elliptic/Protocols.hpp"
#include "Elliptic/Systems/Poisson/Equations.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
template <typename>
class Variables;
template <typename, typename>
class div;
}  // namespace Tags

namespace LinearSolver {
namespace Tags {
template <typename>
struct Operand;
}  // namespace Tags
}  // namespace LinearSolver
/// \endcond

namespace Poisson {

template <size_t Dim>
struct SecondOrderSystem : elliptic::Protocols::SecondOrderSystem {
  static constexpr size_t volume_dim = Dim;

  // The physical fields to solve for
  using fields_tag = Tags::Variables<tmpl::list<Field>>;

  using compute_fluxes = ComputeSecondOrderFluxes<
      Dim, db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>,
      LinearSolver::Tags::Operand<Field>>;
  using compute_sources = ComputeSecondOrderSources<
      Dim, db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>,
      LinearSolver::Tags::Operand<Field>>;

  // Boundary conditions
  // This interface will likely change with generalized boundary conditions
  using compute_analytic_fluxes =
      ComputeSecondOrderFluxes<Dim,
                               db::add_tag_prefix<::Tags::Analytic, fields_tag>,
                               ::Tags::Analytic<Field>>;

  // The tag of the operator to compute magnitudes on the manifold, e.g. to
  // normalize vectors on the faces of an element
  template <typename Tag>
  using magnitude_tag = Tags::EuclideanMagnitude<Tag>;
};

/*!
 * \brief The Poisson equation formulated as a set of coupled first-order PDEs.
 *
 * \details This system formulates the Poisson equation \f$-\Delta u(x) =
 * f(x)\f$ as the set of coupled first-order PDEs
 *
 * \f[
 * -\nabla \cdot \boldsymbol{v}(x) = f(x) \\
 * -\nabla u(x) + \boldsymbol{v}(x) = 0
 * \f]
 *
 * where we make use of an auxiliary variable \f$\boldsymbol{v}\f$. This scheme
 * also goes by the name of _mixed_ or _flux_ formulation (see e.g.
 * \cite Arnold2002). The reason for the latter name is that we can write the
 * set of coupled first-order PDEs in flux-form
 *
 * \f[
 * -\partial_i F^i_A + S_A = f_A(x)
 * \f]
 *
 * by choosing the fluxes and sources in terms of the system variables
 * \f$u(x)\f$ and $\boldsymbol{v}(x)}\f$ as
 *
 * \f{align}
 * F^i_u &= v^i \quad F^i_{v^j} &= u \delta^{ij} \\
 * S_u &= 0 \quad S_{v^j} &= v^j
 * f_u &= f(x) \quad f_{v^j} &= 0 \text{.}
 * \f}
 *
 * Note that we use the system variables to index the fluxes and sources, which
 * we also do in the code by using the variables' DataBox tags.
 * Also note that we have defined \f$f_A\f$ as those source terms that are
 * independent of the system variables.
 *
 * The auxiliary variable \f$\boldsymbol{v}\f$ is treated on the same footing as
 * the field \f$u\f$ in this first-order formulation. This allows us to make use
 * of the DG architecture developed for coupled first-order PDEs, in particular
 * the flux communication and lifting code. It does, however, introduce
 * auxiliary degrees of freedom that can be avoided in the _primal formulation_.
 * Furthermore, the linear operator that represents the DG discretization for
 * this system is not symmetric (since no mass operator is applied) and has both
 * positive and negative eigenvalues. These properties further increase the
 * computational cost (see \ref LinearSolverGroup) and are remedied in the
 * primal formulation.
 */
template <size_t Dim>
struct FirstOrderSystem {
  static constexpr size_t volume_dim = Dim;

  // The physical fields to solve for
  using fields_tag = Tags::Variables<tmpl::list<Field, AuxiliaryField<Dim>>>;

  using compute_fluxes = ComputeFirstOrderFluxes<
      Dim, db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>,
      LinearSolver::Tags::Operand<Field>,
      LinearSolver::Tags::Operand<AuxiliaryField<Dim>>>;
  using compute_sources = ComputeFirstOrderSources<
      Dim, db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>,
      LinearSolver::Tags::Operand<Field>,
      LinearSolver::Tags::Operand<AuxiliaryField<Dim>>>;

  // Boundary conditions
  // This interface will likely change with generalized boundary conditions
  using primal_variables = tmpl::list<LinearSolver::Tags::Operand<Field>>;
  using compute_analytic_fluxes = ComputeFirstOrderFluxes<
      Dim, db::add_tag_prefix<::Tags::Analytic, fields_tag>,
      ::Tags::Analytic<Field>, ::Tags::Analytic<AuxiliaryField<Dim>>>;

  // The tag of the operator to compute magnitudes on the manifold, e.g. to
  // normalize vectors on the faces of an element
  template <typename Tag>
  using magnitude_tag = Tags::EuclideanMagnitude<Tag>;
};
}  // namespace Poisson
