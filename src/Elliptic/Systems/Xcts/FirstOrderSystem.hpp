// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Elliptic/Systems/Xcts/Equations.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "ParallelAlgorithms/NonlinearSolver/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
template <typename>
class Variables;
}  // namespace Tags
/// \endcond

namespace Xcts {

template <Equations EnabledEquations>
struct LinearizedFirstOrderSystem;

/*!
 * \brief The Extended Conformal Thin Sandwich (XCTS) decomposition of the
 * Einstein constraint equations, formulated as a set of coupled first-order
 * partial differential equations
 *
 * See \ref Xcts for details on the XCTS equations. This system introduces as
 * auxiliary variables the conformal factor gradient \f$v_i=\partial_i\psi\f$,
 * the symmetric shift strain \f$B_{ij}=\partial_{(i}\beta_{j)}\f$ and the
 * gradient of the lapse times the conformal factor
 * \f$w_i=\partial_i\left(\alpha\psi\right)\f$. With these variables, and under
 * the simplifying assumptions listed in \ref Xcts, the longitudinal shift
 * reduced to \f$\left(L\beta\right)^{ij}=2B^{ij} -
 * \frac{2}{3}\gamma^{ij}\mathrm{Tr}(B)\f$ and thus the XCTS equations reduce to
 * this set of coupled first-order equations:
 *
 * \f{align}
 * -\partial_i v^i = \frac{1}{8}\frac{\psi^7}{\left(\alpha\psi\right)^2}
 * \mathrm{Tr}(B)^2 + 2\pi\psi^5\rho
 * \\
 * -\partial_i \left(L\beta\right)^{ij} = -\left(L\beta\right)^{ij}\left(
 * \frac{w_i}{\alpha\psi} - 7 \frac{v_i}{\psi}\right)
 * - 16\pi\left(\alpha\psi\right)\psi^3 S^i
 * \\
 * -\partial_i w^i = -\frac{7}{8}\frac{\psi^6}{\alpha\psi}\mathrm{Tr}(B)^2
 * - 2\pi\left(\alpha\psi\right)\psi^4\left(\rho + 2S\right)
 * \f}
 *
 * In the first-order flux-formulation, where we write the equations as
 *
 * \f[
 * -\partial_i F^i_A + S_A = f_A(x)
 * \f]
 *
 * (see also `Poisson::FirstOrderSystem`), the fluxes \f$F^i_A\f$, sources
 * \f$S_A\f$ and fixed-sources \f$f_A\f$ are:
 *
 * \f{align}
 * F^i_\psi &= v^i \\
 * S_\psi &= -\frac{1}{8}\frac{\psi^7}{\left(\alpha\psi\right)^2}
 * \mathrm{Tr}(B)^2 - 2\pi\psi^5\rho \\
 * f_\psi &= 0 \\
 * F^i_{\beta^j} &= \left(L\beta\right)^{ij} \\
 * S_{\beta^i} &= \left(L\beta\right)^{ij}\left(
 * \frac{w_j}{\alpha\psi} - 7 \frac{v_j}{\psi}\right)
 * + 16\pi\left(\alpha\psi\right)\psi^3 S^i \\
 * f_{\beta^i} &= 0 \\
 * F^i_{\alpha\psi} &= w^i \\
 * S_{\alpha\psi} &= \frac{7}{8}\frac{\psi^6}{\alpha\psi}\mathrm{Tr}(B)^2
 * + 2\pi\left(\alpha\psi\right)\psi^4\left(\rho + 2S\right) \\
 * f_{\alpha\beta} &= 0 \\
 * F^i_{v_j} &= \delta^i_j \psi \\
 * S_{v_j} &= v_j \\
 * f_{v_j} &= 0 \\
 * F^i_{B_{jk}} &= \delta^i_{(j} \beta_{k)} \\
 * S_{B_{jk}} &= B_{jk} \\
 * f_{B_{jk}} &= 0 \\
 * F^i_{w_j} &= \delta^i_j \alpha\psi \\
 * S_{w_j} &= w_j \\
 * f_{w_j} &= 0 \\
 * \f}
 *
 * Note that the Hamiltonian and lapse constraints are just Poisson equations
 * with a nonlinear source, and the momentum constraint is an Elasticity
 * equation with a nonlinear source where the "displacement" is the shift vector
 * \f$\beta^i\f$, the symmetric "strain" is \f$B_{ij}\f$ and the "stress" is the
 * longitudinal shift \f$\left(L\beta\right)^{ij}\f$. For this reason we
 * occasionally refer to the momentum constraint as the "minimal distortion"
 * equation.
 *
 * The template parameter `EnabledEquations` selects which subset of the XCTS
 * equations is being solved.
 */
template <Equations EnabledEquations>
struct FirstOrderSystem {
 private:
  using conformal_factor = Tags::ConformalFactor<DataVector>;
  using conformal_factor_gradient =
      ::Tags::deriv<conformal_factor, tmpl::size_t<3>, Frame::Inertial>;
  using lapse_times_conformal_factor =
      Tags::LapseTimesConformalFactor<DataVector>;
  using lapse_times_conformal_factor_gradient =
      ::Tags::deriv<lapse_times_conformal_factor, tmpl::size_t<3>,
                    Frame::Inertial>;
  using shift = gr::Tags::Shift<3, Frame::Inertial, DataVector>;
  using shift_strain = Tags::ShiftStrain<3, Frame::Inertial, DataVector>;

 public:
  static constexpr size_t volume_dim = 3;

  // The physical fields to solve for
  using primal_fields = tmpl::flatten<tmpl::list<
      conformal_factor,
      tmpl::conditional_t<EnabledEquations == Equations::HamiltonianAndLapse or
                              EnabledEquations ==
                                  Equations::HamiltonianLapseAndShift,
                          lapse_times_conformal_factor, tmpl::list<>>,
      tmpl::conditional_t<EnabledEquations ==
                              Equations::HamiltonianLapseAndShift,
                          shift, tmpl::list<>>>>;
  using auxiliary_fields = tmpl::flatten<tmpl::list<
      conformal_factor_gradient,
      tmpl::conditional_t<EnabledEquations == Equations::HamiltonianAndLapse or
                              EnabledEquations ==
                                  Equations::HamiltonianLapseAndShift,
                          lapse_times_conformal_factor_gradient, tmpl::list<>>,
      tmpl::conditional_t<EnabledEquations ==
                              Equations::HamiltonianLapseAndShift,
                          shift_strain, tmpl::list<>>>>;
  using fields_tag =
      ::Tags::Variables<tmpl::append<primal_fields, auxiliary_fields>>;

  // The variables to compute bulk contributions and fluxes for.
  using variables_tag = fields_tag;
  using primal_variables = primal_fields;
  using auxiliary_variables = auxiliary_fields;

  using fluxes = Fluxes<EnabledEquations>;
  using sources = Sources<EnabledEquations>;

  using linearized_system = LinearizedFirstOrderSystem<EnabledEquations>;

  // The tag of the operator to compute magnitudes on the manifold, e.g. to
  // normalize vectors on the faces of an element
  template <typename Tag>
  using magnitude_tag = ::Tags::EuclideanMagnitude<Tag>;
};

template <Equations EnabledEquations>
struct LinearizedFirstOrderSystem {
 private:
  using nonlinear_system = FirstOrderSystem<EnabledEquations>;

 public:
  static constexpr size_t volume_dim = 3;

  // The physical fields to solve for
  using primal_fields =
      db::wrap_tags_in<NonlinearSolver::Tags::Correction,
                       typename nonlinear_system::primal_fields>;
  using auxiliary_fields =
      db::wrap_tags_in<NonlinearSolver::Tags::Correction,
                       typename nonlinear_system::auxiliary_fields>;
  using fields_tag = db::add_tag_prefix<NonlinearSolver::Tags::Correction,
                                        typename nonlinear_system::fields_tag>;

  // The variables to compute bulk contributions and fluxes for.
  using variables_tag =
      db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
  using primal_variables =
      db::wrap_tags_in<LinearSolver::Tags::Operand, primal_fields>;
  using auxiliary_variables =
      db::wrap_tags_in<LinearSolver::Tags::Operand, auxiliary_fields>;

  using fluxes = Fluxes<EnabledEquations>;
  using sources = LinearizedSources<EnabledEquations>;

  // The tag of the operator to compute magnitudes on the manifold, e.g. to
  // normalize vectors on the faces of an element
  template <typename Tag>
  using magnitude_tag = ::Tags::EuclideanMagnitude<Tag>;
};

}  // namespace Xcts
