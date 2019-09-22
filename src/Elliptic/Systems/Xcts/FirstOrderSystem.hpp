// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Elliptic/Systems/Xcts/Equations.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
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

template <size_t Dim, Equations EnabledEquations>
struct LinearizedFirstOrderSystem;

template <size_t Dim, Equations EnabledEquations>
struct FirstOrderSystem {
 private:
  using conformal_factor = Tags::ConformalFactor<DataVector>;
  using conformal_factor_gradient =
      ::Tags::deriv<conformal_factor, tmpl::size_t<Dim>, Frame::Inertial>;
  using lapse_times_conformal_factor =
      Tags::LapseTimesConformalFactor<DataVector>;
  using lapse_times_conformal_factor_gradient =
      ::Tags::deriv<lapse_times_conformal_factor, tmpl::size_t<Dim>,
                    Frame::Inertial>;
  using shift = gr::Tags::Shift<Dim, Frame::Inertial, DataVector>;
  using shift_strain = Tags::ShiftStrain<Dim, Frame::Inertial, DataVector>;

 public:
  static constexpr size_t volume_dim = Dim;

  // The physical fields to solve for
  using primal_fields = tmpl::append<
      tmpl::list<conformal_factor>,
      tmpl::conditional_t<
          EnabledEquations == Equations::HamiltonianAndLapse or
              EnabledEquations == Equations::HamiltonianLapseAndShift,
          tmpl::list<lapse_times_conformal_factor>, tmpl::list<>>,
      tmpl::conditional_t<EnabledEquations ==
                              Equations::HamiltonianLapseAndShift,
                          tmpl::list<shift>, tmpl::list<>>>;
  using auxiliary_fields = tmpl::append<
      tmpl::list<conformal_factor_gradient>,
      tmpl::conditional_t<
          EnabledEquations == Equations::HamiltonianAndLapse or
              EnabledEquations == Equations::HamiltonianLapseAndShift,
          tmpl::list<lapse_times_conformal_factor_gradient>, tmpl::list<>>,
      tmpl::conditional_t<EnabledEquations ==
                              Equations::HamiltonianLapseAndShift,
                          tmpl::list<shift_strain>, tmpl::list<>>>;
  using fields_tag =
      ::Tags::Variables<tmpl::append<primal_fields, auxiliary_fields>>;

  // The variables to compute bulk contributions and fluxes for.
  using variables_tag = fields_tag;
  using primal_variables = primal_fields;
  using auxiliary_variables = auxiliary_fields;

  using fluxes = Fluxes<Dim, EnabledEquations>;
  using sources = Sources<Dim, EnabledEquations>;

  using linearized_system = LinearizedFirstOrderSystem<Dim, EnabledEquations>;

  // The tag of the operator to compute magnitudes on the manifold, e.g. to
  // normalize vectors on the faces of an element
  template <typename Tag>
  using magnitude_tag = ::Tags::EuclideanMagnitude<Tag>;
};

template <size_t Dim, Equations EnabledEquations>
struct LinearizedFirstOrderSystem {
 private:
  using nonlinear_system = FirstOrderSystem<Dim, EnabledEquations>;

 public:
  static constexpr size_t volume_dim = Dim;

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

  using fluxes = Fluxes<Dim, EnabledEquations>;
  using sources = LinearizedSources<Dim, EnabledEquations>;

  // The tag of the operator to compute magnitudes on the manifold, e.g. to
  // normalize vectors on the faces of an element
  template <typename Tag>
  using magnitude_tag = ::Tags::EuclideanMagnitude<Tag>;
};

}  // namespace Xcts
