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

template <size_t Dim>
struct LinearizedFirstOrderHamiltonianSystem {
 private:
  using conformal_factor = Tags::ConformalFactor<DataVector>;
  using conformal_factor_gradient =
      Tags::ConformalFactorGradient<Dim, Frame::Inertial, DataVector>;

 public:
  static constexpr size_t volume_dim = Dim;

  using fields_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::Correction,
                         ::Tags::Variables<tmpl::list<
                             conformal_factor, conformal_factor_gradient>>>;

  using compute_fluxes = ComputeFirstOrderHamiltonianFluxes<
      Dim, db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>,
      LinearSolver::Tags::Operand<
          NonlinearSolver::Tags::Correction<conformal_factor>>,
      LinearSolver::Tags::Operand<
          NonlinearSolver::Tags::Correction<conformal_factor_gradient>>>;
  using compute_sources = ComputeFirstOrderLinearizedHamiltonianSources<
      Dim, db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>,
      LinearSolver::Tags::Operand<
          NonlinearSolver::Tags::Correction<conformal_factor>>,
      LinearSolver::Tags::Operand<
          NonlinearSolver::Tags::Correction<conformal_factor_gradient>>,
      conformal_factor>;

  // Boundary conditions
  // This interface will likely change with generalized boundary conditions
  using primal_variables = tmpl::list<LinearSolver::Tags::Operand<
      NonlinearSolver::Tags::Correction<conformal_factor>>>;
  using auxiliary_variables = tmpl::list<LinearSolver::Tags::Operand<
      NonlinearSolver::Tags::Correction<conformal_factor_gradient>>>;
};

template <size_t Dim>
struct FirstOrderHamiltonianSystem {
 private:
  using conformal_factor = Tags::ConformalFactor<DataVector>;
  using conformal_factor_gradient =
      Tags::ConformalFactorGradient<Dim, Frame::Inertial, DataVector>;

 public:
  static constexpr size_t volume_dim = Dim;

  // The physical fields to solve for
  using fields_tag = ::Tags::Variables<
      tmpl::list<conformal_factor, conformal_factor_gradient>>;

  using background_fields = tmpl::list<gr::Tags::EnergyDensity<DataVector>>;

  using compute_fluxes =
      ComputeFirstOrderHamiltonianFluxes<Dim, fields_tag, conformal_factor,
                                         conformal_factor_gradient>;
  using compute_sources =
      ComputeFirstOrderHamiltonianSources<Dim, fields_tag, conformal_factor,
                                          conformal_factor_gradient>;

  // Boundary conditions
  // This interface will likely change with generalized boundary conditions
  using primal_variables = tmpl::list<conformal_factor>;
  using auxiliary_variables = tmpl::list<conformal_factor_gradient>;
  using compute_analytic_fluxes = ComputeFirstOrderHamiltonianFluxes<
      Dim, db::add_tag_prefix<::Tags::Analytic, fields_tag>,
      ::Tags::Analytic<conformal_factor>,
      ::Tags::Analytic<conformal_factor_gradient>>;

  using linearized_system = LinearizedFirstOrderHamiltonianSystem<Dim>;

  // The tag of the operator to compute magnitudes on the manifold, e.g. to
  // normalize vectors on the faces of an element
  template <typename Tag>
  using magnitude_tag = ::Tags::EuclideanMagnitude<Tag>;
};

template <size_t Dim>
struct LinearizedFirstOrderHamiltonianAndLapseSystem {
 private:
  using conformal_factor = Tags::ConformalFactor<DataVector>;
  using conformal_factor_gradient =
      Tags::ConformalFactorGradient<Dim, Frame::Inertial, DataVector>;
  using lapse_times_conformal_factor =
      Tags::LapseTimesConformalFactor<DataVector>;
  using lapse_times_conformal_factor_gradient =
      Tags::LapseTimesConformalFactorGradient<Dim, Frame::Inertial, DataVector>;

 public:
  static constexpr size_t volume_dim = Dim;

  using fields_tag = db::add_tag_prefix<
      NonlinearSolver::Tags::Correction,
      ::Tags::Variables<tmpl::list<
          conformal_factor, lapse_times_conformal_factor,
          conformal_factor_gradient, lapse_times_conformal_factor_gradient>>>;

  using compute_fluxes = ComputeFirstOrderHamiltonianAndLapseFluxes<
      Dim, db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>,
      LinearSolver::Tags::Operand<
          NonlinearSolver::Tags::Correction<conformal_factor>>,
      LinearSolver::Tags::Operand<
          NonlinearSolver::Tags::Correction<conformal_factor_gradient>>,
      LinearSolver::Tags::Operand<
          NonlinearSolver::Tags::Correction<lapse_times_conformal_factor>>,
      LinearSolver::Tags::Operand<NonlinearSolver::Tags::Correction<
          lapse_times_conformal_factor_gradient>>>;
  using compute_sources = ComputeFirstOrderLinearizedHamiltonianAndLapseSources<
      Dim, db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>,
      LinearSolver::Tags::Operand<
          NonlinearSolver::Tags::Correction<conformal_factor>>,
      LinearSolver::Tags::Operand<
          NonlinearSolver::Tags::Correction<conformal_factor_gradient>>,
      LinearSolver::Tags::Operand<
          NonlinearSolver::Tags::Correction<lapse_times_conformal_factor>>,
      LinearSolver::Tags::Operand<NonlinearSolver::Tags::Correction<
          lapse_times_conformal_factor_gradient>>,
      conformal_factor, lapse_times_conformal_factor>;

  // Boundary conditions
  // This interface will likely change with generalized boundary conditions
  using primal_variables =
      tmpl::list<LinearSolver::Tags::Operand<
                     NonlinearSolver::Tags::Correction<conformal_factor>>,
                 LinearSolver::Tags::Operand<NonlinearSolver::Tags::Correction<
                     lapse_times_conformal_factor>>>;
  using auxiliary_variables =
      tmpl::list<LinearSolver::Tags::Operand<NonlinearSolver::Tags::Correction<
                     conformal_factor_gradient>>,
                 LinearSolver::Tags::Operand<NonlinearSolver::Tags::Correction<
                     lapse_times_conformal_factor_gradient>>>;
};

template <size_t Dim>
struct FirstOrderHamiltonianAndLapseSystem {
 private:
  using conformal_factor = Tags::ConformalFactor<DataVector>;
  using conformal_factor_gradient =
      Tags::ConformalFactorGradient<Dim, Frame::Inertial, DataVector>;
  using lapse_times_conformal_factor =
      Tags::LapseTimesConformalFactor<DataVector>;
  using lapse_times_conformal_factor_gradient =
      Tags::LapseTimesConformalFactorGradient<Dim, Frame::Inertial, DataVector>;

 public:
  static constexpr size_t volume_dim = Dim;

  // The physical fields to solve for
  using fields_tag = ::Tags::Variables<tmpl::list<
      conformal_factor, lapse_times_conformal_factor, conformal_factor_gradient,
      lapse_times_conformal_factor_gradient>>;

  using background_fields = tmpl::list<gr::Tags::EnergyDensity<DataVector>>;

  using compute_fluxes = ComputeFirstOrderHamiltonianAndLapseFluxes<
      Dim, fields_tag, conformal_factor, conformal_factor_gradient,
      lapse_times_conformal_factor, lapse_times_conformal_factor_gradient>;
  using compute_sources = ComputeFirstOrderHamiltonianAndLapseSources<
      Dim, fields_tag, conformal_factor, conformal_factor_gradient,
      lapse_times_conformal_factor, lapse_times_conformal_factor_gradient>;

  // Boundary conditions
  // This interface will likely change with generalized boundary conditions
  using primal_variables =
      tmpl::list<conformal_factor, lapse_times_conformal_factor>;
  using auxiliary_variables = tmpl::list<conformal_factor_gradient,
                                         lapse_times_conformal_factor_gradient>;
  using compute_analytic_fluxes = ComputeFirstOrderHamiltonianAndLapseFluxes<
      Dim, db::add_tag_prefix<::Tags::Analytic, fields_tag>,
      ::Tags::Analytic<conformal_factor>,
      ::Tags::Analytic<conformal_factor_gradient>,
      ::Tags::Analytic<lapse_times_conformal_factor>,
      ::Tags::Analytic<lapse_times_conformal_factor_gradient>>;

  using linearized_system = LinearizedFirstOrderHamiltonianAndLapseSystem<Dim>;

  // The tag of the operator to compute magnitudes on the manifold, e.g. to
  // normalize vectors on the faces of an element
  template <typename Tag>
  using magnitude_tag = ::Tags::EuclideanMagnitude<Tag>;
};

template <size_t Dim>
struct LinearizedHamiltonianAndLapseSystem {
 private:
  using conformal_factor = Tags::ConformalFactor<DataVector>;
  using lapse_times_conformal_factor =
      Tags::LapseTimesConformalFactor<DataVector>;

 public:
  static constexpr size_t volume_dim = Dim;

  using fields_tag =
      db::add_tag_prefix<NonlinearSolver::Tags::Correction,
                         ::Tags::Variables<tmpl::list<
                             conformal_factor, lapse_times_conformal_factor>>>;

  using compute_fluxes = ComputeHamiltonianAndLapseFluxes<
      Dim, db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>,
      LinearSolver::Tags::Operand<
          NonlinearSolver::Tags::Correction<conformal_factor>>,
      LinearSolver::Tags::Operand<
          NonlinearSolver::Tags::Correction<lapse_times_conformal_factor>>>;
  using compute_sources = ComputeLinearizedHamiltonianAndLapseSources<
      Dim, db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>,
      LinearSolver::Tags::Operand<
          NonlinearSolver::Tags::Correction<conformal_factor>>,
      LinearSolver::Tags::Operand<
          NonlinearSolver::Tags::Correction<lapse_times_conformal_factor>>,
      conformal_factor, lapse_times_conformal_factor>;
  using compute_normal_fluxes = ComputeHamiltonianAndLapseNormalFluxes<
      Dim, db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>,
      LinearSolver::Tags::Operand<
          NonlinearSolver::Tags::Correction<conformal_factor>>,
      LinearSolver::Tags::Operand<
          NonlinearSolver::Tags::Correction<lapse_times_conformal_factor>>>;
  using compute_normal_fluxes_of_fields =
      ComputeHamiltonianAndLapseNormalFluxes<
          Dim, fields_tag, NonlinearSolver::Tags::Correction<conformal_factor>,
          NonlinearSolver::Tags::Correction<lapse_times_conformal_factor>>;
};

template <size_t Dim>
struct HamiltonianAndLapseSystem {
 private:
  using conformal_factor = Tags::ConformalFactor<DataVector>;
  using lapse_times_conformal_factor =
      Tags::LapseTimesConformalFactor<DataVector>;

 public:
  static constexpr size_t volume_dim = Dim;

  // The physical fields to solve for
  using fields_tag = ::Tags::Variables<
      tmpl::list<conformal_factor, lapse_times_conformal_factor>>;

  using background_fields = tmpl::list<gr::Tags::EnergyDensity<DataVector>>;

  using compute_fluxes =
      ComputeHamiltonianAndLapseFluxes<Dim, fields_tag, conformal_factor,
                                       lapse_times_conformal_factor>;
  using compute_sources =
      ComputeHamiltonianAndLapseSources<Dim, fields_tag, conformal_factor,
                                        lapse_times_conformal_factor>;
  using compute_normal_fluxes =
      ComputeHamiltonianAndLapseNormalFluxes<Dim, fields_tag, conformal_factor,
                                             lapse_times_conformal_factor>;
  using compute_normal_fluxes_of_fields =
      ComputeHamiltonianAndLapseNormalFluxes<Dim, fields_tag, conformal_factor,
                                             lapse_times_conformal_factor>;

  // Boundary conditions
  // This interface will likely change with generalized boundary conditions
  using compute_analytic_fluxes = ComputeHamiltonianAndLapseFluxes<
      Dim, db::add_tag_prefix<::Tags::Analytic, fields_tag>,
      ::Tags::Analytic<conformal_factor>,
      ::Tags::Analytic<lapse_times_conformal_factor>>;

  using linearized_system = LinearizedHamiltonianAndLapseSystem<Dim>;

  // The tag of the operator to compute magnitudes on the manifold, e.g. to
  // normalize vectors on the faces of an element
  template <typename Tag>
  using magnitude_tag = ::Tags::EuclideanMagnitude<Tag>;
};

}  // namespace Xcts
