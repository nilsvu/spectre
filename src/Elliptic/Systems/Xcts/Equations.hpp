// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "DataStructures/VariablesHelpers.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Systems/Poisson/Equations.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"  // IWYU pragma: keep
#include "Options/Options.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
// IWYU pragma: no_forward_declare Tags::deriv
// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables

/// \cond
class DataVector;
template <size_t>
class Mesh;
namespace Tags {
template <typename>
struct Normalized;
template <typename>
struct NormalDotFlux;
template <typename>
struct NormalDotNumericalFlux;
}  // namespace Tags
namespace LinearSolver {
namespace Tags {
template <typename>
struct Operand;
}  // namespace Tags
}  // namespace LinearSolver
namespace NonlinearSolver {
namespace Tags {
template <typename>
struct Correction;
template <typename>
struct OperatorAppliedTo;
}  // namespace Tags
}  // namespace NonlinearSolver
namespace Xcts {
namespace Tags {
template <typename DataType>
struct ConformalFactor;
template <size_t Dim, typename Frame, typename DataType>
struct ConformalFactorGradient;
template <typename Tag>
struct Conformal;
}  // namespace Tags
}  // namespace Xcts
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Xcts {

template <size_t Dim>
struct ComputeFirstOrderNonlinearOperatorAction {
  using argument_tags = tmpl::list<
      // Solution quantities
      Xcts::Tags::ConformalFactor<DataVector>,
      Xcts::Tags::ConformalFactorGradient<Dim, Frame::Inertial, DataVector>,
      // Background quantities
      gr::Tags::EnergyDensity<DataVector>,
      // Domain quantities
      ::Tags::Mesh<Dim>,
      ::Tags::InverseJacobian<::Tags::ElementMap<Dim>,
                              ::Tags::Coordinates<Dim, Frame::Logical>>>;
  using const_global_cache_tags = tmpl::list<>;
  using return_tags =
      db::wrap_tags_in<NonlinearSolver::Tags::OperatorAppliedTo,
                       tmpl::list<Xcts::Tags::ConformalFactor<DataVector>,
                                  Xcts::Tags::ConformalFactorGradient<
                                      Dim, Frame::Inertial, DataVector>>>;
  static void apply(
      gsl::not_null<Scalar<DataVector>*> hamiltonian_divergence_operator,
      gsl::not_null<tnsr::I<DataVector, Dim>*> hamiltonian_auxiliary_operator,
      const Scalar<DataVector>& conformal_factor,
      const tnsr::I<DataVector, Dim>& conformal_factor_gradient,
      const Scalar<DataVector>& energy_density, const Mesh<Dim>& mesh,
      const InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>&
          inverse_jacobian) noexcept;
};

/*!
 * \brief The bulk contribution to the linear operator action for the
 * first order formulation of the XCTS equations.
 */
template <size_t Dim>
struct ComputeFirstOrderLinearOperatorAction {
  using argument_tags = tmpl::list<
      // Correction quantities
      LinearSolver::Tags::Operand<NonlinearSolver::Tags::Correction<
          Xcts::Tags::ConformalFactor<DataVector>>>,
      ::Tags::deriv<
          LinearSolver::Tags::Operand<NonlinearSolver::Tags::Correction<
              Xcts::Tags::ConformalFactor<DataVector>>>,
          tmpl::size_t<Dim>, Frame::Inertial>,
      LinearSolver::Tags::Operand<
          NonlinearSolver::Tags::Correction<Xcts::Tags::ConformalFactorGradient<
              Dim, Frame::Inertial, DataVector>>>,
      // Solution quantities
      Xcts::Tags::ConformalFactor<DataVector>,
      // Background quantities
      gr::Tags::EnergyDensity<DataVector>,
      // Domain quantities
      ::Tags::Mesh<Dim>,
      ::Tags::InverseJacobian<::Tags::ElementMap<Dim>,
                              ::Tags::Coordinates<Dim, Frame::Logical>>>;
  static void apply(
      gsl::not_null<Scalar<DataVector>*> hamiltonian_divergence_operator,
      gsl::not_null<tnsr::I<DataVector, Dim>*> hamiltonian_auxiliary_operator,
      const Scalar<DataVector>& conformal_factor_correction,
      const tnsr::i<DataVector, Dim>&
          computed_conformal_factor_gradient_correction,
      const tnsr::I<DataVector, Dim>&
          auxiliary_conformal_factor_gradient_correction,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& energy_density, const Mesh<Dim>& mesh,
      const InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>&
          inverse_jacobian) noexcept;
};

template <size_t Dim, typename ConformalFactorTag,
          typename ConformalFactorGradientTag>
struct ComputeFirstOrderNormalDotFluxes {
  using argument_tags =
      tmpl::list<ConformalFactorTag, ConformalFactorGradientTag,
                 ::Tags::Normalized<::Tags::UnnormalizedFaceNormal<Dim>>>;
  using return_tags = db::wrap_tags_in<
      ::Tags::NormalDotFlux,
      tmpl::list<ConformalFactorTag, ConformalFactorGradientTag>>;
  static void apply(
      gsl::not_null<Scalar<DataVector>*> hamiltonian_normal_dot_flux,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          hamiltonian_auxiliary_normal_dot_flux,
      const Scalar<DataVector>& conformal_factor,
      const tnsr::I<DataVector, Dim, Frame::Inertial>&
          conformal_factor_gradient,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          interface_unit_normal) noexcept;
};

template <size_t Dim, typename ConformalFactorTag,
          typename ConformalFactorGradientTag>
struct FirstOrderInternalPenaltyFlux {
 public:
  struct PenaltyParameter {
    using type = double;
    // Currently this is used as the full prefactor to the penalty term. When it
    // becomes possible to compute a measure of the size $h$ of an element and
    // the number of collocation points $p$ on both sides of the mortar, this
    // should be changed to be just the parameter multiplying $\frac{p^2}{h}$.
    static constexpr OptionString help = {
        "The prefactor to the penalty term of the flux."};
  };
  using options = tmpl::list<PenaltyParameter>;
  static constexpr OptionString help = {
      "Computes the internal penalty flux for an XCTS system."};

  FirstOrderInternalPenaltyFlux() = default;
  explicit FirstOrderInternalPenaltyFlux(double penalty_parameter)
      : penalty_parameter_(penalty_parameter),
        poisson_flux_(penalty_parameter) {}

  // clang-tidy: non-const reference
  void pup(PUP::er& p) noexcept {
    p | penalty_parameter_;
    if (p.isUnpacking()) {
      poisson_flux_ =
          Poisson::FirstOrderInternalPenaltyFlux<Dim>{penalty_parameter_};
    }
  }  // NOLINT

  struct NormalTimesConformalFactorFlux : db::SimpleTag {
    using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
    static std::string name() noexcept {
      return "NormalTimesConformalFactorFlux";
    }
  };

  struct NormalDotConformalFactorGradientFlux : db::SimpleTag {
    using type = Scalar<DataVector>;
    static std::string name() noexcept {
      return "NormalDotConformalFactorGradientFlux";
    }
  };

  // These tags are sliced to the interface of the element and passed to
  // `package_data` to provide the data needed to compute the numerical fluxes.
  using argument_tags = tmpl::list<
      ConformalFactorTag,
      ::Tags::deriv<ConformalFactorTag, tmpl::size_t<Dim>, Frame::Inertial>,
      ::Tags::Normalized<::Tags::UnnormalizedFaceNormal<Dim>>>;
  using return_tags = db::wrap_tags_in<
      ::Tags::NormalDotNumericalFlux,
      tmpl::list<ConformalFactorTag, ConformalFactorGradientTag>>;

  // This is the data needed to compute the numerical flux.
  // `SendBoundaryFluxes` calls `package_data` to store these tags in a
  // Variables. Local and remote values of this data are then combined in the
  // `()` operator.
  using package_tags =
      tmpl::list<ConformalFactorTag, NormalTimesConformalFactorFlux,
                 NormalDotConformalFactorGradientFlux>;

  // Following the packaged_data pointer, this function expects as arguments the
  // types in `argument_tags`.
  void package_data(gsl::not_null<Variables<package_tags>*> packaged_data,
                    const Scalar<DataVector>& conformal_factor,
                    const tnsr::i<DataVector, Dim, Frame::Inertial>&
                        conformal_factor_gradient,
                    const tnsr::i<DataVector, Dim, Frame::Inertial>&
                        interface_unit_normal) const noexcept;

  // This function combines local and remote data to the numerical fluxes.
  // The numerical fluxes as not-null pointers are the first arguments. The
  // other arguments are the packaged types for the interior side followed by
  // the packaged types for the exterior side.
  void operator()(
      gsl::not_null<Scalar<DataVector>*> hamiltonian_numerical_flux,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          hamiltonian_auxiliary_numerical_flux,
      const Scalar<DataVector>& conformal_factor_interior,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          normal_times_conformal_factor_interior,
      const Scalar<DataVector>& normal_dot_conformal_factor_gradient_interior,
      const Scalar<DataVector>& conformal_factor_exterior,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          minus_normal_times_conformal_factor_exterior,
      const Scalar<DataVector>&
          minus_normal_dot_conformal_factor_gradient_exterior) const noexcept;

  // This function computes the boundary contributions from Dirichlet boundary
  // conditions. This data is what remains to be added to the boundaries when
  // homogeneous (i.e. zero) boundary conditions are assumed in the calculation
  // of the numerical fluxes, but we wish to impose inhomogeneous (i.e. nonzero)
  // boundary conditions. Since this contribution does not depend on the
  // numerical field values, but only on the Dirichlet boundary data, it may be
  // added as contribution to the source of the elliptic systems. Then, it
  // remains to solve the homogeneous problem with the modified source.
  // The first arguments to this function are the boundary contributions to
  // compute as not-null pointers, in the order they appear in the
  // `system::fields_tag`. They are followed by the field values of the tags in
  // `system::impose_boundary_conditions_on_fields`. The last argument is the
  // normalized unit covector to the element face.
  void compute_dirichlet_boundary(
      gsl::not_null<Scalar<DataVector>*> hamiltonian_numerical_flux,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          hamiltonian_auxiliary_numerical_flux,
      const Scalar<DataVector>& dirichlet_conformal_factor,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& interface_unit_normal)
      const noexcept;

 private:
  double penalty_parameter_{};
  Poisson::FirstOrderInternalPenaltyFlux<Dim> poisson_flux_{};
};

}  // namespace Xcts
