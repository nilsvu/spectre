// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>
#include <string>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/Systems/Xcts/Geometry.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace Xcts::BoundaryConditions {
namespace detail {

template <Xcts::Geometry ConformalGeometry>
struct ApparentHorizonImpl {
  static constexpr Options::String help =
      "Impose the boundary is a quasi-equilibrium apparent horizon. The "
      "boundary must be a coordinate-sphere.";

  struct Spin {
    using type = std::array<double, 3>;
    static constexpr Options::String help = "The spin parameter on the surface";
  };

  using options = tmpl::list<Spin>;

  ApparentHorizonImpl() = default;
  ApparentHorizonImpl(const ApparentHorizonImpl&) noexcept = default;
  ApparentHorizonImpl& operator=(const ApparentHorizonImpl&) noexcept = default;
  ApparentHorizonImpl(ApparentHorizonImpl&&) noexcept = default;
  ApparentHorizonImpl& operator=(ApparentHorizonImpl&&) noexcept = default;
  ~ApparentHorizonImpl() noexcept = default;

  ApparentHorizonImpl(const std::array<double, 3>& spin) noexcept;

  const std::array<double, 3>& spin() const noexcept;

  using argument_tags = tmpl::flatten<tmpl::list<
      ::Tags::Normalized<
          domain::Tags::UnnormalizedFaceNormal<3, Frame::Inertial>>,
      domain::Tags::Coordinates<3, Frame::Inertial>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>,
      Tags::ShiftBackground<DataVector, 3, Frame::Inertial>,
      Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<DataVector, 3,
                                                              Frame::Inertial>,
      tmpl::conditional_t<ConformalGeometry == Xcts::Geometry::Curved,
                          tmpl::list<Tags::InverseConformalMetric<
                                         DataVector, 3, Frame::Inertial>,
                                     Tags::ConformalChristoffelSecondKind<
                                         DataVector, 3, Frame::Inertial>>,
                          tmpl::list<>>>>;
  using volume_tags = tmpl::list<>;

  void apply(
      const gsl::not_null<Scalar<DataVector>*> conformal_factor,
      const gsl::not_null<Scalar<DataVector>*> lapse_times_conformal_factor,
      const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
      const gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
      const gsl::not_null<Scalar<DataVector>*>
          n_dot_lapse_times_conformal_factor_gradient,
      const gsl::not_null<tnsr::I<DataVector, 3>*>
          n_dot_longitudinal_shift_excess,
      const tnsr::i<DataVector, 3>& face_normal,
      const tnsr::I<DataVector, 3>& x,
      const Scalar<DataVector>& extrinsic_curvature_trace,
      const tnsr::I<DataVector, 3>& shift_background,
      const tnsr::II<DataVector, 3>& longitudinal_shift_background)
      const noexcept;

  void apply(
      const gsl::not_null<Scalar<DataVector>*> conformal_factor,
      const gsl::not_null<Scalar<DataVector>*> lapse_times_conformal_factor,
      const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
      const gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
      const gsl::not_null<Scalar<DataVector>*>
          n_dot_lapse_times_conformal_factor_gradient,
      const gsl::not_null<tnsr::I<DataVector, 3>*>
          n_dot_longitudinal_shift_excess,
      const tnsr::i<DataVector, 3>& face_normal,
      const tnsr::I<DataVector, 3>& x,
      const Scalar<DataVector>& extrinsic_curvature_trace,
      const tnsr::I<DataVector, 3>& shift_background,
      const tnsr::II<DataVector, 3>& longitudinal_shift_background,
      const tnsr::II<DataVector, 3>& inv_conformal_metric,
      const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind)
      const noexcept;

  using argument_tags_linearized = tmpl::flatten<tmpl::list<
      ::Tags::Normalized<
          domain::Tags::UnnormalizedFaceNormal<3, Frame::Inertial>>,
      domain::Tags::Coordinates<3, Frame::Inertial>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>,
      Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<DataVector, 3,
                                                              Frame::Inertial>,
      Tags::ConformalFactor<DataVector>,
      Tags::LapseTimesConformalFactor<DataVector>,
      ::Tags::NormalDotFlux<Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>,
      tmpl::conditional_t<ConformalGeometry == Xcts::Geometry::Curved,
                          tmpl::list<Tags::InverseConformalMetric<
                                         DataVector, 3, Frame::Inertial>,
                                     Tags::ConformalChristoffelSecondKind<
                                         DataVector, 3, Frame::Inertial>>,
                          tmpl::list<>>>>;
  using volume_tags_linearized = tmpl::list<>;

  void apply_linearized(
      const gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
      const gsl::not_null<Scalar<DataVector>*>
          lapse_times_conformal_factor_correction,
      const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess_correction,
      const gsl::not_null<Scalar<DataVector>*>
          n_dot_conformal_factor_gradient_correction,
      const gsl::not_null<Scalar<DataVector>*>
          n_dot_lapse_times_conformal_factor_gradient_correction,
      const gsl::not_null<tnsr::I<DataVector, 3>*>
          n_dot_longitudinal_shift_excess_correction,
      const tnsr::i<DataVector, 3>& face_normal,
      const tnsr::I<DataVector, 3>& x,
      const Scalar<DataVector>& extrinsic_curvature_trace,
      const tnsr::II<DataVector, 3>& longitudinal_shift_background,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& lapse_times_conformal_factor,
      const tnsr::I<DataVector, 3>& n_dot_longitudinal_shift_excess)
      const noexcept;

  void apply_linearized(
      const gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
      const gsl::not_null<Scalar<DataVector>*>
          lapse_times_conformal_factor_correction,
      const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess_correction,
      const gsl::not_null<Scalar<DataVector>*>
          n_dot_conformal_factor_gradient_correction,
      const gsl::not_null<Scalar<DataVector>*>
          n_dot_lapse_times_conformal_factor_gradient_correction,
      const gsl::not_null<tnsr::I<DataVector, 3>*>
          n_dot_longitudinal_shift_excess_correction,
      const tnsr::i<DataVector, 3>& face_normal,
      const tnsr::I<DataVector, 3>& x,
      const Scalar<DataVector>& extrinsic_curvature_trace,
      const tnsr::II<DataVector, 3>& longitudinal_shift_background,
      const Scalar<DataVector>& conformal_factor,
      const Scalar<DataVector>& lapse_times_conformal_factor,
      const tnsr::I<DataVector, 3>& n_dot_longitudinal_shift_excess,
      const tnsr::II<DataVector, 3>& inv_conformal_metric,
      const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind)
      const noexcept;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept;

 private:
  std::array<double, 3> spin_;
};

template <Xcts::Geometry ConformalGeometry>
bool operator==(const ApparentHorizonImpl<ConformalGeometry>& lhs,
                const ApparentHorizonImpl<ConformalGeometry>& rhs) noexcept;

template <Xcts::Geometry ConformalGeometry>
bool operator!=(const ApparentHorizonImpl<ConformalGeometry>& lhs,
                const ApparentHorizonImpl<ConformalGeometry>& rhs) noexcept;

}  // namespace detail

// The following implements the registration and factory-creation mechanism

/// \cond
template <Xcts::Geometry ConformalGeometry, typename Registrars>
struct ApparentHorizon;

namespace Registrars {
template <Xcts::Geometry ConformalGeometry>
struct ApparentHorizon {
  template <typename Registrars>
  using f = BoundaryConditions::ApparentHorizon<ConformalGeometry, Registrars>;
};
}  // namespace Registrars
/// \endcond

/*!
 * \brief Impose the surface is a quasi-equilibrium apparent horizon. The
 * boundary must be a coordinate-sphere.
 *
 * These boundary conditions on the conformal factor \f$\psi\f$, the lapse
 * \f$\alpha\f$ and the shift \f$\beta^i\f$ impose the surface is an apparent
 * horizon, i.e. that the expansion on the surface vanishes \f$\Theta=0\f$.
 * Specifically, we impose:
 *
 * \f{align}
 * \bar{s}^k\bar{D}_k\psi &= \frac{\psi}{4}\left(
 * \psi^2 J - \bar{m}^{ij}\bar{D}_i\bar{s}_j\right)
 * \\
 * \bar{s}^k\bar{D}_k(\alpha\psi) &= 0
 * \\
 * \beta_\mathrm{excess}^i &= \frac{\alpha}{\psi^2}\bar{s}^i
 * + \epsilon^{ijk}Omega^\mathrm{spin}_j\hat{n}_k
 * \f}
 *
 * following section 7.2 of \cite Pfeiffer2005zm, section 12.3.2 of
 * \cite BaumgarteShapiro or section B.1 of \cite Varma2018sqd. In these
 * equations \f$\bar{s}_i\f$ is the conformal surface normal to the apparent
 * horizon, \f$\bar{m}^ij=\bar{\gamma}^{ij}-\bar{s}^i\bar{s}^j\f$ is the induced
 * conformal surface metric (denoted \f$\tilde{h}^{ij}\f$ in
 * \cite Pfeiffer2005zm) and \f$\bar{D}\f$ is the covariant derivative w.r.t. to
 * the conformal metric \f$\bar{\gamma}_{ij}\f$. To incur a spin on the apparent
 * horizon we can freely choose the parameter
 * \f$\boldsymbol{\Omega}_\mathrm{spin}\f$. It is defined w.r.t. a unit sphere
 * in Cartesian coordinates aligned with the simulation's "inertial" Cartesian
 * coordinates. \f$\hat{n}_i\f$ denotes the normal to this unit sphere.
 *
 * Note that the quasi-equilibrium conditions don't restrict the boundary
 * condition for the lapse, so we are free to change the condition on
 * \f$\alpha\psi\f$ and still obtain an apparent horizon at the surface, albeit
 * in different coordinates.
 *
 * \par Negative-expansion boundary conditions:
 * Support for negative-expansion boundary conditions following section B.2 of
 * \cite Varma2018sqd can be added here by taking the mass and the
 * coordinate-system of a Kerr solution as additional parameters and computing
 * its expansion at the excision surface. Choosing an excision surface _within_
 * the apparent horizon of the Kerr solution will result in a negative expansion
 * that can be added to the boundary condition for the conformal factor.
 */
template <Xcts::Geometry ConformalGeometry,
          typename Registrars =
              tmpl::list<Registrars::ApparentHorizon<ConformalGeometry>>>
class ApparentHorizon
    : public elliptic::BoundaryConditions::BoundaryCondition<3, Registrars>,
      public detail::ApparentHorizonImpl<ConformalGeometry> {
 private:
  using Base = elliptic::BoundaryConditions::BoundaryCondition<3, Registrars>;

 public:
  ApparentHorizon() = default;
  ApparentHorizon(const ApparentHorizon&) noexcept = default;
  ApparentHorizon& operator=(const ApparentHorizon&) noexcept = default;
  ApparentHorizon(ApparentHorizon&&) noexcept = default;
  ApparentHorizon& operator=(ApparentHorizon&&) noexcept = default;
  ~ApparentHorizon() noexcept = default;

  using detail::ApparentHorizonImpl<ConformalGeometry>::ApparentHorizonImpl;

  /// \cond
  explicit ApparentHorizon(CkMigrateMessage* m) noexcept : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ApparentHorizon);
  /// \endcond

  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> get_clone()
      const noexcept override {
    return std::make_unique<ApparentHorizon>(*this);
  }

  void pup(PUP::er& p) noexcept override {
    Base::pup(p);
    detail::ApparentHorizonImpl<ConformalGeometry>::pup(p);
  }
};

/// \cond
template <Xcts::Geometry ConformalGeometry, typename Registrars>
PUP::able::PUP_ID ApparentHorizon<ConformalGeometry, Registrars>::my_PUP_ID =
    0;  // NOLINT
/// \endcond

}  // namespace Xcts::BoundaryConditions
