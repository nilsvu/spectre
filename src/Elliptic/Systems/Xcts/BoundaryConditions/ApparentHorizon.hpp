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
#include "Options/Auto.hpp"
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

  struct Center {
    using type = std::array<double, 3>;
    static constexpr Options::String help =
        "The center of the coordinate sphere representing the apparent-horizon "
        "surface";
  };
  struct Spin {
    using type = std::array<double, 3>;
    static constexpr Options::String help =
        "The spin parameter 'Omega' on the surface. ";
  };
  struct Mass {
    using type = Options::Auto<double>;
    static constexpr Options::String help =
        "Mass of a corresponding Kerr solution. When you provide a mass, the "
        "corresponding Kerr solution's lapse at the excision boundary is "
        "imposed as a Dirichlet condition on the lapse. Alternatively, set the "
        "mass to 'Auto' to impose a zero von-Neumann boundary condition on the "
        "lapse. Note that the latter will not result in the standard "
        "Kerr-Schild slicing for a single black hole.";
  };

  using options = tmpl::list<Center, Spin, Mass>;

  ApparentHorizonImpl() = default;
  ApparentHorizonImpl(const ApparentHorizonImpl&) = default;
  ApparentHorizonImpl& operator=(const ApparentHorizonImpl&) = default;
  ApparentHorizonImpl(ApparentHorizonImpl&&) = default;
  ApparentHorizonImpl& operator=(ApparentHorizonImpl&&) = default;
  ~ApparentHorizonImpl() = default;

  ApparentHorizonImpl(std::array<double, 3> center, std::array<double, 3> spin,
                      std::optional<double> mass) noexcept;

  const std::array<double, 3>& center() const noexcept;
  const std::array<double, 3>& spin() const noexcept;
  const std::optional<double>& mass() const noexcept;

  using argument_tags = tmpl::flatten<tmpl::list<
      ::Tags::Normalized<
          domain::Tags::UnnormalizedFaceNormal<3, Frame::Inertial>>,
      ::Tags::deriv<domain::Tags::UnnormalizedFaceNormal<3, Frame::Inertial>,
                    tmpl::size_t<3>, Frame::Inertial>,
      ::Tags::Magnitude<
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
      const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
      const Scalar<DataVector>& face_normal_magnitude,
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
      const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
      const Scalar<DataVector>& face_normal_magnitude,
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
      ::Tags::deriv<domain::Tags::UnnormalizedFaceNormal<3, Frame::Inertial>,
                    tmpl::size_t<3>, Frame::Inertial>,
      ::Tags::Magnitude<
          domain::Tags::UnnormalizedFaceNormal<3, Frame::Inertial>>,
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
      const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
      const Scalar<DataVector>& face_normal_magnitude,
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
      const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
      const Scalar<DataVector>& face_normal_magnitude,
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
  std::array<double, 3> center_{};
  std::array<double, 3> spin_{};
  std::optional<double> mass_{};
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
 * horizon, i.e. that the expansion on the surface vanishes: \f$\Theta=0\f$.
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
  ApparentHorizon(const ApparentHorizon&) = default;
  ApparentHorizon& operator=(const ApparentHorizon&) = default;
  ApparentHorizon(ApparentHorizon&&) = default;
  ApparentHorizon& operator=(ApparentHorizon&&) = default;
  ~ApparentHorizon() = default;

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
