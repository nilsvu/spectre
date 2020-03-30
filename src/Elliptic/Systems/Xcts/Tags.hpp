// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"

/*!
 * \ingroup EllipticSystemsGroup
 * \brief Items related to solving the Extended Conformal Thin Sandwich (XCTS)
 * decomposition of the Einstein constraint equations
 *
 * The XCTS equations
 *
 * \f{align}
 * \bar{D}^2 \psi - \frac{1}{8}\psi\bar{R} - \frac{1}{12}\psi^5 K^2 +
 * \frac{1}{8}\psi^{-7}\bar{A}_{ij}\bar{A}^{ij} = -2\pi\psi^5\rho
 * \\
 * \left(\bar{\Delta}_L\beta\right)^i - \left(\bar{L}\beta\right)^{ij}\bar{D}_j
 * \ln(\bar{\alpha}) = \bar{\alpha}\bar{D}_j\left(\bar{\alpha^{-1}\bar{u}^{ij}
 * \right) + \frac{4}{3}\bar{\alpha}\psi^6\bar{D}^i K + 16\pi\bar{\alpha}\psi^10
 * S^i
 * \\
 * \bar{D}^2\left(\alpha\psi\right) = \alpha\psi\left(
 * \frac{7}{8}\psi^{-8}\bar{A}_{ij}\bar{A}^{ij}
 * + \frac{5}{12}\psi^4 K^2 + \frac{1}{8}\bar{R}
 * + 2\pi\psi^4\left(\rho + 2S\right)\right)
 * - \psi^5\delta_t K + \psi^5\beta^i\bar{D}_i K
 * \f}
 *
 * See \cite BaumgarteShapiro, Box 3.3
 *
 * For conformal flatness \f$\gamma_{ij}=f_{ij}\f$, maximal slicing \f$K=0\f$
 * and preserving these conditions instantaneously in time
 * \f$\bar{u}_{ij}=0=\delta_t K\f$ in Cartesian coordinates
 *
 * \f{align}
 * \partial^i\partial_i\psi = -\frac{1}{8}\psi^{-7}\bar{A}_ij\bar{A}^ij -
 * 2\pi\psi^5\rho
 * \\
 * \partial^j\partial_j\beta^i + \frac{1}{3}\partial^i\partial_j\beta^j =
 * 2\bar{A}^{ij}\partial_j\left(\alpha\psi^{-6}\right) + 16\pi\alpha\psi^4 S^i
 * \\
 * \partial^i\partial_i\left(\alpha\psi\right) = \alpha\psi\left(
 * \frac{7}{8}\psi^{-8}\bar{A}_{ij}\bar{A}^{ij} + 2\pi\psi^4\left(\rho +
 * 2S\right)\right)
 * \f}
 *
 * See \cite BaumgarteShapiro, Eqns 3.119-3.121
 *
 * Under these assumptions we will also refer to the momentum constraint as
 * "minimal distortion" constraint, and to the lapse equation as the "maximal
 * slicing" constriant.
 */
namespace Xcts {
/// Tags related to the XCTS equations
namespace Tags {

/*!
 * \brief The conformal factor \f$\psi(x)\f$ that rescales the spatial metric
 * \f$\gamma_{ij}=\psi^4\bar{\gamma}_{ij}\f$.
 */
template <typename DataType>
struct ConformalFactor : db::SimpleTag {
  using type = Scalar<DataType>;
};

/*!
 * \brief The quantity `Tag` scaled by the `Xcts::Tags::ConformalFactor` to the
 * given `Power`
 */
template <typename Tag, int Power>
struct Conformal : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
  static constexpr int conformal_factor_power = Power;
};

/*!
 * \brief The conformally scaled spatial metric
 * \f$\bar{\gamma}_{ij}=\psi^{-4}\gamma_{ij}\f$, where \f$\psi\f$ is the
 * `Xcts::Tags::ConformalFactor` and \f$gamma_{ij}\f$ is the
 * `gr::Tags::SpatialMetric`
 */
template <typename DataType, size_t Dim, typename Frame>
using ConformalMetric =
    Conformal<gr::Tags::SpatialMetric<Dim, Frame, DataType>, -4>;

/*!
 * \brief The conformally scaled inverse spatial metric
 * \f$\bar{\gamma}^{ij}=\psi^{4}\gamma^{ij}\f$, where \f$\psi\f$ is the
 * `ConformalFactor` and \f$gamma\f$ is the `gr::Tags::SpatialMetric`
 */
template <typename DataType, size_t Dim, typename Frame>
using InverseConformalMetric =
    Conformal<gr::Tags::InverseSpatialMetric<Dim, Frame, DataType>, 4>;

/*!
 * \brief The product of lapse \f$\alpha(x)\f$ and conformal factor
 * \f$\psi(x)\f$
 *
 * This quantity is commonly used in formulations of the XCTS equations.
 */
template <typename DataType>
struct LapseTimesConformalFactor : db::SimpleTag {
  using type = Scalar<DataType>;
};

/*!
 * \brief The constant part \f$\beta^i_\mathrm{background}\f$ of the shift
 * \f$\beta^i=\beta^i_\mathrm{background} + \beta^i_\mathrm{excess}\f$
 *
 * \see `Xcts::Tags::ShiftExcess`
 */
template <typename DataType, size_t Dim, typename Frame>
struct ShiftBackground : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Frame>;
};

/*!
 * \brief The dynamic part \f$\beta^i_\mathrm{excess}\f$ of the shift
 * \f$\beta^i=\beta^i_\mathrm{background} + \beta^i_\mathrm{excess}\f$
 *
 * We commonly split off the part of the shift that diverges at large coordinate
 * distances (the "background" shift \f$\beta^i_\mathrm{background}\f$) and
 * solve only for the remainder (the "excess" shift
 * \f$\beta^i_\mathrm{excess}\f$). For example, the background shift might be a
 * uniform rotation \f$\beta^i_\mathrm{background}=(-\Omega y, \Omega x, 0)\f$
 * with angular velocity \f$\Omega\f$ around the z-axis, given here in Cartesian
 * coordinates.
 *
 * \see `Xcts::Tags::Background`
 */
template <typename DataType, size_t Dim, typename Frame>
struct ShiftExcess : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Frame>;
};

/*!
 * \brief The symmetric "strain" of the shift vector
 * \f$B_{ij} = \bar{\nabla}_{(i}\bar{\gamma}_{j)k}\beta^k =
 * \left(\partial_{(i}\bar{\gamma}_{j)k} - \bar{\Gamma}_{kij}\right)\beta^k\f$
 *
 * This quantity is used in our formulations of the XCTS equations.
 *
 * Note that the shift is not a conformal quantity, so its index is generally
 * raised and lowered with the spatial metric, not with the conformal metric.
 * However, to compute this "strain" we use the conformal metric as defined
 * above. The conformal longitudinal shift in terms of this quantity is then:
 *
 * \f{equation}
 * (\bar{L}\beta)^{ij} = 2\left(\bar{\gamma}^{ik}\bar{\gamma}^{jl}
 * - \frac{1}{3}\bar{\gamma}^{ij}\bar{\gamma}^{kl}\right) B_{kl}
 * \f}
 *
 * Note that the conformal longitudinal shift is (minus) the "stress" quantity
 * of a linear elasticity system in which the shift takes the role of the
 * displacement vector and the definition of its "strain" remains the same. This
 * auxiliary elasticity system is formulated on an isotropic constitutive
 * relation based on the conformal metric with vanishing bulk modulus \f$K=0\f$
 * (not to be confused with the extrinsic curvature trace \f$K\f$ in this
 * context) and unit shear modulus \f$\mu=1\f$. See the
 * `Elasticity::FirstOrderSystem` and the
 * `Elasticity::ConstitutiveRelations::IsotropicHomogeneous` for details.
 */
template <typename DataType, size_t Dim, typename Frame>
struct ShiftStrain : db::SimpleTag {
  using type = tnsr::ii<DataType, Dim, Frame>;
};

/*!
 * \brief The conformal longitudinal operator applied to the background shift
 * vector minus the time derivative of the conformal metric
 * \f$(\bar{L}\beta^\mathrm{background})^{ij} - \bar{u}^ij\f$
 *
 * Note that the time derivative of the conformal metric has its indices raised
 * as \f$\bar{u}^ij = \bar{\gamma}^{ik}\bar{\gamma}^{jl}
 * \partial_t\bar{\gamma}_{kl}\f$, i.e. by the conformal metric, as usual for
 * conformal quantities.
 *
 * \see `Xcts::Tags::ShiftBackground`
 */
template <typename DataType, size_t Dim, typename Frame>
struct LongitudinalShiftBackgroundMinusDtConformalMetric : db::SimpleTag {
  using type = tnsr::II<DataType, Dim, Frame>;
};

/*!
 * \brief The conformal longitudinal operator applied to the shift vector minus
 * the time derivative of the conformal metric, squared:
 * \f$\left((\bar{L}\beta)^{ij} - \bar{u}^ij\right)
 * \left((\bar{L}\beta)_{ij} - \bar{u}_ij\right)\f$
 */
template <typename DataType>
struct LongitudinalShiftMinusDtConformalMetricSquare : db::SimpleTag {
  using type = Scalar<DataType>;
};

/*!
 * \brief The conformal longitudinal operator applied to the shift vector minus
 * the time derivative of the conformal metric, squared and divided by the
 * square of the lapse:
 * \f$\frac{1}{\alpha^2}\left((\bar{L}\beta)^{ij} - \bar{u}^ij\right)
 * \left((\bar{L}\beta)_{ij} - \bar{u}_ij\right)\f$
 */
template <typename DataType>
struct LongitudinalShiftMinusDtConformalMetricOverLapseSquare : db::SimpleTag {
  using type = Scalar<DataType>;
};

template <typename DataType>
struct ShiftDotDerivExtrinsicCurvatureTrace : db::SimpleTag {
  using type = Scalar<DataType>;
};

template <typename DataType, size_t Dim, typename Frame>
struct ConformalChristoffelFirstKind : db::SimpleTag {
  using type = tnsr::ijj<DataType, Dim, Frame>;
};

template <typename DataType, size_t Dim, typename Frame>
struct ConformalChristoffelSecondKind : db::SimpleTag {
  using type = tnsr::Ijj<DataType, Dim, Frame>;
};

template <typename DataType, size_t Dim, typename Frame>
struct ConformalChristoffelContracted : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Frame>;
};

template <typename DataType>
struct ConformalRicciScalar : db::SimpleTag {
  using type = Scalar<DataType>;
};

}  // namespace Tags
}  // namespace Xcts
