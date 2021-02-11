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
 * \bar{D}^2\left(\alpha\psi\right) =
 * \alpha\psi\left(\frac{7}{8}\psi^{-8}\bar{A}_{ij}\bar{A}^{ij}
 * + \frac{5}{12}\psi^4 K^2 + \frac{1}{8}\bar{R}
 * + 2\pi\psi^4\left(\rho + 2S\right)\right)
 * - \psi^5\partial_t K + \psi^5\beta^i\bar{D}_i K
 * \\
 * \text{with} \quad
 * \bar{A} = \frac{1}{2\bar{\alpha}} \left(
 * \left(\bar{L}\beta\right)^{ij} - \bar{u}^{ij}\right)
 * \quad \text{and}
 * \bar{\alpha} = \alpha \psi^{-6}
 * \f}
 *
 * are a set of nonlinear elliptic equations that the spacetime metric in
 * general relativity must satisfy at all times. For an introduction see e.g.
 * \cite BaumgarteShapiro, in particular Box 3.3 which is largely mirrored here.
 * We solve the XCTS equations for the conformal factor \f$\psi\f$, the product
 * of lapse times conformal factor \f$\alpha\psi\f$ and the shift vector
 * \f$\beta\f$. The remaining quantities in the equations, i.e. the conformal
 * metric \f$\bar{\gamma}_{ij}\f$, the extrinsic curvature \f$K\f$, their
 * respective time derivatives \f$\bar{u}_{ij}\f$ and \f$\partial_t K\f$, the
 * energy density \f$\rho\f$, the stress-energy trace \f$S\f$ and the momentum
 * density \f$S^i\f$, are freely specifyable fields that define the physical
 * scenario at hand. Of particular importance is the conformal metric, which
 * serves as the background geometry for these equations by defining the
 * covariant derivative \f$\bar{D}\f$, the Ricci scalar \f$\bar{R}\f$ and the
 * longitudinal operator
 *
 * \f{begin}
 * \left(\bar{L}\beta\right)^{ij} = \bar{D}^i\beta^j + \bar{D}^j\beta^i
 * - \frac{2}{3}\bar{\gamma}^{ij}\bar{D}_k\beta^k
 * \text{.}
 * \f}
 *
 * Note that the XCTS equations are essentially two Poisson equations and one
 * Elasticity equation with nonlinear sources on a curved geometry.
 *
 * Once the XCTS equations are solved we can construct the spatial metric and
 * extrinsic curvature as
 *
 * \f{align} \gamma_{ij} &= \psi^4\bar{\gamma}_{ij} \\
 * K_{ij} &= \psi^{-2}\bar{A}_{ij} + \frac{1}{3}\gamma_{ij} K \f}
 *
 * from which we can compose the full spacetime metric.
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
 * \f$B_{ij} = \bar{D}_{(i}\bar{\gamma}_{j)k}\beta^k =
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

template <typename DataType, size_t Dim, typename Frame>
struct LongitudinalShiftExcess : db::SimpleTag {
  using type = tnsr::II<DataType, Dim, Frame>;
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

/*!
 * \brief The shift vector contracted with the extrinsic curvature trace
 * gradient: \f$\beta^i\partial_i K\f$
 */
template <typename DataType>
struct ShiftDotDerivExtrinsicCurvatureTrace : db::SimpleTag {
  using type = Scalar<DataType>;
};

/*!
 * \brief The Christoffel symbols of the first kind related to the conformal
 * metric \f$\bar{\gamma}_{ij}\f$
 */
template <typename DataType, size_t Dim, typename Frame>
struct ConformalChristoffelFirstKind : db::SimpleTag {
  using type = tnsr::ijj<DataType, Dim, Frame>;
};

/*!
 * \brief The Christoffel symbols of the second kind related to the conformal
 * metric \f$\bar{\gamma}_{ij}\f$
 */
template <typename DataType, size_t Dim, typename Frame>
struct ConformalChristoffelSecondKind : db::SimpleTag {
  using type = tnsr::Ijj<DataType, Dim, Frame>;
};

/*!
 * \brief The Christoffel symbols of the second kind (related to the conformal
 * metric \f$\bar{\gamma}_{ij}\f$) contracted in their first two indices
 */
template <typename DataType, size_t Dim, typename Frame>
struct ConformalChristoffelContracted : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Frame>;
};

/*!
 * \brief The Ricci scalar related to the conformal metric
 * \f$\bar{\gamma}_{ij}\f$
 */
template <typename DataType>
struct ConformalRicciScalar : db::SimpleTag {
  using type = Scalar<DataType>;
};

}  // namespace Tags
}  // namespace Xcts
