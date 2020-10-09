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
namespace Tags {

/*!
 * \brief The conformal factor \f$\psi(x)\f$ that rescales the spatial metric
 * \f$\gamma_{ij}=\psi^4\overline{\gamma}_{ij}\f$.
 */
template <typename DataType>
struct ConformalFactor : db::SimpleTag {
  using type = Scalar<DataType>;
};

/*!
 * \brief The `Tag` scaled by the specified `Power` of the conformal factor
 * \f$\psi(x)\f$
 */
template <typename Tag, int Power>
struct Conformal : db::PrefixTag {
  using type = typename Tag::type;
  using tag = Tag;
  static std::string name() noexcept {
    return "Conformal(" + db::tag_name<Tag>() + ", " + std::to_string(Power) +
           ")";
  }
};

/*!
 * \brief The product of lapse \f$\alpha(x)\f$ and conformal factor
 * \f$\psi(x)\f$
 *
 * \details This quantity is commonly used in formulations of the XCTS
 * equations.
 */
template <typename DataType>
struct LapseTimesConformalFactor : db::SimpleTag {
  using type = Scalar<DataType>;
};

template <size_t Dim, typename Frame, typename DataType>
struct ShiftStrain : db::SimpleTag {
  using type = tnsr::ii<DataType, Dim, Frame>;
};

template <typename DataType, size_t Dim, typename Frame>
struct ShiftBackground : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Frame>;
};

template <typename DataType, size_t Dim, typename Frame>
struct ShiftExcess : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Frame>;
};

}  // namespace Tags
}  // namespace Xcts
