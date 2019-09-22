// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"

/*!
 * \ingroup EllipticSystemsGroup
 * \brief Items related to solving the Extended Conformal Thin Sandwich (XCTS)
 * equations.
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
  static std::string name() noexcept { return "ConformalFactor"; }
};

/*!
 * \brief The `Tag` scaled by the specified `Power` of the conformal factor
 * \f$\psi(x)\f$
 */
// template <typename Tag, int Power>
// struct Conformal : db::PrefixTag {
//   using type = typename Tag::type;
//   using tag = Tag;
//   static std::string name() noexcept {
//     return "Conformal(" + db::tag_name<Tag>() + ", " + std::to_string(Power)
//     +
//            ")";
//   }
// };

/*!
 * \brief The product of lapse \f$\alpha(x)\f$ and conformal factor
 * \f$\psi(x)\f$
 *
 * \details This quantity is commonly used in formulations of the XCTS
 * equations.
 */
// template <typename DataType>
// using LapseTimesConformalFactor = Conformal<gr::Tags::Lapse<DataType>, 1>;
template <typename DataType>
struct LapseTimesConformalFactor : db::SimpleTag {
  using type = Scalar<DataType>;
  static std::string name() noexcept { return "LapseTimesConformalFactor"; }
};

template <size_t Dim, typename Frame, typename DataType>
struct ShiftStrain : db::SimpleTag {
  using type = tnsr::ii<DataType, Dim, Frame>;
  static std::string name() noexcept { return "ShiftStrain"; }
};

}  // namespace Tags
}  // namespace Xcts
