// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

/*!
 * \ingroup EllipticSystemsGroup
 * \brief Items related to solving for puncture initial data
 */
namespace Punctures {
namespace Tags {

template <typename DataType>
struct Field : db::SimpleTag {
  using type = Scalar<DataType>;
  static std::string name() noexcept { return "PunctureField"; }
};

template <size_t Dim, typename Frame, typename DataType>
struct FieldGradient : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Frame>;
  static std::string name() noexcept { return "PunctureFieldGradient"; }
};

template <typename DataType>
struct Alpha : db::SimpleTag {
  using type = Scalar<DataType>;
  static std::string name() noexcept { return "Alpha"; }
};

template <typename DataType>
struct Beta : db::SimpleTag {
  using type = Scalar<DataType>;
  static std::string name() noexcept { return "Beta"; }
};

}  // namespace Tags
}  // namespace Xcts
