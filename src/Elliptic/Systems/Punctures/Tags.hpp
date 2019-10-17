// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines DataBox tags for the Poisson system

#pragma once

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class DataVector;
/// \endcond

/*!
 * \ingroup EllipticSystemsGroup
 * \brief Items related to solving for puncture initial data
 */
namespace Punctures {
namespace Tags {

/*!
 * \brief The scalar field \f$u(x)\f$ to solve for
 */
struct Field : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "PunctureField"; }
};

struct Alpha : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "Alpha"; }
};

struct Beta : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "Beta"; }
};

}  // namespace Tags
}  // namespace Poisson
