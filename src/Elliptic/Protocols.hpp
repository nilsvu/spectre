// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Utilities/TypeTraits.hpp"

namespace elliptic {
namespace Protocols {

struct SecondOrderSystem {};

template <typename T>
using is_second_order_system =
    typename std::is_convertible<T*, SecondOrderSystem*>;

template <typename T>
constexpr bool is_second_order_system_v =
    cpp17::is_convertible_v<T*, SecondOrderSystem*>;

}  // namespace Protocols
}  // namespace elliptic
