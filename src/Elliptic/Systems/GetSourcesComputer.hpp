// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

namespace elliptic {
namespace detail {
template <typename System, typename = std::void_t<>>
struct sources_computer_linearized {
  using type = typename System::sources_computer;
};
template <typename System>
struct sources_computer_linearized<
    System, std::void_t<typename System::sources_computer_linearized>> {
  using type = typename System::sources_computer_linearized;
};
}  // namespace detail

template <typename System, bool Linearized>
using get_sources_computer = tmpl::conditional_t<
    Linearized, typename detail::sources_computer_linearized<System>::type,
    typename System::sources_computer>;
}  // namespace elliptic
