// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Utilities/TMPL.hpp"

namespace elliptic {
namespace detail {
struct NoModifyBoundaryData {
  using argument_tags = tmpl::list<>;
  using argument_tags_linearized = tmpl::list<>;
};
}  // namespace detail

/// The `argument_tags` of the `System::modify_boundary_data`, or an empty list
/// if `System::modify_boundary_data` is `void`.
template <typename System>
using get_modify_boundary_data = tmpl::conditional_t<
    std::is_same_v<typename System::modify_boundary_data, void>,
    detail::NoModifyBoundaryData, typename System::modify_boundary_data>;

template <typename System, bool Linearized>
using get_modify_boundary_data_args_tags = tmpl::conditional_t<
    Linearized,
    typename get_modify_boundary_data<System>::argument_tags_linearized,
    typename get_modify_boundary_data<System>::argument_tags>;

}  // namespace elliptic
