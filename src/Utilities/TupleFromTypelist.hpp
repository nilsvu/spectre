// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "Utilities/TMPL.hpp"

namespace tuple_impl_detail {

template <typename Elements>
struct tuple_from_typelist_impl;

template <typename... Elements>
struct tuple_from_typelist_impl<tmpl::list<Elements...>> {
  using type = std::tuple<Elements...>;
};

}  // namespace tuple_impl_detail

/// \ingroup UtilitiesGroup
/// A `std::tuple` from the types in a `tmpl::list`
template <typename Elements>
using tuple_from_typelist =
    tmpl::type_from<tuple_impl_detail::tuple_from_typelist_impl<Elements>>;
