// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

template <typename Invokable, typename... Args, size_t... ArgsIndices>
auto expand_tuple_impl(const std::tuple<Args...>& args, Invokable&& invokable,
                       std::index_sequence<ArgsIndices...> /* meta */) {
  return invokable(std::get<ArgsIndices>(args)...);
}

template <typename Invokable, typename... Args>
auto expand_tuple(const std::tuple<Args...>& args, Invokable&& invokable) {
  return expand_tuple_impl(args, invokable,
                           std::make_index_sequence<sizeof...(Args)>{});
}
