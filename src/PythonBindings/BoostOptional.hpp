// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file Allows using `boost::optional` in Python bindings

#pragma once

#include <boost/optional.hpp>
#include <pybind11/pybind11.h>

namespace pybind11 {
namespace detail {
template <typename T>
struct type_caster<boost::optional<T>> : optional_caster<boost::optional<T>> {};
}  // namespace detail
}  // namespace pybind11
