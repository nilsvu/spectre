// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <tuple>

#include "Utilities/TMPL.hpp"
#include "Utilities/TupleFromTypelist.hpp"

static_assert(std::is_same_v<tuple_from_typelist<tmpl::list<int, float>>,
                             std::tuple<int, float>>);
static_assert(std::is_same_v<tuple_from_typelist<tmpl::list<int, int, float>>,
                             std::tuple<int, int, float>>);
static_assert(
    std::is_same_v<tuple_from_typelist<tmpl::list<int>>, std::tuple<int>>);
static_assert(std::is_same_v<tuple_from_typelist<tmpl::list<>>, std::tuple<>>);
