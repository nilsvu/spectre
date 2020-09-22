// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <map>
#include <optional>
#include <tuple>

#include "DataStructures/DenseVector.hpp"
#include "NumericalAlgorithms/Convergence/HasConverged.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Utilities/TMPL.hpp"

namespace NonlinearSolver::newton_raphson::detail::Tags {

template <typename OptionsGroup>
struct GlobalizationIsComplete
    : Parallel::InboxInserters::Value<GlobalizationIsComplete<OptionsGroup>> {
  using temporal_id = size_t;
  using type = std::map<temporal_id, std::optional<Convergence::HasConverged>>;
};

}  // namespace NonlinearSolver::newton_raphson::detail::Tags
