// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/EventsAndTriggers/Completion.hpp"

namespace Events {
PUP::able::PUP_ID Completion::my_PUP_ID = 0;  // NOLINT
bool Completion::needs_evolved_variables() const noexcept { return false; }
}  // namespace Events
