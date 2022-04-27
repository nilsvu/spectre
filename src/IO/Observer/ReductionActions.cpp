// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/Observer/ReductionActions.hpp"

#include <vector>

#include "Utilities/Gsl.hpp"

namespace observers::ThreadedActions::ReductionActions_detail {

void append_to_reduction_data(
    const gsl::not_null<std::vector<double>*> all_reduction_data,
    const double t) {
  all_reduction_data->push_back(t);
}

void append_to_reduction_data(
    const gsl::not_null<std::vector<double>*> all_reduction_data,
    const std::vector<double>& t) {
  all_reduction_data->insert(all_reduction_data->end(), t.begin(), t.end());
}

void append_to_reduction_data(
    const gsl::not_null<std::vector<double>*> all_reduction_data,
    const DataVector& t) {
  all_reduction_data->insert(all_reduction_data->end(), t.begin(), t.end());
}

}  // namespace observers::ThreadedActions::ReductionActions_detail
