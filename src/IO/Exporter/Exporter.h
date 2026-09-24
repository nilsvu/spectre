// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>

void spectre_interpolate_to_points(
    double* out, const char** volume_files, size_t num_volume_files,
    const char* subfile_name, int observation_step,
    const char** tensor_components, size_t num_tensor_components,
    const double* target_points, size_t num_dimensions, size_t num_points,
    bool extrapolate_into_excisions, bool error_on_missing_points,
    int num_threads);

#ifdef __cplusplus
}
#endif
