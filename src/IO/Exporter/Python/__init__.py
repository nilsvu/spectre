# Distributed under the MIT License.
# See LICENSE.txt for details.

from typing import Optional, Sequence, Type, Union

import numpy as np

from . import _Pybindings
from ._Pybindings import *


def interpolate_to_points(
    volume_files_or_glob: Union[str, Sequence[str]],
    subfile_name: str,
    observation: Union[ObservationId, ObservationStep, ObservationValue],
    tensor_components: Sequence[str],
    target_points,
    extrapolate_into_excisions: bool = False,
    error_on_missing_points: bool = False,
    num_threads: Optional[int] = None,
):
    """Interpolate volume data to target points.

    Arguments:
      volume_files_or_glob: List of H5 volume data files, or a glob pattern.
      subfile_name: Name of the volume data subfile in the H5 files.
      observation: Observation to interpolate, e.g. 'ObservationStep(-1)' for
        the last observation.
      tensor_components: Tensor components to interpolate, e.g. 'Lapse' or
        'Shift_x'.
      target_points: Target points as a tensor 'tnsr.I[DataVector, Dim,
        Frame]', or as an array with shape (Dim, num_points). Arrays are
        interpreted as inertial coordinates.
      extrapolate_into_excisions: Fill excision regions by extrapolation.
      error_on_missing_points: Raise an error if a point is outside the
        domain. Otherwise, the result is NaN at that point.
      num_threads: Number of OpenMP threads (default: all available).

    Returns: List of the interpolated tensor components, each with
      num_points values.
    """
    if isinstance(target_points, (np.ndarray, list, tuple)):
        from spectre.DataStructures import DataVector
        from spectre.DataStructures.Tensor import tnsr

        target_points = np.asarray(target_points, dtype=float)
        if target_points.ndim != 2 or not 1 <= target_points.shape[0] <= 3:
            raise ValueError(
                "Target points must have shape (Dim, num_points) with Dim "
                f"between 1 and 3, but have shape {target_points.shape}. "
                "If your points have shape (num_points, Dim), transpose them."
            )
        target_points = tnsr.I[DataVector, target_points.shape[0]](
            target_points
        )
    return _Pybindings.interpolate_to_points(
        volume_files_or_glob,
        subfile_name=subfile_name,
        observation=observation,
        tensor_components=tensor_components,
        target_points=target_points,
        extrapolate_into_excisions=extrapolate_into_excisions,
        error_on_missing_points=error_on_missing_points,
        num_threads=num_threads,
    )


def interpolate_tensors_to_points(
    *args,
    tensor_names: Sequence[str],
    tensor_types: Sequence[Type],
    **kwargs,
):
    """Wrapper around 'interpolate_to_points' for tensors.

    Interpolates volume data to points by calling 'interpolate_to_points'.
    However, instead of passing a list of tensor components
    (like 'SpatialMetric_xx') you can pass a list of tensor names (like
    'SpatialMetric') and their corresponding types (like
    'tnsr.ii[DataVector, 3]') and this function will assemble the tensors
    and return them as a list.

    Note: It would be nice to move this function into C++, but that's not
    trivial because the tensor types are not known at compile time. This can be
    revisited when we have Python support for Variables.
    """
    from spectre.DataStructures import DataVector

    tensors = {
        tensor_name: tensor_type()
        for tensor_name, tensor_type in zip(tensor_names, tensor_types)
    }
    tensor_components = []
    for name, tensor in tensors.items():
        tensor_components += [
            name + tensor.component_suffix(i) for i in range(tensor.size)
        ]
    data = interpolate_to_points(
        *args,
        **kwargs,
        tensor_components=tensor_components,
    )
    j = 0
    for tensor in tensors.values():
        for i in range(tensor.size):
            tensor[i] = DataVector(data[j])
            j += 1
    return list(tensors.values())


SpacetimeInterpolator = {
    1: SpacetimeInterpolator1D,
    2: SpacetimeInterpolator2D,
    3: SpacetimeInterpolator3D,
}
