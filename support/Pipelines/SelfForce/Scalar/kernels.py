from spectre.IO.Exporter import interpolate_to_points, ObservationStep
from spectre.DataStructures.Tensor import Scalar, DataVector, tnsr

ref_fname = "/Users/nilsvu/Projects/spectre/build-Default-Release/test_ssf62/m1/ScalarSelfForceVolume0.h5"

def error(field: Scalar[DataVector], x: tnsr.I[DataVector, 2]) -> DataVector:
    ref_data, = interpolate_to_points(ref_fname, "VolumeData", observation=ObservationStep(-1), tensor_components=["Re(MMode)"],
    target_points=x)
    # m = 0
    # z = x[1]
    return (field.get() #* (1 - z**2)**(m/2)
            - ref_data)
