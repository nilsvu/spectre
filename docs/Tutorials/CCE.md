\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# %Running CCE {#tutorial_cce}

The basic instructions for getting up and running with a stand-alone
CCE using external data are:
- Clone spectre and build the CharacteristicExtract target
- *Note*: CCE currently requires the linked HDF5 library to be built
  thread-safe. As this is a significant restriction on many systems,
  we hope to eliminate this requirement in the future.
- At this point (provided the build succeeds) you should have the
  executable `build_directory/bin/CharacteristicExtract`; You can now
  run that using an input file that should be patterned on
  `tests/InputFiles/Cce/CharacteristicExtract.yaml` from the spectre
  source tree. There are a few important notes there:
  - for resolution, the example input file has lmax (option group `CCE`,
    option `LMax`) of 12, and filter lmax (option group `Filtering`, option
    `FilterLMax`) of 10; that'll run pretty fast but might be a little low for
    full precision. lmax 16, filter 14 should be pretty good, and typically
    precision doesn't improve above lmax 24, filter 22 (be sure to
    update the filter as you update lmax).
  - if you want to just run through the end of the provided worldtube data,
    you can just omit the `EndTime` option and the executable will figure it
    out from the worldtube file.
  - the `ScriOutputDensity` adds extra interpolation points to the output,
    which is useful for finite-difference derivatives on the output data, but
    otherwise it'll just unnecessarily inflate the output files, so if you
    don't need the extra points, best just set it to 1.
  - if you're extracting at 100M or less, best to reduce the `TargetStepSize`,
    to around .5 at 100M and lower yet for nearer extraction.
  - the `InitializeJ` in the example file uses `InverseCubic` which is a pretty
    primitive scheme, but early tests indicate that it gives the best results
    for most systems.
    If initial data is a concern, you can also try replacing `InverseCubic`
    entry with :
    ```
    NoIncomingRadiation:
      AngularCoordTolerance: 1.0e-13
      MaxIterations: 500
      RequireConvergence: true
    ```
    which are probably pretty good choices for those parameters,
    and the `RequireConvergence: true` will cause the iterative solve in
    this version to error out if it doesn't find a good frame.
  .
- An example of an appropriate submission command for slurm systems is:
  ```
  srun -n 1 -c 1 path/to/build/bin/BondiCharacteristicExtract ++ppn 3 \
 --input-file path/to/input.yaml
  ```
  CCE doesn't currently scale to more than 4 cores, so those slurm options are
  best.
- CCE will work faster if the input file is chunked in small numbers of
  complete rows in the hdf5.
  This is relevant because by default, SpEC writes its worldtube files
  chunked along full time-series columns, which is efficient for
  compression, but not for reading in to SpECTRE -- in that case,
  it is recommended to rechunk the input file before running CCE
  for maximum performance.
- The output data will be written as spin-weighted spherical harmonic
  modes, one physical quantity per dataset, and each row will
  have the time value followed by the real and imaginary parts
  of the complex modes in m-varies-fastest order.

### Input data formats

The worldtube data must be constructed as spheres of constant coordinate
radius, and (for the time being) written to a filename of the format
`...CceRXXXX.h5`, where the `XXXX` is to be replaced by the integer for which
the extraction radius is equal to `XXXX`M. For instance, a 100M extraction
should have filename `...CceR0100.h5`. This scheme of labeling files with the
extraction radius is constructed for compatibility with SpEC worldtube data.

There are two possible formats of the input data, one based on the Cauchy metric
at finite radius, and one based on Bondi data. The metric data format must be
provided as spherical harmonic modes with the following datasets:
- `gxx.dat`, `gxy.dat`, `gxz.dat`, `gyy.dat`, `gyz.dat`, `gzz.dat`
- `Drgxx.dat`, `Drgxy.dat`, `Drgxz.dat`, `Drgyy.dat`, `Drgyz.dat`, `Drgzz.dat`
- `Dtgxx.dat`, `Dtgxy.dat`, `Dtgxz.dat`, `Dtgyy.dat`, `Dtgyz.dat`, `Dtgzz.dat`
- `Shiftx.dat`, `Shifty.dat`, `Shiftz.dat`
- `DrShiftx.dat`, `DrShifty.dat`, `DrShiftz.dat`
- `DtShiftx.dat`, `DtShifty.dat`, `DtShiftz.dat`
- `Lapse.dat`
- `DrLapse.dat`
- `DtLapse.dat`
In this format, each row must start with the time stamp, and the remaining
values are the complex modes in m-varies-fastest format.

The second format is Bondi-Sachs metric component data. This format is far more
space-efficient (by around a factor of 4), and SpECTRE provides a separate
executable for converting to the Bondi-Sachs worldtube format,
`ReduceCceWorldtube`. The format is similar to the metric components, except
in spin-weighted spherical harmonic modes, and the real (spin-weight-0)
quantities omit the redundant negative-m modes and imaginary parts of m=0
modes. The quantities that must be provided by the Bondi-Sachs metric data
format are:
- `Beta.dat`
- `DrJ.dat`
- `DuR.dat`
- `H.dat`
- `J.dat`
- `Q.dat`
- `R.dat`
- `U.dat`
- `W.dat`
