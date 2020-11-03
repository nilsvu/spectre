#!/bin/env sh

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Instructions to **compile** spectre on Minerva:
#
# 1. Clone the spectre repository, if you haven't already. A good place is in
#    your `/home` directory on Minerva:
#    ```sh
#    cd /home/yourname
#    git clone git@github.com:yourname/spectre.git
#    ```
# 2. Set the `$SPECTRE_HOME` environment variable to the location of the
#    spectre repository, e.g. `/home/yourname/spectre`:
#    ```sh
#    export SPECTRE_HOME=path/to/spectre
#    ```
# 3. Source this script:
#    ```sh
#    source path/to/this/script.sh`
#    ```
#    Note: Add steps 2 and 3 to your `.bashrc` file if you don't want to repeat
#    them every time you log in.
# 4. Create a build directory, if you don't have one already. A good place is in
#    your `/work` directory on Minerva:
#    ```sh
#    cd /work/yourname
#    mkdir spectre-build
#    ```
#    Note: Add a timestamp or descriptive labels to the name of the build
#    directory name, since you may create more build directories later, e.g.
#    `build_YYYY-MM-DD` or `build-clang-Debug`.
# 5. Setup modules and run `cmake` in the build directory:
#    ```sh
#    module purge
#    spectre_setup_modules
#    cd path/to/build/directory
#    spectre_run_cmake
#    ```
#    Note: Remember to `module purge` before `spectre_setup_module` to work in
#    a clean environment, unless you have reasons not to.
# 6. Compile! With the build directory set up and this script sourced, you can
#    skip the previous steps from now on.
#    ```sh
#    module purge
#    spectre_setup_modules
#    spectre_load_modules
#    make -j16 ExportCoordinates1D
#    ```
#
# Instructions to **run** spectre executables on Minerva:
#
# 1. Create a run directory. A good place is in your `/scratch` directory on
#    Minerva. Make sure to choose a descriptive name, e.g.
#    `/scratch/yourname/spectre/your_project/00_the_run`.
# 2. Copy `support/SubmitScripts/Minerva.sh` to the run directory and edit it
#    as the comments in that file instruct.
# 3. Submit the job to Minerva's queue:
#    ```sh
#    sbatch Minerva.sh
#    ```

spectre_setup_modules() {
    source /home/nfischer/spack/share/spack/setup-env.sh
}

spectre_load_modules() {
    module load gcc-9.2.0-gcc-9.2.0-fjz3awm
    module load llvm-10.0.1-gcc-9.2.0-j22quug
    spack env activate spectre_2020-11-02
}

spectre_unload_modules() {
    module unload gcc-9.2.0-gcc-9.2.0-fjz3awm
    module unload llvm-10.0.1-gcc-9.2.0-j22quug
    spack env deactivate
}

spectre_run_cmake() {
    if [ -z ${SPECTRE_HOME} ]; then
        echo "You must set SPECTRE_HOME to the cloned SpECTRE directory"
        return 1
    fi
    spectre_load_modules
    cmake \
      -D CMAKE_C_COMPILER=clang \
      -D CMAKE_CXX_COMPILER=clang++ \
      -D CMAKE_Fortran_COMPILER=gfortran \
      -D CHARM_ROOT=$(spack location -i charmpp) \
      -D CMAKE_BUILD_TYPE=Release \
      -D DEBUG_SYMBOLS=OFF \
      -D BUILD_PYTHON_BINDINGS=ON \
      "$@" $SPECTRE_HOME
}
