#!/bin/env sh

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Instructions to **compile** spectre on Minerva:
#
# 1. We recommend you compile spectre on a compute node to avoid disrupting the
#    login node for other users. This is how you can request a compute node on
#    the `devel` queue for a few hours:
#    ```sh
#    srun -p devel --nodes=1 --ntasks-per-node=16 --time=08:00:00 --pty bash -i
#    ```
#    You will be dropped right into a shell on the compute node as soon as the
#    scheduler has allocated one for you so you can proceed with these
#    instructions.
# 2. Clone the spectre repository, if you haven't already. A good place is in
#    your `/home` directory on Minerva:
#    ```sh
#    cd /home/yourname
#    git clone git@github.com:yourname/spectre.git
#    ```
# 3. Set the `$SPECTRE_HOME` environment variable to the location of the spectre
#    repository, e.g. `/home/yourname/spectre`:
#    ```sh
#    export SPECTRE_HOME=path/to/spectre
#    ```
# 4. Source this script and setup modules:
#    ```sh
#    source path/to/this/script.sh`
#    spectre_setup_modules
#    ```
#    Note: Add steps 3 and 4 to your `.bashrc` file if you don't want to repeat
#    them every time you log in. The `spectre_setup_modules` function only
#    adjusts your `MODULEPATH` to make the installed modules visible but doesn't
#    load any, so it's safe to add to your `.bashrc`.
# 5. Create a build directory, if you don't have one already. A good place is in
#    your `/work` directory on Minerva:
#    ```sh
#    cd /work/yourname
#    mkdir spectre-build
#    ```
#    Note: Add a timestamp or descriptive labels to the name of the build
#    directory, since you may create more build directories later, e.g.
#    `build_YYYY-MM-DD` or `build-clang-Debug`.
# 6. Run `cmake` to configure the build directory:
#    ```sh
#    cd path/to/build/directory
#    module purge
#    spectre_run_cmake
#    ```
#    Note: Remember to `module purge` to work in a clean environment, unless you
#    have reasons not to.
# 7. Compile! With the build directory configured, this script sourced and
#    modules set up, you can skip the previous steps from now on.
#    ```sh
#    module purge
#    spectre_load_modules
#    make -j16 SPECTRE_EXECUTABLE
#    ```
#    Replace `SPECTRE_EXECUTABLE` with the target you want to build, e.g.
#    `unit-tests` or `SolvePoisson3D`.
#
# Instructions to **run** spectre executables on Minerva:
#
# 1. Create a run directory. A good place is in your `/scratch` directory on
#    Minerva. Make sure to choose a descriptive name, e.g.
#    `/scratch/yourname/spectre/your_project/00_the_run`.
# 2. Copy `support/SubmitScripts/Minerva.sh` to the run directory and edit it as
#    the comments in that file instruct.
# 3. Submit the job to Minerva's queue:
#    ```sh
#    sbatch Minerva.sh
#    ```

spectre_setup_modules() {
    export MODULEPATH="\
/home/SPACK2021/share/spack/modules/linux-centos7-haswell:$MODULEPATH"
    export MODULEPATH="\
/home/nfischer/spack/share/spack/modules/linux-centos7-haswell:$MODULEPATH"
}

spectre_load_modules() {
    module load gcc-10.2.0-gcc-10.2.0-vaerku7
    module load llvm-11.0.1-gcc-10.2.0-twtxwft
    source /home/nfischer/spack/var/spack/environments/spectre_2021-02-22/loads
    export CHARM_ROOT="\
/home/nfischer/spack/opt/spack/linux-centos7-haswell/clang-11.0.1/\
charmpp-6.10.2-rrjct3ksof7de2sufhxurj6ljzlndvrm"
}

spectre_unload_modules() {
    echo "Unloading a subset of modules is not supported."
    echo "Run 'module purge' to unload all modules."
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
      -D CHARM_ROOT=$CHARM_ROOT \
      -D CMAKE_BUILD_TYPE=Release \
      -D DEBUG_SYMBOLS=OFF \
      -D BUILD_PYTHON_BINDINGS=ON \
      -Wno-dev "$@" $SPECTRE_HOME
}
