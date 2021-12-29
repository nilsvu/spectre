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
#    Note: Add this step to your `.bashrc` file if you don't want to repeat it
#    every time you log in.
# 4. Source this script and setup modules:
#    ```sh
#    source $SPECTRE_HOME/support/Environments/minerva_gcc.sh`
#    module purge
#    spectre_setup_modules
#    ```
#    This may take a while, because it builds all dependencies. You only have
#    to do this once, or when you want to rebuild your dependency tree.
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
#    spectre_load_modules
#    spectre_run_cmake
#    ```
#    Note: Remember to `module purge` to work in a clean environment, unless you
#    have reasons not to. When you are coming directly from
#    `spectre_setup_modules`, you can skip the `module purge` and
#    `spectre_load_modules`.
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
# 2. Copy `$SPECTRE_HOME/support/SubmitScripts/Minerva.sh` to the run directory
#    and edit it as the comments in that file instruct.
# 3. Submit the job to Minerva's queue:
#    ```sh
#    sbatch Minerva.sh
#    ```

spectre_load_sys_modules() {
    module load gcc-10.2.0-gcc-10.2.0-vaerku7
    module load binutils-2.36.1-gcc-10.2.0-wtzd7wm
    module load git-2.29.0-gcc-10.2.0-m4kt4vn
    module load python-3.8.7-gcc-10.2.0-pc7wahl
}

spectre_unload_sys_modules() {
    module unload gcc-10.2.0-gcc-10.2.0-vaerku7
    module unload binutils-2.36.1-gcc-10.2.0-wtzd7wm
    module unload git-2.29.0-gcc-10.2.0-m4kt4vn
    module unload python-3.8.7-gcc-10.2.0-pc7wahl
}

spectre_upgrade_environment() {
    # Use this command to update the .lock file in the repo

    spectre_load_sys_modules

    # Recreate the environment from scratch
    spack env deactivate
    spack env remove -y spectre_minerva_gcc
    spack env create spectre_minerva_gcc \
      $SPECTRE_HOME/support/DevEnvironments/spack.yaml
    spack env activate spectre_minerva_gcc -p
    spack compiler find
    spack external find

    # Configure Charm++ installation
    spack remove charmpp && spack add \
      intel-mpi charmpp@6.10.2 backend=mpi scotch

    # Concretize
    spack concretize -f --reuse
    cp $SPECTRE_HOME/support/spack/var/spack/environments/spectre_minerva_gcc/spack.lock \
      $SPECTRE_HOME/support/Environments/minerva_gcc.lock
}

spectre_setup_modules() {
    if [ -z ${SPECTRE_HOME} ]; then
        echo "You must set SPECTRE_HOME to the cloned SpECTRE directory"
        return 1
    fi

    spectre_load_sys_modules

    # Install or upgrade Spack in $SPECTRE_HOME/support/spack
    if [ -f $SPECTRE_HOME/support/spack/share/spack/setup-env.sh ]; then
        git -C $SPECTRE_HOME/support/spack/ pull
    else
        git clone -c feature.manyFiles=true \
          https://github.com/spack/spack.git $SPECTRE_HOME/support/spack
        git -C $SPECTRE_HOME/support/spack/ apply \
          $SPECTRE_HOME/support/Environments/spack.patch
        # Add upstream install tree to reuse existing packages
        cat >$SPECTRE_HOME/support/spack/etc/spack/upstreams.yaml <<EOF
upstreams:
  spack2021:
    install_tree: /home/SPACK2021/opt/spack
    modules:
      lmod: /home/SPACK2021/share/spack/modules
EOF
    fi

    # Create environment from the lockfile
    source $SPECTRE_HOME/support/spack/share/spack/setup-env.sh
    spack env deactivate
    spack env remove -y spectre_minerva_gcc
    spack env create spectre_minerva_gcc \
      $SPECTRE_HOME/support/Environments/minerva_gcc.lock
    spack env activate spectre_minerva_gcc -p

    # Install
    spack install --only-concrete
    spack env loads -r
    spack find -x
}

spectre_load_modules() {
    spectre_load_sys_modules
    source $SPECTRE_HOME/support/spack/share/spack/setup-env.sh
    spack env activate spectre_minerva_gcc -p
    export CHARM_ROOT=`spack location --install-dir charmpp`
}

spectre_unload_modules() {
    spack env deactivate
    spectre_unload_sys_modules
}

spectre_run_cmake() {
    if [ -z ${SPECTRE_HOME} ]; then
        echo "You must set SPECTRE_HOME to the cloned SpECTRE directory"
        return 1
    fi
    cmake \
      -D CMAKE_C_COMPILER=gcc \
      -D CMAKE_CXX_COMPILER=g++ \
      -D CMAKE_Fortran_COMPILER=gfortran \
      -D CHARM_ROOT=$CHARM_ROOT \
      -D CMAKE_BUILD_TYPE=Release \
      -D DEBUG_SYMBOLS=OFF \
      -D BUILD_SHARED_LIBS=ON \
      -D MEMORY_ALLOCATOR=SYSTEM \
      -D BUILD_PYTHON_BINDINGS=ON \
      -Wno-dev "$@" $SPECTRE_HOME
}
