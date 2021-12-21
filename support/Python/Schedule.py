# Distributed under the MIT License.
# See LICENSE.txt for details.

import jinja2
import logging
import os
import re
import rich.console
import spectre.tools.CleanOutput
import subprocess

SPECTRE_HOME = "@CMAKE_SOURCE_DIR@"
SPECTRE_BUILD_DIR = "@CMAKE_BINARY_DIR@"


def get_executable_name(input_file_contents):
    match = re.search(r'# Executable: (.+)', input_file_contents)
    if not match:
        raise ValueError(
            "The file does not not specify an executable. Add a comment such "
            "as '# Executable: EvolveScalarWave' to the file.")
    return match.group(1)


def schedule(run_dir,
             input_file_template,
             scheduler,
             clean_output=False,
             job_name=None,
             submit_script_template=None,
             input_file_name=None,
             submit_script_name='Submit.sh',
             build_dir=None,
             force=False,
             **kwargs) -> subprocess.CompletedProcess:
    """Schedule executable runs with an input file

    The input file, submit script, etc. will be configured to the `run_dir`, and
    then the `scheduler` is invoked to submit the run. You can also bypass the
    scheduler and run the executable directly by setting the `scheduler` to
    `None`.

    The input file must specify its corresponding executable in a comment like
    the following, which is typically placed at the top of the file:

    ```yaml
    # Executable: EvolveScalarWave
    ```

    The input file, submit script, `run_dir` and `job_name` can have
    placeholders like '{{ num_nodes }}'. They must conform to the
    [Jinja template format](https://jinja.palletsprojects.com/en/3.0.x/templates/).
    The following parameters are always available:

    - All arguments to this function, including all additional `**kwargs`.
    - executable_name: The executable name as read from the input file template.

    The `run_dir` can use the resolved `job_name`, if one was passed to this
    function. Both the input file template and the submit script template can
    use the following additional parameters:

    - run_dir: Absolute path to the `run_dir`.
    - input_file_path: Absolute path to the input file in the `run_dir`.
    - out_file: Absolute path to the log file.

    All additional `**kwargs` to this function are also forwarded to the
    configuration. You can use this to schedule multiple runs using the same
    input file template. For example, you can do a convergence test by using a
    placeholder for the number of grid points in your input file:

    ```yaml
    # In the domain creator:
    InitialGridPoints: {{ num_points }}
    ```

    When a parameter in `**kwargs` is an iterable, the `schedule` function will
    recurse for every element in the iterable. For example, you can schedule
    multiple runs for a convergence test like this:

    ```py
    schedule(
        run_dir='p{{ num_points - 1 }}',
        # ...
        num_points=range(3, 6))
    ```

    The additional parameters `num_procs`, `num_nodes` and `procs_per_node`
    determine the number of cores and nodes used for running the executable.
    Either specify `num_procs` or both `num_nodes` and `procs_per_node`.

    Typical additional parameters used in submit scripts are `queue` and
    `time_limit`.

    You can determine many of the scheduler-related arguments using
    `spectre.support.Machines.this_machine()` to identify the current machine,
    and retrieving its attributes (see `spectre.support.Machines.Machine`).

    Args:
      run_dir: Required when the input file is outside the current working
        directory. The directory to which input file, submit script, etc. are
        copied, and relative to which the executable will run. It can be a
        Jinja template (see above).
      input_file_template: Path to an input file. It will be copied to the
        `run_dir`. It can be a Jinja template (see above).
      scheduler: `None` to run the executable directly, or a scheduler such as
        `"sbatch"` to submit the run to a queue.
      clean_output: Optional. When `True`, use
        `spectre.tools.CleanOutput.clean_output` to
        clean up existing output files in the `run_dir` before scheduling the
        run. (Default: `False`)
      job_name: Optional. A string describing the job. It can be a Jinja
        template (see above). (Default: basename of the `run_dir`)
      submit_script_template: Optional only when `scheduler` is `None`. Path to
        a submit script. It will be copied to `run_dir`. It can be a Jinja
        template (see above).
      input_file_name: Optional. Filename of the input file in the `run_dir`.
        (Default: basename of the `input_file_template`)
      submit_script_name: Optional. Filename of the submit script. (Default:
        "Submit.sh")
      build_dir: Optional. Path to the build directory which is used to launch
        the executable. It should contain a 'bin' directory with the compiled
        executable. (Default: The directory corresponding to this installation,
        which is '@CMAKE_BINARY_DIR@')

    Returns: The `subprocess.CompletedProcess` representing the executable run
      if `scheduler` is `None`, or otherwise representing the scheduler process.
    """
    # Defaults
    if not input_file_name:
        input_file_name = os.path.basename(input_file_template)
    if not build_dir:
        build_dir = SPECTRE_BUILD_DIR
    build_dir = os.path.abspath(build_dir)

    # Snapshot function arguments
    all_args = locals().copy()
    del all_args['kwargs']

    # Recursively schedule ranges of runs
    for key, value in kwargs.items():
        if isinstance(value, str):
            continue
        try:
            iter(value)
        except TypeError:
            continue
        for value_i in value:
            logging.info(f"Recurse for {key}={value_i}")
            try:
                schedule(**all_args,
                         **dict(list(kwargs.items()) + [(key, value_i)]))
            except:
                logging.exception(f"Recursion for {key}={value_i} failed.")
        return

    # Read input file template
    with open(input_file_template, 'r') as open_input_file:
        input_file_contents = open_input_file.read()

    # Collect parameters for templates
    try:
        executable_name = get_executable_name(input_file_contents)
    except ValueError as err:
        raise ValueError(
            f"The input file '{input_file_template}' does not specify an "
            "executable. Add a comment such as "
            "'# Executable: EvolveScalarWave' to the file.")
    context = dict(**all_args, **kwargs, executable_name=executable_name)

    # Resolve number of cores, nodes, etc.
    num_procs = kwargs.get('num_procs')
    num_nodes = kwargs.get('num_nodes')
    procs_per_node = kwargs.get('procs_per_node')
    if num_nodes:
        assert procs_per_node, (
            "When you specify 'num_nodes', you also need to specify "
            "'procs_per_node'.")
        assert not num_procs or num_procs == num_nodes * procs_per_node, (
            "Mismatch between 'num_procs', 'num_nodes' and 'procs_per_node'. "
            "Specify either 'num_procs' or 'num_nodes', not both.")
        num_procs = num_nodes * procs_per_node
    elif num_procs:
        assert not procs_per_node or num_procs == procs_per_node, (
            "Mismatch between 'num_procs', 'num_nodes' and 'procs_per_node'. "
            "Specify either 'num_procs' or 'num_nodes', not both.")
        num_nodes = 1
        procs_per_node = num_procs
    elif procs_per_node:
        raise ValueError(
            "When you specify 'procs_per_node' you must also specify "
            "'num_nodes'.")
    else:
        num_procs, num_nodes, procs_per_node = (1, 1, 1)
    context.update(num_procs=num_procs,
                   num_nodes=num_nodes,
                   procs_per_node=procs_per_node)

    # Resolve job_name
    if job_name:
        job_name = jinja2.Template(
            job_name,
            undefined=jinja2.StrictUndefined).render(context).strip()
        context.update(job_name=job_name)

    # Resolve run_dir
    if not run_dir:
        if os.path.abspath(
                os.path.dirname(input_file_template)) == os.path.abspath('.'):
            run_dir = '.'
        else:
            raise ValueError("Please specify a 'run_dir'.")
    run_dir = jinja2.Template(
        run_dir, undefined=jinja2.StrictUndefined).render(context).strip()
    context.update(run_dir=os.path.abspath(run_dir))
    if not job_name:
        job_name = os.path.basename(os.path.abspath(run_dir))
        context.update(job_name=job_name)

    # Resolve outfile
    out_file = os.path.join(run_dir, 'spectre.out')
    context.update(out_file=os.path.abspath(out_file))

    logging.info(f"Configure run directory '{run_dir}'")
    os.makedirs(run_dir, exist_ok=True)

    # Configure input file
    input_file_path = os.path.join(run_dir, input_file_name)
    context.update(input_file_path=os.path.abspath(input_file_path))
    rendered_input_file = jinja2.Template(
        input_file_contents,
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True).render(context)
    if (os.path.abspath(input_file_path) !=
            os.path.abspath(input_file_template)
            or rendered_input_file != input_file_contents):
        if not force and os.path.exists(input_file_path):
            raise OSError(
                f"File already exists at '{input_file_path}'. Delete it, retry "
                "with 'force', or choose another `run_dir`.")
        with open(input_file_path, 'w') as open_input_file:
            open_input_file.write(rendered_input_file)

    # Clean output
    if clean_output:
        spectre.tools.CleanOutput.clean_output(input_file=input_file_path,
                                               output_dir=run_dir,
                                               force=True)

    # If requested, run executable directly and return early
    if not scheduler:
        assert num_nodes == 1, (
            "Running executables directly is currently only supported on a "
            "single node. Set the `scheduler` to submit a multi-node job to "
            "the queue")
        logging.info(f"Run '{executable_name}' in '{run_dir}' on {num_procs} "
                     f"core{'s'[:num_procs!=1]}.")
        run_command = [
            os.path.abspath(os.path.join(build_dir, 'bin', executable_name)),
            '--input-file',
            os.path.abspath(input_file_path), '+p',
            str(num_procs)
        ]
        process = subprocess.Popen(run_command,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   cwd=run_dir,
                                   text=True)
        console = rich.console.Console()
        with open(out_file, 'w') as open_out_file:
            for line in process.stdout:
                console.print(line.strip())
                open_out_file.write(line)
        return process

    # Configure submit script
    assert submit_script_template, (
        "Please specify the 'submit_script_template'.")
    with open(submit_script_template, 'r') as open_submit_script:
        submit_script_contents = open_submit_script.read()
    rendered_submit_script = jinja2.Template(
        submit_script_contents,
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True).render(context)
    submit_script_path = os.path.abspath(
        os.path.join(run_dir, submit_script_name))
    if (os.path.abspath(submit_script_path) !=
            os.path.abspath(submit_script_template)
            or rendered_input_file != rendered_submit_script):
        if not force and os.path.exists(submit_script_path):
            raise OSError(
                f"File already exists at '{submit_script_path}'. Delete it, "
                "retry with 'force', or choose another `run_dir`.")
        with open(submit_script_path, 'w') as open_submit_script:
            open_submit_script.write(rendered_submit_script)

    # Submit
    submit_process = subprocess.run([scheduler, submit_script_path],
                                    cwd=run_dir,
                                    capture_output=True,
                                    text=True)
    matched_submit_msg = re.match('Submitted batch job (\d+)',
                                  submit_process.stdout)
    if matched_submit_msg is None:
        raise RuntimeError(
            f"Failed submitting job '{job_name}': {submit_process}\n"
            f"{submit_process.stderr.strip()}")
    jobid = matched_submit_msg.group(1)
    logging.info(f"Submitted job '{job_name}' ({jobid}).")
    with open(os.path.join(run_dir, 'jobid.txt'), 'w') as open_jobid_file:
        open_jobid_file.write(jobid)
    return submit_process
