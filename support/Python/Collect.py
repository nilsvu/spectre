# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import h5py
import os
import glob
import yaml
import re
import rich.table
import rich.console
import pandas as pd


def get_reduction_data(reduction_file_path,
                       subfile_patterns,
                       num_iterations=None):
    logging.info(f"Collect reduction data from '{reduction_file_path}'")
    reduction_data = {}
    with h5py.File(reduction_file_path, 'r') as open_reductions_file:
        if not subfile_patterns:
            raise ValueError("Specify subfile patterns. Available subfiles: "
                             f"{open_reductions_file.keys()}")
        for key in open_reductions_file:
            if not any(
                    map(lambda subfile_pattern: re.match(subfile_pattern, key),
                        subfile_patterns)):
                continue
            dataset = open_reductions_file[key]
            if 'Legend' not in dataset.attrs:
                continue
            reduction_data.update(zip(dataset.attrs['Legend'], dataset[-1]))
    return reduction_data


def get_startup_time(stdout):
    match = re.search("Charm\+\+ startup time in seconds: (\d+.\d+)", stdout)
    if match is None:
        return None
    return float(match.group(1)) if match else None


def get_walltime(stdout):
    match = re.search("Wall time in seconds: (\d+.\d+)", stdout)
    return float(match.group(1)) if match else None


def get_num_procs(stdout):
    match = re.search("(\d+) PEs total", stdout)
    if match:
        return float(match.group(1))
    match = re.search(r'(\d+) threads \(PEs\)', stdout)
    if match:
        return float(match.group(1))
    return None


def collect(input_files, subfile_patterns):
    """Collect"""
    if isinstance(input_files, str):
        input_files = [input_files]
    data = []
    for input_file_path in sum(
        [glob.glob(input_file_glob) for input_file_glob in input_files], []):
        logging.info(f"Collect from '{input_file_path}'")
        run_dir = os.path.dirname(input_file_path)
        row = {}

        # Read input file
        input_file = yaml.safe_load(open(input_file_path, 'r'))

        # Read reductions file
        reduction_file_path = os.path.join(
            run_dir, input_file['Observers']['ReductionFileName'] + '.h5')
        row.update(get_reduction_data(reduction_file_path, subfile_patterns))

        # Read spectre.out file
        outfile_path = os.path.join(run_dir, 'spectre.out')
        if os.path.exists(outfile_path):
            with open(outfile_path, 'r') as open_outfile:
                outfile_content = open_outfile.read()
                if ("ERROR" in outfile_content or "KILLED" in outfile_content
                        or "MPI error" in outfile_content):
                    logging.error(f"{run_dir} has failed.")
                    continue
                if not 'Done!' in outfile_content:
                    logging.warning(f"{run_dir} is not done yet.")
                    continue
            row.update(NumProcs=get_num_procs(outfile_content))
            row.update(Walltime=get_walltime(outfile_content))
            row.update(StartupTime=get_startup_time(outfile_content))
        else:
            logging.warning(
                f"{run_dir} has not started yet or is not scheduled.")

        data.append(row)
    df = pd.DataFrame(data)
    console = rich.console.Console()
    console.print(df)
