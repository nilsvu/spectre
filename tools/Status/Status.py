# Distributed under the MIT License.
# See LICENSE.txt for details.

import datetime
import glob
import logging
import os
import pathlib
import re
import subprocess
from io import StringIO
from typing import Any, Optional, Sequence

import click
import h5py
import humanize
import numpy as np
import pandas as pd
import rich.console
import rich.table
import yaml
from spectre.Visualization.ReadInputFile import find_event, get_executable

from .ExecutableStatus import match_executable_status

logger = logging.getLogger(__name__)


def fetch_job_data(fields: Sequence[str],
                   user: Optional[str],
                   allusers: bool = False,
                   state: Optional[str] = None,
                   starttime: Optional[str] = None) -> pd.DataFrame:
    """Query Slurm 'sacct' to get metadata of recent jobs on the machine.

    Arguments:
      fields: List of Slurm fields that 'sacct --format' accepts.
        Run 'sacct --helpformat' to print all available fields.
      user: Slurm user IDs or names, or None for the current user.
        See documentation for 'sacct -u' for details.
      allusers: Fetch data for all users.
        See documentation for 'sacct -a' for details.
      state: Fetch only jobs with this state.
        See documentation for 'sacct -s' for details.
      starttime: Fetch only jobs after this time.
        See documentation for 'sacct -S' for details.

    Returns: Pandas DataFrame with the job data.
    """
    completed_process = subprocess.run(
        ["sacct", "-P", "--format", ",".join(fields)] +
        (["-u", user] if user else []) + (["-a"] if allusers else []) +
        (["-s", state] if state else []) +
        (["-S", starttime] if starttime else []),
        capture_output=True,
        text=True)
    try:
        completed_process.check_returncode()
    except subprocess.CalledProcessError as err:
        raise ValueError(completed_process.stderr) from err
    job_data = pd.read_table(StringIO(completed_process.stdout),
                             sep="|",
                             keep_default_na=False)
    # Filter out some derived jobs. Not sure what these jobs are.
    job_data = job_data[~job_data["JobName"].
                        isin(["batch", "extern", "pmi_proxy", "orted"])]
    job_data = job_data[job_data["User"] != ""]
    # Parse dates and times. Do this in postprocessing because
    # `pd.read_table(parse_dates=...)` doesn't handle NaN values well.
    date_cols = set(fields).intersection({"Start", "End"})
    for date_col in date_cols:
        job_data[date_col] = job_data[date_col].apply(
            lambda v: v.replace("Unknown", "NaN"))
        job_data[date_col] = pd.to_datetime(job_data[date_col],
                                            infer_datetime_format=True)
    # We could parse the elapsed time as a timedelta, but the string
    # representation is fine so we don't right now. Here's the code for it:
    # if "Elapsed" in fields:
    #     job_data["Elapsed"] = pd.to_timedelta(
    #         job_data["Elapsed"].apply(lambda v: v.replace("-", " days ")))
    return job_data


def get_input_file(comment: Optional[str], work_dir: str) -> Optional[str]:
    """Find the input file corresponding to a job.

    Arguments:
      comment: The Slurm comment field. The input file is extracted from it if
        it includes a line like "SPECTRE_INPUT_FILE=path/to/input/file".
      work_dir: The working directory of the job. If no input file was found
        in the Slurm comment, we see if there's a single YAML file in the
        work dir and assume that's the input file.

    Returns: Path to the input file, or None.
    """
    if comment:
        # Get the input file from the Slurm comment if the submission
        # specified it
        match = re.search(r"^SPECTRE_INPUT_FILE=(.*)", comment)
        if match:
            return match.group(1)
        else:
            logger.debug("Could not find 'SPECTRE_INPUT_FILE' "
                         "in Slurm comment:\n" + comment)
    # Fallback: Check if there's a single YAML file in the work dir
    yaml_files = glob.glob(os.path.join(work_dir, "*.yaml"))
    if len(yaml_files) == 1:
        return yaml_files[0]
    else:
        logger.debug("No input file found. "
                     "Didn't find a single YAML file in '{work_dir}'. "
                     f"YAML files found: {yaml_files}")
        return None


def get_executable_name(comment: Optional[str],
                        input_file_path: Optional[str]) -> Optional[str]:
    """Determine the executable name of a job.

    Arguments:
      comment: The Slurm comment field. The executable name is extracted from it
        if it includes a line like "SPECTRE_EXECUTABLE=path/to/executable".
      input_file_path: Path to input file. If no executable name was found in
        the Slurm comment, we try to extract it from the input file.

    Returns: Executable name, or None.
    """
    if comment:
        # Get the executable from the Slurm comment if the submission
        # specified it
        match = re.search(r"^SPECTRE_EXECUTABLE=(.*)", comment)
        if match:
            return os.path.basename(match.group(1))
        else:
            logger.debug("Could not find 'SPECTRE_EXECUTABLE' "
                         "in Slurm comment:\n" + comment)
    # Fallback: Look for a comment like "# Executable: ..." in the input file
    if not input_file_path:
        return None
    executable = get_executable(pathlib.Path(input_file_path).read_text())
    return os.path.basename(executable) if executable else None


def _state_order(state):
    order = ["RUNNING", "PENDING", "COMPLETED", "TIMEOUT", "FAILED"]
    try:
        return order.index(state)
    except ValueError:
        return None


def _format(field: str, value: Any) -> str:
    if field == "State":
        style = {
            "RUNNING": "[blue]",
            "COMPLETED": "[green]",
            "PENDING": "[magenta]",
            "FAILED": "[red]",
            "TIMEOUT": "[red]",
        }
        return style.get(value, "") + str(value)
    elif field in ["Start", "End"]:
        if pd.isnull(value):
            return "-"
        else:
            return humanize.naturaldate(value) + " " + value.strftime("%X")
    else:
        return str(value)


@click.command()
@click.option("-u", "--uid", "--user", "user")
@click.option("-a", "--allusers", is_flag=True)
@click.option("-p", "--show-paths", is_flag=True)
@click.option("-U", "--show-unidentified", is_flag=True)
@click.option("-s", "--state")
@click.option("-S", "--starttime")
def status_command(show_paths, show_unidentified, **kwargs):
    """Gives an overview of simulations running on this machine."""
    job_data = fetch_job_data([
        "JobID",
        "User",
        "JobName",
        "AllocCPUS",
        "AllocNodes",
        "Elapsed",
        "End",
        "State",
        "WorkDir",
        "Comment",
    ], **kwargs)

    # Do nothing if job list is empty
    if len(job_data) == 0:
        return

    # List most recent jobs first
    job_data.sort_values("JobID", inplace=True, ascending=False)

    # Get the input file corresponding to each job
    job_data["InputFile"] = [
        get_input_file(comment, work_dir)
        for comment, work_dir in zip(job_data["Comment"], job_data["WorkDir"])
    ]

    # Get the executable name corresponding to each job.
    job_data["ExecutableName"] = [
        get_executable_name(comment,
                            input_file) for comment, input_file in zip(
                                job_data["Comment"], job_data["InputFile"])
    ]

    # Add metadata so jobs can be grouped by state
    job_data["StateOrder"] = job_data["State"].apply(_state_order)

    # Start printing things
    console = rich.console.Console()

    # We'll print these columns
    standard_fields = [
        "State",
        "End",
        "JobID",
        "JobName",
        "Elapsed",
        "AllocCPUS",
        "AllocNodes",
    ]
    if kwargs["allusers"]:
        standard_fields.insert(2, "User")
    # Transform some column names for better readability
    col_names = {
        "AllocCPUS": "Cores",
        "AllocNodes": "Nodes",
    }
    standard_columns = [col_names.get(col, col) for col in standard_fields]

    # Group output by executable
    first_section = True
    for executable_name, exec_data in job_data.groupby("ExecutableName"):
        if first_section:
            first_section = False
        else:
            console.print("")
        console.rule(f"[bold]{executable_name}", align="left")
        executable_status = match_executable_status(executable_name)

        extra_columns = [(field + f" [{unit}]") if unit else field
                         for field, unit in executable_status.fields.items()]
        columns = standard_columns + extra_columns
        table = rich.table.Table(*columns, box=None)

        # Group by job state
        for state_index, data in exec_data.groupby("StateOrder"):
            for _, row in data.iterrows():

                # Extract job status and format row for output
                row_formatted = [
                    _format(field, row[field]) for field in standard_fields
                ]
                with open(row["InputFile"], "r") as open_input_file:
                    try:
                        input_file = yaml.safe_load(open_input_file)
                    except:
                        logger.debug("Unable to load input file.",
                                     exc_info=True)
                        input_file = None
                try:
                    status = executable_status.status(input_file,
                                                      row["WorkDir"])
                except:
                    logger.debug("Unable to extract executable status.",
                                 exc_info=True)
                    status = {}
                row_formatted += [
                    executable_status.format(field, status[field])
                    if field in status else "-"
                    for field in executable_status.fields.keys()
                ]
                table.add_row(*row_formatted)

                # Print paths if requested
                if show_paths:
                    console.print(table)
                    # Print WorkDir in its own line so it wraps nicely in the
                    # terminal and can be copied
                    console.print(" [bold]WorkDir:[/bold] " + row['WorkDir'])
                    console.print(" [bold]InputFile:[/bold] " +
                                  row['InputFile'])
                    table = rich.table.Table(*columns, box=None)
        if not show_paths:
            console.print(table)

    # Output jobs that couldn't be parsed
    unidentified_jobs = job_data[job_data["ExecutableName"].isnull()]
    if len(unidentified_jobs) > 0 and show_unidentified:
        console.print("")
        console.rule("[bold]Unidentified Jobs", align="left")
        table = rich.table.Table(*standard_columns, box=None)
        for i, row in unidentified_jobs.iterrows():
            row_formatted = [
                _format(field, row[field]) for field in standard_fields
            ]
            table.add_row(*row_formatted)
        console.print(table)
