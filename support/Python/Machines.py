# Distributed under the MIT License.
# See LICENSE.txt for details.
"""Support for host machines, such as supercomputers

Machines are defined as YAML files in 'support/Machines/'. To add support for a
new machine, add a YAML file to that directory and define the attributes listed
in the `Machine` class under a `Machine:` key.

Use the `this_machine()` function to retrieve the machine we are currently
running on.

Attributes:
  all_machines (dict[str, Machine]): All available machines.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import yaml
from dataclasses import dataclass

SPECTRE_HOME = "@CMAKE_SOURCE_DIR@"


@dataclass(frozen=True)
class Machine(yaml.YAMLObject):
    """A machine we know how to run on, such as a particular supercomputer

    Attributes:
      Name: A short name for the machine. Must match the YAML file name.
      Description: A description of the machine. Give some basic context and
        any information that may help people get started using the machine.
        Provide links to wiki pages, signup pages, etc., for additional
        information.
      HostnameRegex: A regular expression that identifies the machine from the
        hostname. The regex pattern must be compatible with `re.compile`. It
        must identify the machine uniquely among all available machines. On
        most machines you can run `hostname` in a shell to find the string
        that is matched against this pattern. Alternatively, run:

        ```py
        python -c "import platform; print(platform.node())"
        ```
      Scheduler: The scheduler used to submit jobs to the queue. Typically
        'sbatch'.
      DefaultProcsPerNode: Default number of worker threads spawned per node.
        It is often advised to leave one core per node or socket free for
        communication, so this might be the number of cores or hyperthreads
        per node minus one.
      DefaultQueue: Default queue that jobs are submitted to. On Slurm systems
        you can see the available queues with `sinfo`.
      DefaultTimeLimit: Default wall time limit for submitted jobs. For
        acceptable formats, see: https://slurm.schedmd.com/sbatch.html#OPT_time
    """
    yaml_tag = '!Machine'
    yaml_loader = yaml.SafeLoader
    # The YAML machine files can have these attributes:
    Name: str
    Description: str
    HostnameRegex: re.Pattern
    Scheduler: str
    DefaultProcsPerNode: int
    DefaultQueue: str
    DefaultTimeLimit: str


# Parse YAML machine files as Machine objects
yaml.SafeLoader.add_path_resolver('!Machine', ['Machine'], dict)
# Parse 'HostnameRegex' field as regex
yaml.SafeLoader.add_path_resolver('!python/regexp',
                                  ['Machine', 'HostnameRegex'], str)
yaml.SafeLoader.add_constructor(
    '!python/regexp',
    lambda loader, node: re.compile(loader.construct_scalar(node)))


def _load_machines() -> dict[str, Machine]:
    machinefiles_dir = os.path.join(SPECTRE_HOME, 'support', 'Machines')
    all_machinefiles = [
        filename for filename in os.listdir(machinefiles_dir)
        if filename.endswith('.yaml')
    ]
    machines = {
        machinefile.replace('.yaml', ''):
        yaml.safe_load(open(os.path.join(machinefiles_dir, machinefile),
                            'r'))['Machine']
        for machinefile in all_machinefiles
    }
    return machines


all_machines: dict[str, Machine] = _load_machines()


class UnknownMachineError(Exception):
    """Indicates we were unsuccessful in identifying the current machine"""
    pass


class AmbiguousMachineError(Exception):
    """Indicates the hostname does not uniquely identify the current machine"""
    pass


def this_machine(hostname=platform.node()) -> Machine:
    """Determine the machine we are running on

    Raises `UnknownMachineError` when the machine could not be identified, and
    `AmbiguousMachineError` when multiple machines match the hostname.

    Args:
      hostname: The hostname that identifies the machine. Leave blank to
        determine the current machine's hostname automatically.
    """
    logging.debug(f"Hostname: {hostname}")
    found_machine = None
    for machine in all_machines.values():
        match_hostname = re.match(machine.HostnameRegex, hostname)
        if found_machine and match_hostname:
            raise AmbiguousMachineError(
                f"The hostname '{hostname}' did not uniquely identify a "
                f"machine. It matches both '{found_machine.Name}' and "
                f"'{machine.Name}'. Please make their 'HostnameRegex' entry "
                "more precise.")
        if match_hostname:
            found_machine = machine
    if not found_machine:
        raise UnknownMachineError(
            "We were unable to identify the current machine from the hostname "
            f"'{hostname}'. If you are running on a new machine, please add a "
            "machinefile to 'support/Machines/'.")
    return found_machine
