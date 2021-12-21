# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import os
import re
import rich.logging
import rich.traceback
import subprocess
from spectre.support.Machines import this_machine, UnknownMachineError
from spectre.support.Schedule import schedule

SPECTRE_HOME = "@CMAKE_SOURCE_DIR@"
SPECTRE_BUILD_DIR = "@CMAKE_BINARY_DIR@"
SPECTRE_VERSION = "@SPECTRE_VERSION@"


def _parse_param(value):
    if not isinstance(value, str):
        return value
    value = value.strip()
    # Exponent prefix: 2**x or 10**x, where x is parsed recursively
    match = re.match(r'(\d+)[*]{2}(.+)$', value)
    if match:
        logging.debug(f"'{value}' is exponentiated")
        base = int(match.group(1))
        exponent = _parse_param(match.group(2))
        try:
            return base**exponent
        except TypeError:
            return [base**exponent_i for exponent_i in exponent]
    # List
    value_list = value.strip(',').split(',')
    if len(value_list) > 1:
        logging.debug(f"'{value}' is a list")
        return [_parse_param(element.strip()) for element in value_list]
    # Exclusive range: 0..3 or 0..<3 (the latter is clearer, but the '<' is a
    # special character in the shell)
    match = re.match(r'(\d+)[.]{2}[<]?(\d+)$', value)
    if match:
        logging.debug(f"'{value}' is an exclusive range")
        return range(int(match.group(1)), int(match.group(2)))
    # Inclusive range: 0...3
    match = re.match(r'(\d+)[.]{3}(\d+)$', value)
    if match:
        logging.debug(f"'{value}' is an inclusive range")
        return range(int(match.group(1)), int(match.group(2)) + 1)
    # Integers
    match = re.match(r'(\d+)$', value)
    if match:
        logging.debug(f"'{value}' is an int")
        return int(match.group(1))
    # Floats
    match = re.match(r'(\d+[.]\d*)$', value)
    if match:
        logging.debug(f"'{value}' is a float")
        return float(match.group(1))
    return value


def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        handlers=[rich.logging.RichHandler(rich_tracebacks=True)],
        format="%(message)s",
        datefmt="[%X]")
    rich.traceback.install(show_locals=True)

    try:
        machine = this_machine()
    except UnknownMachineError:
        machine = None

    class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter,
                        argparse.RawDescriptionHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        prog='spectre',
        description=
        (f"SpECTRE version: {SPECTRE_VERSION}\n"
         f"Installed from: {SPECTRE_BUILD_DIR}\n" +
         (f"Running on machine: {machine.Name}" if machine else
          "\nNot running on a known machine, so scheduling is unavailable. See "
          "'spectre.support.Machines' for documentation on adding support for "
          "more machines.")),
        formatter_class=HelpFormatter)
    parser.add_argument('--version', action='version', version=SPECTRE_VERSION)

    subparsers = parser.add_subparsers(help="Subprograms",
                                       required=True,
                                       dest='subparser')

    parent_parser = argparse.ArgumentParser(add_help=False,
                                            formatter_class=HelpFormatter)
    parent_parser.add_argument(
        'input_file_template',
        help=
        ("Path to an input file. It will be copied to the 'run_dir'. It "
         "can be a [Jinja template](https://jinja.palletsprojects.com/en/3.0.x/templates/) "
         "(see main help text for possible placeholders)."))
    parent_parser.add_argument(
        '--run-dir',
        '-o',
        required=False,
        help=("The directory to which input file, submit script, etc. are "
              "copied, relative to which the executable will run, and to "
              "which output files are written."))
    parent_parser.add_argument(
        '--job-name',
        required=False,
        help=("A string descripting the job (see main help text)."))
    parent_parser.add_argument(
        '--build-dir',
        required=False,
        help=("Path to the build directory which is used to launch the "
              "executable (see main help text)."))
    parent_parser.add_argument(
        '--clean-output',
        action='store_true',
        help=("Clean up existing output files in the 'run_dir' (see main "
              "help text)."))
    parent_parser.add_argument(
        '--force',
        '-f',
        action='store_true',
        help=("Overwrite existing files in the `run_dir`. You may also want "
              "to use '--clean-output'."))
    parent_parser.add_argument(
        '--param',
        '-p',
        help=("Forward additional parameters to input file and submit script "
              "templates."),
        action='append',
        type=lambda kv: kv.split('='),
        dest='params',
        default=[])
    parent_parser.add_argument(
        '--num-procs',
        '-j',
        help=("Number of worker threads (see main help text)."))
    parent_parser.add_argument('--num-nodes',
                               '-N',
                               help=("Number of nodes (see main help text)"))
    parent_parser.add_argument(
        '--procs-per-node',
        '--ppn',
        default=machine.DefaultProcsPerNode if machine else None,
        help=("Number of worker threads per node (see main help text)."))

    parser_run = subparsers.add_parser(
        'run',
        parents=[parent_parser],
        help=("Run an executable directly (no scheduler)"),
        description=(
            "Run an executable directly. You can also use the "
            "'spectre.support.Schedule.schedule' function to do so "
            "programmatically:\n\nspectre.support.Schedule.schedule:\n\n" +
            schedule.__doc__),
        formatter_class=HelpFormatter)
    parser_run.set_defaults(subprogram=schedule, scheduler=None)

    if machine:
        parser_schedule = subparsers.add_parser(
            'schedule',
            parents=[parent_parser],
            help=("Submit a run to the queue on this machine"),
            description=(
                "Submit a run to the queue, using the scheduler "
                f"'{machine.Scheduler}' on this machine by default "
                f"(detected machine: '{machine.Name}'). You can also use the "
                "'spectre.support.Schedule.schedule' function to do so "
                "programmatically:\n\nspectre.support.Schedule.schedule:\n\n" +
                schedule.__doc__),
            formatter_class=HelpFormatter)
        parser_schedule.set_defaults(subprogram=schedule)
        parser_schedule.add_argument(
            '--submit-script-template',
            required=False,
            default=os.path.join(SPECTRE_HOME, 'support', 'SubmitScripts',
                                 machine.Name + '.sh'),
            help=
            ("Path to a submit script. It will be copied to the 'run_dir'. It "
             "can be a [Jinja template](https://jinja.palletsprojects.com/en/3.0.x/templates/) "
             "(see main help text for possible placeholders)."))
        parser_schedule.add_argument('--queue',
                                     required=False,
                                     default=machine.DefaultQueue,
                                     help=("Name of the queue."))
        parser_schedule.add_argument(
            '--time-limit',
            required=False,
            default=machine.DefaultTimeLimit,
            help=(
                "Wall time limit. Must be compatible with the chosen queue."))
        parser_schedule.add_argument(
            '--scheduler',
            required=False,
            default=machine.Scheduler,
            help=("Scheduler used to submit the job."))

    args = parser.parse_args()
    subprogram = args.subprogram
    del args.subprogram
    del args.subparser

    # Parse additional parameters
    args.num_procs = _parse_param(args.num_procs)
    args.num_nodes = _parse_param(args.num_nodes)
    args.procs_per_node = _parse_param(args.procs_per_node)
    params = args.params
    del args.params
    for key, value in params:
        setattr(args, key, _parse_param(value))

    subprogram(**vars(args))


if __name__ == '__main__':
    main()
