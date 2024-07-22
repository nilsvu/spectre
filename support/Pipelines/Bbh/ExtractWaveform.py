# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
from pathlib import Path
from typing import Optional, Union, Sequence

import click

from spectre.support.Schedule import schedule, scheduler_options

logger = logging.getLogger(__name__)

CCE_INPUT_FILE_TEMPLATE = Path(__file__).parent / "ExtractWaveform.yaml"


def extract_waveform(
    boundary_data_files: Sequence[Union[str, Path]],
    cce_input_file_template: Union[str, Path] = CCE_INPUT_FILE_TEMPLATE,
    pipeline_dir: Optional[Union[str, Path]] = None,
    run_dir: Optional[Union[str, Path]] = None,
    segments_dir: Optional[Union[str, Path]] = None,
    **scheduler_kwargs,
):
    """Extract waveform with Cauchy-characteristic extraction (CCE)

    Point the BOUNDARY_DATA_FILES to the Bondi-Sachs output data files from the
    Cauchy evolution. You can collect the files from multiple segments or from
    separate inspiral and ringdown directories using a glob pattern, e.g.
    'path/to/Segment_*/BondiSachsCceR0200.h5'. The files will be combined and
    passed to the CCE evolution.

    Parameters for the CCE evolution are inserted into the
    'cce_input_file_template'. The remaining options are forwarded to the
    'schedule' command. See 'schedule' docs for details.
    """
    logger.warning(
        "The BBH pipeline is still experimental. Please review the"
        " generated input files."
    )

    # Resolve directories
    if pipeline_dir:
        pipeline_dir = Path(pipeline_dir).resolve()
    assert segments_dir is None, (
        "CCE doesn't use segments at the moment. Specify"
        " '--run-dir' / '-o' or '--pipeline-dir' / '-d' instead."
    )
    if pipeline_dir and not run_dir:
        run_dir = pipeline_dir / "003_Cce"

    # Determine CCE parameters
    assert (
        len(boundary_data_files) == 1
    ), "Joining multiple files is not yet implemented."
    cce_params = {
        "BoundaryDataFiles": boundary_data_files[0],
    }

    # Schedule!
    return schedule(
        cce_input_file_template,
        **cce_params,
        **scheduler_kwargs,
        pipeline_dir=pipeline_dir,
        run_dir=run_dir,
        segments_dir=segments_dir,
    )


@click.command(name="extract-waveform", help=extract_waveform.__doc__)
@click.argument(
    "boundary_data_files",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
    nargs=-1,
    required=True,
)
@click.option(
    "--cce-input-file-template",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
    default=CCE_INPUT_FILE_TEMPLATE,
    help="Input file template for the CCE evolution.",
    show_default=True,
)
@click.option(
    "--pipeline-dir",
    "-d",
    type=click.Path(
        writable=True,
        path_type=Path,
    ),
    help="Directory where steps in the pipeline are created.",
)
@scheduler_options
def extract_waveform_command(**kwargs):
    _rich_traceback_guard = True  # Hide traceback until here
    extract_waveform(**kwargs)


if __name__ == "__main__":
    extract_waveform_command(help_option_names=["-h", "--help"])
