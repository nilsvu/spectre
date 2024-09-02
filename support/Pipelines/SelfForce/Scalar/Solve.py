# Distributed under the MIT License.
# See LICENSE.txt for details.

import functools
import logging
import multiprocessing
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union
from scipy.optimize import minimize

import click
import numpy as np
from rich.pretty import pretty_repr

from spectre.support.Schedule import schedule
from spectre.PointwiseFunctions.AnalyticData.ScalarSelfForce import (
    CircularOrbit,
)

logger = logging.getLogger(__name__)

INPUT_FILE_TEMPLATE = Path(__file__).parent / "ScalarSelfForce.yaml"


def solve_m_mode(
    m_mode: int,
    pipeline_dir: Union[str, Path],
    input_file_template: Union[str, Path] = INPUT_FILE_TEMPLATE,
    **params: float,
):
    schedule(
        input_file_template,
        m_mode=m_mode,
        **params,
        L_left=m_mode,
        L_right=2 + m_mode,
        L_angular=0,
        P=4,
        scheduler=None,
        run_dir=pipeline_dir / f"m_{m_mode}",
    )


def solve_scalar_self_force(
    spin: float,
    orbital_radius: float,
    num_modes: int,
    pipeline_dir: Union[str, Path],
    inner_radius: float = -50,
    outer_radius: float = 300,
    worldtube_size: Tuple[float, float] = (7.5, 0.5),
    input_file_template: Union[str, Path] = INPUT_FILE_TEMPLATE,
):
    # Resolve directories
    pipeline_dir = Path(pipeline_dir).resolve()

    if orbital_radius == 0.0:
        orbital_radius = minimize(
            lambda r_isco: np.abs(
                1.0
                - 6.0 / r_isco
                + 8.0 * spin * np.sqrt(1.0 / r_isco**3)
                - 3.0 * spin**2 / r_isco**2
            ),
            x0=6.0,
            bounds=[[2.0, 12.0]],
        ).x[0]
        logger.info(f"Orbital radius at ISCO is {orbital_radius}.")

    # Compute puncture position in code coordinates (r_*, cos(theta))
    puncture_position = CircularOrbit(
        black_hole_mass=1,
        black_hole_spin=spin,
        orbital_radius=orbital_radius,
        m_mode_number=0,
    ).puncture_position()[0]
    logger.info(f"Puncture is at tortoise coordinate {puncture_position}.")

    solve_m_mode_partial = functools.partial(
        solve_m_mode,
        pipeline_dir=pipeline_dir,
        input_file_template=input_file_template,
        spin=spin,
        orbital_radius=orbital_radius,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        # FIXME: This avoids evaluating the `effsource` code exactly at the
        # puncture. However, that should be ok unless we try to evaluate the
        # singular field.
        puncture_position=round(puncture_position, 3),
        worldtube_radius=worldtube_size[0],
        worldtube_dcostheta=worldtube_size[1],
    )
    # with multiprocessing.Pool() as pool:
    #     pool.map(solve_m_mode_partial, range(num_modes))
    for m_mode in range(num_modes):
        solve_m_mode_partial(m_mode)


@click.command(name="solve", help=solve_scalar_self_force.__doc__)
@click.option(
    "--spin",
    "-a",
    type=click.FloatRange(0.0, 1.0),
    help="Spin of the black hole around which the scalar charge orbits.",
    required=True,
)
@click.option(
    "--orbital-radius",
    "-r0",
    type=click.FloatRange(0.0, None),
    help="Orbital Boyer-Lindquist radius of the scalar charge.",
    required=True,
)
@click.option(
    "--num-modes",
    "-n",
    type=click.IntRange(1, None),
    help="Number of m-modes.",
    required=True,
)
@click.option(
    "--inner-radius",
    "-r",
    type=float,
    default=-50,
    help=(
        "Radius of the inner boundary near the black hole horizon in tortoise"
        " coordinates, where -inf is the horizon."
    ),
    show_default=True,
)
@click.option(
    "--outer-radius",
    "-R",
    type=float,
    default=350,
    help="Radius of the outer boundary in tortoise coordinates.",
    show_default=True,
)
@click.option(
    "--worldtube-size",
    "-w",
    type=float,
    nargs=2,
    default=(7.5, 0.5),
    help=(
        "Size of the worldtube within which we regularize the field, in"
        " (r_*, cos(theta))."
    ),
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
@click.option(
    "--input-file-template",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
    default=INPUT_FILE_TEMPLATE,
    help="Input file template for the solve.",
    show_default=True,
)
def solve_scalar_self_force_command(**kwargs):
    _rich_traceback_guard = True  # Hide traceback until here
    solve_scalar_self_force(**kwargs)


if __name__ == "__main__":
    solve_scalar_self_force_command(help_option_names=["-h", "--help"])
