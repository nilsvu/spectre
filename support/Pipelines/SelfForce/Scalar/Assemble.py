# Distributed under the MIT License.
# See LICENSE.txt for details.

import functools
import logging
import multiprocessing
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union
import pandas as pd

import click
import numpy as np
import h5py

from spectre.support.Schedule import schedule
from spectre.Visualization.ReadH5 import to_dataframe

logger = logging.getLogger(__name__)


def assemble_scalar_self_force(reduction_files: float):
    self_force = pd.DataFrame(columns=["m", "SelfForce_r"]).set_index("m")
    for m_mode, reduction_file in enumerate(reduction_files):
        with h5py.File(reduction_file, "r") as open_h5_file:
            print(reduction_file, to_dataframe(open_h5_file["SelfForce.dat"]))
            # self_force.loc[m_mode, "SelfForce_r"] = to_dataframe(
            #     open_h5_file["SelfForce.dat"]
            # ).iloc[-1]["Re(SelfForce_r)"]
    # print(self_force)
    # print(self_force["SelfForce_r"].cumsum())
    # print(self_force["SelfForce_r"].sum())


@click.command(name="assemble", help=assemble_scalar_self_force.__doc__)
@click.argument(
    "reduction_files",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    nargs=-1,
)
def assemble_scalar_self_force_command(**kwargs):
    _rich_traceback_guard = True  # Hide traceback until here
    assemble_scalar_self_force(**kwargs)


if __name__ == "__main__":
    assemble_scalar_self_force_command(help_option_names=["-h", "--help"])
