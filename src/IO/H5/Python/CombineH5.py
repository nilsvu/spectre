# Distributed under the MIT License.
# See LICENSE.txt for details.

import click

from spectre.IO.H5.CombineH5Dat import combine_h5_dat_command
from spectre.IO.H5.CombineH5Vol import combine_h5_vol_command


@click.group(name="combine-h5")
def combine_h5_command():
    """Combines multiple HDF5 files"""
    pass


combine_h5_command.add_command(combine_h5_dat_command, name="dat")
combine_h5_command.add_command(combine_h5_vol_command, name="vol")


if __name__ == "__main__":
    combine_h5_command(help_option_names=["-h", "--help"])
