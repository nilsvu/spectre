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


# The total sum from m_max+1 to infinity in Eq. 5.1
def sum_Fr4(r0, a, m_max, M=1.0):
    return (M**2*(11520*a**13*M**2*(M - 8*r0)*r0 - \
23040*a**14*M**2*np.sqrt(M*r0) + \
384*a**12*M*r0*np.sqrt(M*r0)*(461*M**2 - 81*M*r0 - 360*r0**2) - \
192*a**11*M*r0**2*(307*M**3 - 3780*M**2*r0 + 1233*M*r0**2 + \
480*r0**3) + 32*a**9*M*r0**3*(2835*M**4 - 69401*M**3*r0 + \
46565*M**2*r0**2 + 13779*M*r0**3 - 8748*r0**4) - \
64*a**10*r0**2*np.sqrt(M*r0)*(8549*M**4 - 3593*M**3*r0 - \
16212*M**2*r0**2 + 6336*M*r0**3 + 360*r0**4) + \
192*a**8*r0**3*np.sqrt(M*r0)*(4470*M**5 - 3621*M**4*r0 - \
15645*M**3*r0**2 + 12662*M**2*r0**3 - 1529*M*r0**4 - 342*r0**5) + \
9*r0**11*np.sqrt(M*r0)*(25375*M**5 - 47015*M**4*r0 + 29014*M**3*r0**2 \
- 4814*M**2*r0**3 - 1365*M*r0**4 + 405*r0**5) + \
36*a*r0**10*(25375*M**6 - 47369*M**5*r0 + 31856*M**4*r0**2 - \
8692*M**3*r0**3 + 705*M**2*r0**4 - 75*M*r0**5 + 40*r0**6) + \
16*a**7*r0**4*(-1479*M**6 + 210966*M**5*r0 - 224760*M**4*r0**2 - \
49213*M**3*r0**3 + 93619*M**2*r0**4 - 19953*M*r0**5 + 180*r0**6) + \
18*a**2*r0**8*np.sqrt(M*r0)*(76125*M**6 - 145182*M**5*r0 + \
70771*M**4*r0**2 + 16696*M**3*r0**3 - 19905*M**2*r0**4 + 3190*M*r0**5 \
+ 225*r0**6) - 16*a**6*r0**4*np.sqrt(M*r0)*(43101*M**6 - \
61443*M**5*r0 - 271980*M**4*r0**2 + 343776*M**3*r0**3 - \
100489*M**2*r0**4 - 7763*M*r0**5 + 4158*r0**6) + \
3*a**4*r0**5*np.sqrt(M*r0)*(76125*M**7 - 176307*M**6*r0 - \
1157559*M**5*r0**2 + 1949709*M**4*r0**3 - 855873*M**3*r0**4 - \
26505*M**2*r0**5 + 76235*M*r0**6 - 8065*r0**7) + \
12*a**3*r0**7*(76125*M**7 - 152637*M**6*r0 - 93174*M**5*r0**2 + \
281414*M**4*r0**3 - 166063*M**3*r0**4 + 36555*M**2*r0**5 - \
3480*M*r0**6 + 460*r0**7) + 12*a**5*r0**5*(-2367*M**7 - \
221220*M**6*r0 + 337457*M**5*r0**2 + 71894*M**4*r0**3 - \
262111*M**3*r0**4 + 111498*M**2*r0**5 - 14459*M*r0**6 + \
588*r0**7)))/(960.*np.pi*r0**9*(a*M + \
r0*np.sqrt(M*r0))*(2*a*np.sqrt(M*r0) + r0*(-3*M + r0))**3.5*(a**2 + \
r0*(-2*M + r0))**1.5*3.0*(m_max-0.5)*(m_max+0.5)*(m_max+1.5))    


def assemble_scalar_self_force(reduction_files: float):
    df = pd.DataFrame(columns=["m", "SelfForce_r"]).set_index("m")
    for m_mode, reduction_file in enumerate(reduction_files):
        with h5py.File(reduction_file, "r") as open_h5_file:
            # print(reduction_file, to_dataframe(open_h5_file["SelfForce.dat"]))
            df.loc[m_mode, "SelfForce_r"] = to_dataframe(
                open_h5_file["SelfForce.dat"]
            ).iloc[-1]["Re(SelfForce_r)"]
    self_force = df["SelfForce_r"].sum()
    print(df)
    print("Before regularization:", self_force)
    
    # Regularization
    m_max = len(reduction_files) - 1
    self_force += sum_Fr4(r0=10.0, a=0.5, m_max=m_max)

    print(self_force)


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
