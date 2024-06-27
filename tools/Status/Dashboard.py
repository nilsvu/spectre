# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
from functools import partial
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
import yaml
from streamlit_autorefresh import st_autorefresh

from spectre.tools.Status.Status import (
    AVAILABLE_COLUMNS,
    DEFAULT_COLUMNS,
    _format,
    fetch_job_data,
    get_input_file,
    match_executable_status,
    status,
)
from spectre.Visualization.PlotTrajectories import (
    import_A_and_B,
    plot_trajectory,
)
from spectre.Visualization.ReadH5 import to_dataframe

logger = logging.getLogger(__name__)

plt.style.use(
    Path(__file__).resolve().parent
    / "../../src/Visualization/Python/plots.mplstyle"
)


def page(row, executable_status):
    run_dir = Path(row["WorkDir"])
    st.header(row["JobName"])
    st.write(run_dir)

    # Print README
    if row["SegmentsDir"]:
        readme_file = Path(row["SegmentsDir"]) / "../README.md"
        if readme_file.exists():
            st.markdown(readme_file.read_text())

    columns = list(DEFAULT_COLUMNS)
    columns.remove("JobName")
    st.table(pd.DataFrame([row[columns]]).set_index("JobID"))

    with open(row["InputFile"], "r") as open_input_file:
        _, input_file = yaml.safe_load_all(open_input_file)
    reductions_file = run_dir / "BbhReductions.h5"

    # Status metrics
    status = executable_status.status(input_file, row["WorkDir"])
    for (field, unit), col in zip(
        executable_status.fields.items(),
        st.columns(len(executable_status.fields)),
    ):
        with col:
            st.metric(
                (field + f" [{unit}]") if unit else field,
                (
                    executable_status.format(field, status[field])
                    if field in status
                    else "-"
                ),
            )

    # Constraints
    with h5py.File(reductions_file, "r") as open_h5file:
        constraints = to_dataframe(open_h5file["ConstraintEnergy.dat"])
    fig = px.line(
        constraints, x="Time", y="L2Norm(ConstraintEnergy)", log_y=True
    )
    fig.update_yaxes(exponentformat="e")
    st.plotly_chart(fig)

    # Trajectories
    traj_A, traj_B = import_A_and_B(
        [reductions_file],
        "ApparentHorizons/ControlSystemAhA_Centers.dat",
        "ApparentHorizons/ControlSystemAhB_Centers.dat",
    )
    plot_trajectory(traj_A, traj_B, (8, 8))
    fig = plt.gcf()
    st.pyplot(fig)

    # TODO: plot 3d slice of the domain near horizons

    # Input file
    with st.expander(Path(row["InputFile"]).name, expanded=False):
        with open(row["InputFile"], "r") as open_input_file:
            st.code(
                "".join(open_input_file.readlines()),
                language="yaml",
                line_numbers=True,
            )

    # Outfile
    outfile = run_dir / "spectre.out"
    if outfile.exists():
        with st.expander(outfile.name, expanded=False):
            with st.container(border=False):
                with open(outfile, "r") as open_outfile:
                    st.code("".join(["...\n"] + open_outfile.readlines()[-30:]))


job_data = status(user=None, allusers=False)

pages = []

for executable_name, exec_data in job_data.groupby("ExecutableName"):
    executable_status = match_executable_status(executable_name)

    for _, row in exec_data.iterrows():
        pages.append(
            st.Page(
                partial(page, row=row, executable_status=executable_status),
                title=f"{row['JobName']} ({row['JobID']})",
                url_path=f"/{row['JobID']}",
            )
        )

pg = st.navigation(pages)
pg.run()

count = st_autorefresh(interval=10000)
