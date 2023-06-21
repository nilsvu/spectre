# Distributed under the MIT License.
# See LICENSE.txt for details.

import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

from spectre.tools.Status.Status import get_job_data
from spectre.Visualization.ReadH5 import to_dataframe

job_data = get_job_data(
    user=None, allusers=st.checkbox("All users", True), starttime="now-10days"
)
job_data["Name"] = [
    row["User"] + str(row["JobID"]) + row["JobName"]
    for i, row in job_data.iterrows()
]


def _load_dataset(filename, subfile):
    with h5py.File(filename, "r") as open_file:
        subfile = open_file[subfile]
        # Downsample to avoid overwhelming the plotting engine
        num_samples = 1000
        stride = max(1, len(subfile) // num_samples)
        return to_dataframe(subfile, slice=np.s_[::stride])


for executable_name, exec_data in job_data.groupby("ExecutableName"):
    st.title(executable_name)

    if st.checkbox("Show raw data", key=f"show_raw_data_{executable_name}"):
        st.dataframe(exec_data)

    st.metric("Jobs", len(exec_data))

    job_selection = st.multiselect(
        "Jobs",
        exec_data["Name"],
        key=f"jobs_select_{executable_name}",
        default=exec_data["Name"],
    )

    constraint_energy = pd.DataFrame(
        columns=list(exec_data["Name"]), dtype=float
    )

    for i, (input_file_path, work_dir, name) in exec_data[
        ["InputFile", "WorkDir", "Name"]
    ].iterrows():
        if name not in job_selection:
            continue
        if not (input_file_path and work_dir):
            continue
        try:
            with open(input_file_path, "r") as open_input_file:
                metadata, input_file = yaml.safe_load_all(open_input_file)
            reductions_file = os.path.join(
                work_dir, input_file["Observers"]["ReductionFileName"] + ".h5"
            )
            norms = _load_dataset(reductions_file, "Norms.dat").set_index(
                "Time"
            )
            constraint_energy[name] = norms["L2Norm(ConstraintEnergy)"]
        except:
            continue
        # st.dataframe(norms)
        # fig = px.line(norms, x="Time", y="L2Norm(ConstraintEnergy)")
        # st.plotly_chart(fig)
        # charts.append(
        #     alt.Chart(norms)
        #     .mark_line()
        #     .encode(x="Time", y="L2Norm(ConstraintEnergy)")
        # )
    fig = px.line(constraint_energy)
    st.plotly_chart(fig, use_container_width=True)
    # st.altair_chart(alt.layer(*charts), use_container_width=True)
