# Distributed under the MIT License.
# See LICENSE.txt for details.

import glob
import logging
import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectre.Pipelines.EccentricityControl.EccentricityControl import (
    coordinate_separation_eccentricity_control,
)
from spectre.Visualization.GenerateXdmf import generate_xdmf
from spectre.Visualization.PlotControlSystem import plot_control_system
from spectre.Visualization.PlotSizeControl import plot_size_control
from spectre.Visualization.PlotTrajectories import (
    import_A_and_B,
    plot_trajectory,
)
from spectre.Visualization.ReadH5 import to_dataframe

from .ExecutableStatus import EvolutionStatus, list_reduction_files

logger = logging.getLogger(__name__)


class EvolveGhBinaryBlackHole(EvolutionStatus):
    executable_name_patterns = [r"^EvolveGhBinaryBlackHole"]
    fields = {
        "Time": "M",
        "Speed": "M/h",
        "Orbits": None,
        "Separation": "M",
        "3-Index Constraint": None,
    }

    def status(self, input_file, work_dir):
        try:
            reductions_file = input_file["Observers"]["ReductionFileName"]
            open_reductions_file = h5py.File(
                os.path.join(work_dir, reductions_file + ".h5"), "r"
            )
        except:
            logger.debug("Unable to open reductions file.", exc_info=True)
            return {}
        with open_reductions_file:
            result = self.time_status(input_file, open_reductions_file)
            # Number of orbits. We use the rotation control system for this.
            try:
                rotation_z = to_dataframe(
                    open_reductions_file["ControlSystems/Rotation/z.dat"],
                    slice=np.s_[-1:],
                )
                # Assume the initial rotation angle is 0 for now. We can update
                # this to read the initial rotation angle once we can read
                # previous segments / checkpoints of a simulation.
                covered_angle = rotation_z["FunctionOfTime"].iloc[-1]
                result["Orbits"] = covered_angle / (2.0 * np.pi)
            except:
                logger.debug("Unable to extract orbits.", exc_info=True)
            # Euclidean separation between horizons
            try:
                ah_centers = [
                    to_dataframe(
                        open_reductions_file[
                            f"ApparentHorizons/ControlSystemAh{ab}_Centers.dat"
                        ],
                        slice=np.s_[-1:],
                    ).iloc[-1]
                    for ab in "AB"
                ]
                ah_separation = np.sqrt(
                    sum(
                        (
                            ah_centers[0]["InertialCenter" + xyz]
                            - ah_centers[1]["InertialCenter" + xyz]
                        )
                        ** 2
                        for xyz in ["_x", "_y", "_z"]
                    )
                )
                result["Separation"] = ah_separation
            except:
                logger.debug("Unable to extract separation.", exc_info=True)
            # Norms
            try:
                norms = to_dataframe(
                    open_reductions_file["Norms.dat"], slice=np.s_[-1:]
                )
                result["3-Index Constraint"] = norms.iloc[-1][
                    "L2Norm(PointwiseL2Norm(ThreeIndexConstraint))"
                ]
            except:
                logger.debug(
                    "Unable to extract three index constraint.", exc_info=True
                )
        return result

    def format(self, field, value):
        if field == "Separation":
            return f"{value:g}"
        elif field == "Orbits":
            return f"{value:g}"
        elif field == "3-Index Constraint":
            return f"{value:.2e}"
        return super().format(field, value)

    def render_dashboard(self, job: dict, input_file: dict):
        import plotly.express as px
        import streamlit as st

        run_dir = Path(job["WorkDir"])
        reduction_files = list_reduction_files(job=job, input_file=input_file)

        # Constraints
        st.subheader("Constraints")

        def get_constraints_data(reductions_file):
            with h5py.File(reductions_file, "r") as open_h5file:
                return pd.concat(
                    [
                        to_dataframe(
                            open_h5file["ConstraintEnergy.dat"]
                        ).set_index("Time")[["L2Norm(ConstraintEnergy)"]],
                        to_dataframe(open_h5file["Norms.dat"]).set_index(
                            "Time"
                        )[
                            [
                                "L2Norm(PointwiseL2Norm(GaugeConstraint))",
                                "L2Norm(PointwiseL2Norm(ThreeIndexConstraint))",
                                "L2Norm(PointwiseL2Norm(FourIndexConstraint))",
                            ]
                        ],
                    ],
                    axis=1,
                )

        constraints = pd.concat(map(get_constraints_data, reduction_files))
        constraints.sort_index(inplace=True)
        fig = px.line(constraints.iloc[1:], log_y=True)
        fig.update_layout(
            legend=dict(
                title=None,
                orientation="h",
                yanchor="bottom",
                y=1,
                xanchor="left",
                x=0,
            )
        )
        fig.update_yaxes(exponentformat="e", title=None)
        st.plotly_chart(fig)

        # Time steps
        super().render_time_steps(input_file, reduction_files)

        # Trajectories
        st.subheader("Trajectories")
        centers_A, centers_B = import_A_and_B(
            reduction_files,
            "ApparentHorizons/ControlSystemAhA_Centers.dat",
            "ApparentHorizons/ControlSystemAhB_Centers.dat",
        )
        plot_trajectory(centers_A, centers_B, (8, 8))
        fig = plt.gcf()
        st.pyplot(fig)

        # Grid
        st.subheader("Grid")

        @st.experimental_fragment
        def render_grid():
            if st.button("Render grid"):
                from spectre.Visualization.Render3D.Domain import render_domain

                volume_files = glob.glob(str(run_dir / "BbhVolume*.h5"))
                generate_xdmf(
                    volume_files,
                    output=str(run_dir / "Bbh.xmf"),
                    subfile_name="VolumeData",
                )
                render_domain(
                    str(run_dir / "Bbh.xmf"),
                    output=str(run_dir / "domain.png"),
                    slice=True,
                    zoom_factor=50.0,
                    time_step=-1,
                )
            if (run_dir / "domain.png").exists():
                st.image(str(run_dir / "domain.png"))

        render_grid()

        # Control systems
        st.subheader("Control systems")

        @st.experimental_fragment
        def render_control_systems():
            if st.checkbox("Show control systems"):
                st.pyplot(plot_control_system(reduction_files))
                with st.expander("Size control A", expanded=False):
                    plot_size_control(reduction_files, "A")
                    st.pyplot(plt.gcf())
                with st.expander("Size control B", expanded=False):
                    plot_size_control(reduction_files, "A")
                    st.pyplot(plt.gcf())

        render_control_systems()

        # Eccentricity
        st.subheader("Eccentricity")

        @st.experimental_fragment
        def render_eccentricity():
            if st.checkbox("Show eccentricity"):
                ecc_control_result = coordinate_separation_eccentricity_control(
                    reduction_files[0],
                    "ApparentHorizons/ControlSystemAhA_Centers.dat",
                    "ApparentHorizons/ControlSystemAhB_Centers.dat",
                    tmin=60,
                    tmax=1000,
                    angular_velocity_from_xcts=None,
                    expansion_from_xcts=None,
                    output="temp.pdf",
                )["H4"]["fit result"]
                st.pyplot(plt.gcf())
                st.metric(
                    "Eccentricity", f"{ecc_control_result['eccentricity']:e}"
                )
                st.write(ecc_control_result["xcts updates"])

        render_eccentricity()
