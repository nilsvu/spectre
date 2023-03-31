# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import os
import shutil
import unittest

import numpy as np
import spectre.IO.H5 as spectre_h5
from click.testing import CliRunner
from spectre.Informer import unit_test_build_path

from spectre.tools.Status import status_command
from spectre.tools.Status.ExecutableStatus import (EvolutionStatus,
                                                   match_executable_status)


class TestStatus(unittest.TestCase):
    def setUp(self):
        self.input_file = {
            "Observers": {
                "ReductionFileName": "Reductions"
            },
            "EventsAndTriggers": [("Always", [{
                "ObserveTimeStep": {
                    "SubfileName": "TimeSteps"
                }
            }])]
        }
        self.work_dir = os.path.join(unit_test_build_path(), "tools/Status")
        shutil.rmtree(self.work_dir, ignore_errors=True)
        os.makedirs(self.work_dir, exist_ok=True)
        with spectre_h5.H5File(os.path.join(self.work_dir, "Reductions.h5"),
                               "w") as open_h5_file:
            # Time data
            time_subfile = open_h5_file.insert_dat(
                "/TimeSteps",
                legend=["Time", "Minimum Walltime", "Maximum Walltime"],
                version=0)
            time_subfile.append([0., 1., 2.])
            time_subfile.append([1., 2., 3.])
            open_h5_file.close_current_object()
            # Control systems data
            rotation_subfile = open_h5_file.insert_dat(
                "/ControlSystems/Rotation/z", legend=["Lambda"], version=0)
            rotation_subfile.append([0.])
            rotation_subfile.append([np.pi])
            open_h5_file.close_current_object()
            # AH data
            aha_subfile = open_h5_file.insert_dat(
                "/ApparentHorizons/ControlSystemAhA_Centers",
                legend=[
                    "InertialCenter_x", "InertialCenter_y", "InertialCenter_z"
                ],
                version=0)
            aha_subfile.append([1., 0., 0.])
            open_h5_file.close_current_object()
            ahb_subfile = open_h5_file.insert_dat(
                "/ApparentHorizons/ControlSystemAhB_Centers",
                legend=[
                    "InertialCenter_x", "InertialCenter_y", "InertialCenter_z"
                ],
                version=0)
            ahb_subfile.append([-1., 0., 0.])
            open_h5_file.close_current_object()
            # Constraints
            constraints_subfile = open_h5_file.insert_dat(
                "/Norms", legend=["L2Norm(ConstraintEnergy)"], version=0)
            constraints_subfile.append([1.e-3])

    def tearDown(self):
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_evolution_status(self):
        executable_status = match_executable_status("EvolveSomething")
        status = executable_status.status(self.input_file, self.work_dir)
        self.assertEqual(status, {"Time": 1., "Speed": 3600.})
        self.assertEqual(executable_status.format("Time", 1.5), "1.5")
        self.assertEqual(executable_status.format("Speed", 1.2), "1.2")

    def test_evolve_bbh_status(self):
        executable_status = match_executable_status("EvolveGhBinaryBlackHole")
        status = executable_status.status(self.input_file, self.work_dir)
        self.assertEqual(status["Time"], 1.)
        self.assertEqual(status["Speed"], 3600.)
        self.assertEqual(status["Orbits"], 0.5)
        self.assertEqual(status["Separation"], 2.)
        self.assertEqual(status["Constraint Energy"], 1.e-3)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main(verbosity=2)
