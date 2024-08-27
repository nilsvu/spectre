# Distributed under the MIT License.
# See LICENSE.txt for details.

import argparse
import glob
import logging
import os
import re
import sys
import unittest

import h5py
import numpy as np
import numpy.testing as npt
import yaml


class H5Check:
    """Describes a particular comparison between H5 datasets or groups

    When passed a `unittest.testcase` object to the `perform_checks`,
    the described checks are performed on the H5 files in the appointed run
    directory

    Attributes:
        unit_test: The `unittest.testcase` object, used to invoke asserts
        label: An identifier string for the test
        file_glob: The shell glob matching the h5 files to test
        subfile: The h5 path for the group or dataset to check
        expected_data_subfile: The h5 path for the expected group or dataset
        expected_data: The expected data to compare with directly. Mutually
          exclusive with `expected_data_subfile`.
        absolute_tolerance: The absolute tolerance for approximation checks
        relative_tolerance: The relative tolerance for approximation checks
    """

    def __init__(
        self,
        unit_test,
        label,
        file_glob,
        subfile,
        absolute_tolerance,
        relative_tolerance=None,
        expected_data_subfile=None,
        expected_data=None,
        skip_columns=None,
    ):
        """Initializer for H5Check

        Note: the `unit_test` argument must be the unit test object -- this
        class is to be constructed within a test case, and must have access
        to the calling unit test object so that it can access the assert
        and subTest member functions
        """
        self.unit_test = unit_test
        self.h5_glob = file_glob
        self.test_h5_label = label
        self.test_h5_entity = subfile
        # The `expected` entity, if specified, is what the `test` entity is
        # compared to. For example, one may wish to compare the L2Norm of the
        # scalar field to an expected value from an analytic solution. If an
        # `expected` entity is not specified then the `test` entity is compared
        # to 0.0. This is what would be typically done for error measures, for
        # example.
        self.expected_h5_entity = expected_data_subfile
        if expected_data:
            assert (
                self.expected_h5_entity is None
            ), "Set expected subfile or expected data, not both."
        self.expected_data = expected_data
        self.absolute_tolerance = float(absolute_tolerance)
        self.relative_tolerance = (
            0.0 if relative_tolerance is None else float(relative_tolerance)
        )
        self.skip_columns = [] if skip_columns is None else skip_columns

    def check_h5_file(self, h5_file, test_entity, expected_entity):
        """Perform the unit test comparisons between the dataset or group

        If `expected_entity` is `None`, then the `test_entity` is compared to
        the `expected_data` or 0.0 instead.

        Args:
            h5_file: An h5 File object on which to perform checks
            test_entity: The h5 path string to the test dataset or group
            expected_entity: The h5 path to the expected dataset or group
        """

        if isinstance(h5_file[test_entity], h5py.Dataset):
            logging.info("Checking dataset : " + test_entity)
            np.set_printoptions(threshold=sys.maxsize, precision=16)
            with self.unit_test.subTest(
                test_entity=test_entity, expected_entity=expected_entity
            ):
                test_data = h5_file[test_entity][()]
                column_mask = [
                    x not in self.skip_columns
                    for x in range(test_data.shape[1])
                ]
                if self.expected_h5_entity is not None:
                    expected_data = h5_file[expected_entity][()]
                    self.unit_test.assertEqual(
                        test_data.shape,
                        expected_data.shape,
                        (
                            "test and expected h5 datasets must have identical"
                            " sizes."
                        ),
                    )
                    self.unit_test.assertEqual(
                        test_data.dtype,
                        expected_data.dtype,
                        (
                            "test and expected h5 datasets must have identical"
                            " types."
                        ),
                    )
                    if test_data.dtype == float or test_data.dtype == complex:
                        try:
                            npt.assert_allclose(
                                test_data[:, column_mask],
                                expected_data[:, column_mask],
                                rtol=self.relative_tolerance,
                                atol=self.absolute_tolerance,
                            )
                        except Exception as e:
                            print("Numpy exception message:")
                            print(e)
                            print("Full arrays")
                            print("ACTUAL")
                            actual = test_data[:, column_mask]
                            print(actual)
                            print("DESIRED")
                            desired = np.array(expected_data[:, column_mask])
                            print(desired)
                            for i in range(len(actual)):
                                for j in range(len(actual[0])):
                                    print(f"index: i={i},j={j}")
                                    npt.assert_allclose(
                                        np.array([actual[i][j]]),
                                        np.array([desired[i][j]]),
                                        rtol=self.relative_tolerance,
                                        atol=self.absolute_tolerance,
                                    )
                    else:
                        self.unit_test.assertEqual(test_data, expected_data)
                else:
                    if test_data.dtype == float or test_data.dtype == complex:
                        try:
                            npt.assert_allclose(
                                test_data[:, column_mask],
                                self.expected_data or 0.0,
                                rtol=self.relative_tolerance,
                                atol=self.absolute_tolerance,
                            )
                        except Exception as e:
                            print("Numpy exception message:")
                            print(e)
                            print("Full arrays")
                            print("ACTUAL")
                            actual = test_data[:, column_mask]
                            print(actual)
                            print("DESIRED")
                            desired = (
                                np.full_like(actual, 0.0)
                                if self.expected_data is None
                                else np.array(self.expected_data)
                            )
                            print(desired)
                            for i in range(len(actual)):
                                for j in range(len(actual[0])):
                                    print(f"index: i={i},j={j}")
                                    npt.assert_allclose(
                                        np.array([actual[i][j]]),
                                        np.array([desired[i][j]]),
                                        rtol=self.relative_tolerance,
                                        atol=self.absolute_tolerance,
                                    )
                    else:
                        self.assertTrue(
                            False,
                            msg=(
                                "cannot test non-numeric data without an"
                                " expected data set to compare against"
                            ),
                        )
        elif isinstance(h5_file[test_entity], h5py.Group):
            test_keys = set(h5_file[self.test_entity].keys())
            checks_passed = True
            if self.expected_h5_entity is not None:
                expected_keys = set(h5_file[self.expected_entity].keys())
                keys_difference = test_keys ^ expected_keys
                self.unit_test.assertEqual(
                    keys_difference,
                    {},
                    "test and expected h5 groups must have identical"
                    + "subgroups and data members. Found differences: "
                    + str(keys_difference)
                    + "\nin comparing group "
                    + test_entity
                    + " to group "
                    + expected_entity,
                )
                for key in test_keys:
                    checks_passed = checks_passed and self.check_h5_file(
                        h5_file,
                        test_entity + "/" + key,
                        expected_entity + "/" + key,
                    )
            else:
                for key in test_keys:
                    checks_passed = checks_passed and self.check_h5_file(
                        h5_file, test_entity + "/" + key, None
                    )
            return checks_passed

    def perform_check(self, run_directory):
        """Apply the h5 check to every h5 file within the `run_directory`
        that matches the glob `self.h5_glob`.

        If neither the expected nor test objects are present in an h5 file,
        that file is skipped. However, if none of the matching h5 files have
        the requested datasets, the test fails, under the assumption that
        failure to produce any of the anticipated data should be regarded
        as an error in the executable.
        """
        found_test_entity = False
        found_expected_entity = False
        files_and_entities = ""
        logging.info(
            "Performing checks: " + os.path.join(run_directory, self.h5_glob)
        )
        for filename in glob.glob(os.path.join(run_directory, self.h5_glob)):
            logging.info("Checking file: " + filename)
            with self.unit_test.subTest(filename=filename):
                with h5py.File(filename, "r") as check_h5:
                    files_and_entities = (
                        files_and_entities
                        + filename
                        + ": "
                        + str(list(check_h5.keys()))
                        + "\n"
                    )
                    found_test_entity = found_test_entity or (
                        self.test_h5_entity in check_h5
                    )
                    found_expected_entity = found_expected_entity or (
                        self.expected_h5_entity is not None
                        and self.expected_h5_entity in check_h5
                    )

                    if self.test_h5_entity in check_h5 or (
                        self.expected_h5_entity is not None
                        and self.expected_h5_entity in check_h5
                    ):
                        self.unit_test.assertTrue(
                            self.test_h5_entity in check_h5
                        )
                        self.unit_test.assertTrue(
                            self.expected_h5_entity is None
                            or self.expected_h5_entity in check_h5
                        )
                        self.check_h5_file(
                            check_h5,
                            self.test_h5_entity,
                            self.expected_h5_entity,
                        )
        self.unit_test.assertTrue(
            found_test_entity,
            "Failed to find the subfile '"
            + self.test_h5_entity
            + "' using glob:\n"
            + os.path.join(run_directory, self.h5_glob)
            + "\nFiles and entities:\n"
            + files_and_entities,
        )
        if self.expected_h5_entity is not None:
            self.unit_test.assertTrue(
                found_expected_entity,
                "Failed to find the expected entity/subfile (i.e. the data set "
                "to which the test data is compared to) '"
                + self.expected_h5_entity
                + "' using glob:\n"
                + os.path.join(run_directory, self.h5_glob)
                + "\nFiles and entities:\n"
                + files_and_entities,
            )


class H5CheckTestCase(unittest.TestCase):
    """The unit test object for performing all H5 checks for a given input file

    The parameters for the H5 output checks are determined by the
    command-line arguments. The first argument (sys.argv[1]) is the yaml
    input file. The second argument is the directory in which to find the
    run's.h5 files.

    The H5 checks and arguments are parsed from the "OutputFileChecks" in the
    input file metadata:

    ```yaml
    OutputFileChecks:
      - Label: "label"
        Subfile: "/h5_name.dat"
        FileGlob: "VolumeData*.h5"
        AbsoluteTolerance: 1e-12
      - Label: "another_label"
        Subfile: "/h5_group_name"
        FileGlob: "ReductionData*.h5"
        ExpectedDataSubfile: "/expected_h5_group_name"
        AbsoluteTolerance: 1e-11
        RelativeTolerance: 1e-6
        SkipColumns: [0, 1]
    ```
    """

    def test_h5_output(self):
        h5_check_list = []
        with open(self.input_filename, "r") as open_input_file:
            parsed_yaml = next(yaml.safe_load_all(open_input_file))
        to_snake_case = re.compile(r"(?!^)(?=[A-Z])")
        for check_block in parsed_yaml["OutputFileChecks"]:
            logging.info("Parsed File check : " + check_block.get("Label"))
            h5_check_list.append(
                H5Check(
                    self,
                    **{
                        re.sub(to_snake_case, "_", key).lower(): value
                        for key, value in check_block.items()
                    },
                )
            )
        for h5_check in h5_check_list:
            with self.subTest(test=h5_check.test_h5_label):
                h5_check.perform_check(self.run_directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-filename")
    parser.add_argument("--run-directory")
    logging.basicConfig(level=logging.INFO)
    duplicate_test_case, remaining_args = parser.parse_known_args(
        namespace=H5CheckTestCase
    )
    del duplicate_test_case
    # Use of full command-line arguments breaks the unit-test framework
    # (which needs to take its own command-line arguments), so we only pass
    # on the remaining args after we've retrieved the ones used by
    # `H5CheckTestCase`.
    unittest.main(argv=[parser.prog] + remaining_args, verbosity=2)
