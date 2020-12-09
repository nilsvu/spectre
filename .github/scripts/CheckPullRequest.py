#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import github
import re
import unittest


class TestPullRequest(unittest.TestCase):
    """Tests to run over pull requests before they can be merged."""

    @classmethod
    def setUpClass(cls):
        gh = github.Github(cls.github_token)
        gh_repo = gh.get_repo(cls.github_repository)
        cls.pr = gh_repo.get_pull(cls.pr_id)

    def test_review_checklist_is_complete(self):
        """Check that all items in the code-review checklist are ticked off."""
        CHECKLIST_PATTERN = r'### Code review checklist[\r\n]+(.*)[\r\n]+#'
        match_checklist = re.search(CHECKLIST_PATTERN,
                                    self.pr.body,
                                    flags=re.DOTALL)
        self.assertIsNotNone(
            match_checklist,
            ("Did not find the code review checklist in the PR description, "
             f"matching pattern: {CHECKLIST_PATTERN}"))
        checklist = match_checklist.group(1)
        CHECK_PATTERN = r'^- \[([ xX]?)\]'
        checks = re.findall(CHECK_PATTERN, checklist, flags=re.MULTILINE)
        EXPECTED_NUM_CHECKS = 3
        self.assertEqual(
            len(checks), EXPECTED_NUM_CHECKS,
            (f"Expected {EXPECTED_NUM_CHECKS} checks in the review checklist, "
             f"but found {len(checks)} matching pattern: {CHECK_PATTERN}"))

        def is_checked(check):
            return check in ['x', 'X']

        num_successful_checks = len(list(filter(is_checked, checks)))
        self.assertEqual(
            num_successful_checks, EXPECTED_NUM_CHECKS,
            ("Not all items in the review checklist are complete. Please "
             "make sure you have completed all items in the review checklist "
             "and tick them off."))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--github-repository',
        required=False,
        default='sxs-collaboration/spectre',
        help=("GitHub repository associated with the pull-request ID"))
    parser.add_argument(
        '--github-token',
        required=False,
        help=("Access token for GitHub queries. Refer to the GitHub "
              "documentation for instructions on creating a personal "
              "access token."))
    parser.add_argument('pr_id',
                        type=int,
                        help="ID of the pull-request to test")
    duplicate_test_case, remaining_args = parser.parse_known_args(
        namespace=TestPullRequest)
    del duplicate_test_case
    unittest.main(argv=[parser.prog] + remaining_args, verbosity=2)
