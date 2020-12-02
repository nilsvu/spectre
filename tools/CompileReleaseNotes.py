#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import collections
import git
import itertools
import logging
import os
import re
import textwrap
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


def get_last_release(repo: git.Repo, head_rev: str) -> git.TagReference:
    """Retrieve the release closest to the `head_rev` in the git history

    Returns:
      The tag representing the latest release, as measured by number of commits
      from the `head_rev`.
    """
    def is_version_tag(tag):
        return str(tag).startswith('v')

    def distance_from_head(tag):
        return len(list(repo.iter_commits(rev='{}..{}'.format(tag, head_rev))))

    return sorted(filter(is_version_tag, repo.tags), key=distance_from_head)[0]


@dataclass
class PullRequest:
    id: int
    title: str
    group: Optional[str] = None
    upgrade_instructions: Optional[str] = None


def get_merged_pull_requests(repo: git.Repo, from_rev: str,
                             to_rev: str) -> List[PullRequest]:
    """Parses list of merged PRs from merge commits.

    Parses merge commits in the repository between the revisions `from_rev` and
    `to_rev`. This is faster than querying GitHub and we can filter by commit
    SHAs instead of date.

    Returns:
      Merged pull-requests, ordered from most recently merged to least recently
      merged.
    """
    merge_commit_msg_pattern = '^Merge pull request #([0-9]+) from'
    merged_prs = []
    for commit in repo.iter_commits(rev='{}..{}'.format(from_rev, to_rev)):
        merge_commit_match = re.match(merge_commit_msg_pattern,
                                      commit.message,
                                      flags=re.MULTILINE)
        if not merge_commit_match:
            continue
        merged_prs.append(
            PullRequest(id=int(merge_commit_match.group(1)),
                        title=' '.join(commit.message.splitlines(False)[2:])))
    return merged_prs


def get_upgrade_instructions(pr_description: str):
    """Parse a section labeled "Upgrade instructions" from the PR description.

    This function looks for a section in the PR description that is enclosed in
    the HTML-comments `<!-- UPGRADE INSTRUCTIONS -->`. For example:

    ```md
    ## Upgrade instructions

    <!-- UPGRADE INSTRUCTIONS -->
    - Add the option `Evolution.InitialTime` to evolution input files. Set it
      to the value `0.` to keep the behavior the same as before.
    <!-- UPGRADE INSTRUCTIONS -->
    ```
    """
    FENCE_PATTERN = '<!-- UPGRADE INSTRUCTIONS -->'
    match = re.search(FENCE_PATTERN + '(.*)' + FENCE_PATTERN,
                      pr_description,
                      flags=re.DOTALL)
    if match is None:
        return None
    match = match.group(1).strip()
    if match.isspace():
        return None
    return match


if __name__ == "__main__":
    # The release notes always refer to the repository that contains this file
    repo = git.Repo(__file__, search_parent_directories=True)

    import argparse
    parser = argparse.ArgumentParser(
        description=("Compile release notes based on merged pull-requests. "
                     "Repository: {}. Branch: {}.").format(
                         repo.working_dir, repo.active_branch),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--output',
        '-o',
        required=False,
        help=("Name of the output file, e.g. 'release_notes.md'."))
    parser.add_argument(
        '--from',
        required=False,
        dest='from_rev',
        help=(
            "Commit ID (or tag) that marks the last release. Defaults to the "
            "most recent tag with format YYYY-MM-DD."))
    parser.add_argument('--to',
                        required=False,
                        dest='to_rev',
                        default='HEAD',
                        help=("Commit ID (or tag) that marks this release."))
    parser.add_argument(
        '--github-repository',
        required=False,
        default='sxs-collaboration/spectre',
        help=("GitHub repository associated with pull-request IDs in merge "
              "commits."))
    parser.add_argument(
        '--github-token',
        required=False,
        help=
        ("Access token for GitHub queries. Refer to the GitHub documentation "
         "for instructions on creating a personal access token."))
    parser.add_argument(
        '--no-github',
        action='store_true',
        help=("Disable GitHub queries, working only with the local "
              "repository."))
    parser.add_argument('-v',
                        '--verbose',
                        action='count',
                        default=0,
                        help="Verbosity (-v, -vv, ...)")
    parser.add_argument('--silent',
                        action='store_true',
                        help="Disable any logging")
    args = parser.parse_args()

    # Set the log level
    logging.basicConfig(
        level=logging.CRITICAL if args.silent else (logging.WARNING -
                                                    args.verbose * 10))

    # Retrieve last release
    if args.from_rev is None:
        args.from_rev = get_last_release(repo=repo, head_rev=args.to_rev)
        logging.info("Last release is: {}".format(args.from_rev))

    # Retrieve list of merged PRs for this release
    merged_prs = get_merged_pull_requests(repo=repo,
                                          from_rev=args.from_rev,
                                          to_rev=args.to_rev)
    logger.info("Merged PRs since last release:\n{}".format('\n'.join(
        map(str, merged_prs))))

    # Try to query GitHub for further information on the merged PRs
    pr_groups = ['major new feature', None, 'bugfix']
    if not args.no_github:
        import github
        import tqdm
        gh = github.Github(args.github_token)
        gh_repo = gh.get_repo(args.github_repository)
        for pr in tqdm.tqdm(merged_prs, desc="Downloading PR data", unit="PR"):
            # First, download data
            pr_gh = gh_repo.get_pull(pr.id)
            # Add group information to PR
            labels = [label.name for label in pr_gh.labels]
            for group in pr_groups:
                if group is None:
                    continue
                if group in labels:
                    pr.group = group
                    break
            # Add upgrade instructions to PR
            pr.upgrade_instructions = get_upgrade_instructions(pr_gh.body)

    # Sort PRs into their groups, ordered first by group then by the order they
    # were merged (most recently merged to least recently merged)
    grouped_prs = itertools.groupby(sorted(
        merged_prs,
        key=lambda pr: (pr_groups.index(pr.group), merged_prs.index(pr))),
                                    key=lambda pr: pr.group)

    # Write a nicely formatted release-notes file
    def format_list_of_prs(prs):
        return list(
            sum([
                textwrap.wrap('- {} (#{})'.format(pr.title, pr.id),
                              width=80,
                              subsequent_indent='  ') for pr in prs
            ], []))

    release_notes_content = []

    prs_with_upgrade_instructions = list(
        filter(lambda pr: pr.upgrade_instructions, merged_prs))
    if len(prs_with_upgrade_instructions) > 0:
        release_notes_content += ["## Upgrade instructions", ""]
        for pr in prs_with_upgrade_instructions:
            release_notes_content += [f"**From #{pr.id} ({pr.title}):**", ""]
            release_notes_content += textwrap.wrap(pr.upgrade_instructions,
                                                   width=80,
                                                   replace_whitespace=False)
        release_notes_content += [""]

    release_notes_content += ["## Merged pull-requests", ""]
    if len(merged_prs) > 0:
        for group, prs in grouped_prs:
            group_header = {
                'major new feature': "Major new features",
                'bugfix': "Bugfixes",
                None: "General changes",
            }[group]
            release_notes_content += (
                ["**{}:**".format(group_header), ""] +
                format_list_of_prs(reversed(list(prs)))) + [""]
    else:
        release_notes_content += ["_None_", ""]

    # Output
    if args.output:
        with open(args.output, 'w') as output_file:
            output_file.write('\n'.join(release_notes_content))
        logging.info("Release notes written to file: '{}'".format(args.output))
    else:
        logging.info("Compiled release notes:")
        print('\n'.join(release_notes_content))
