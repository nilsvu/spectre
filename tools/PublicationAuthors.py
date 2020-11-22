# Distributed under the MIT License.
# See LICENSE.txt for details.

import sys
import textwrap
import dataclasses
from typing import Set


@dataclasses.dataclass
class Feature:
    description: str
    authors: Set[str]


FEATURES = [
    Feature(description="Vacuum evolutions, e.g. black-hole binaries",
            authors=["Mr. Vacuum"]),
    Feature(description="Matter evolutions, e.g. involving neutron stars",
            authors=["Mr. Matter"]),
    Feature(description="CCE, e.g. extracting waveforms", authors=["Mr. CCE"]),
    Feature(description="Elliptic solves, e.g. initial data, elasticity",
            authors=["Mr. Elliptic"])
]


def confirm_or_deny(feature) -> bool:
    valid_answers = {
        "yes": True,
        "y": True,
        "ye": True,
        "no": False,
        "n": False
    }
    sys.stdout.write("".join(
        textwrap.wrap("- " + feature.description + " ... [y/n] ",
                      width=80,
                      subsequent_indent='  ',
                      drop_whitespace=False)))
    choice = input().lower()
    if choice in valid_answers:
        return valid_answers[choice]
    else:
        print("Please respond with 'yes'/'y' or 'no'/'n'.\n")
        confirm_or_deny(question)


if __name__ == '__main__':
    print("\nDo you use any of these SpECTRE features in your project?\n")
    author_list = set()
    for feature in FEATURES:
        if confirm_or_deny(feature):
            author_list.update(feature.authors)
    print("\nPlease invite the following people to join "
          "your project:\n\n{}".format('\n'.join(sorted(author_list))))
