# Distributed under the MIT License.
# See LICENSE.txt for details.

SPECTRE_BUILD_DIR = "@CMAKE_BINARY_DIR@"
SPECTRE_VERSION = "@SPECTRE_VERSION@"


def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog='spectre',
        description=(f"SpECTRE version: {SPECTRE_VERSION}\n"
                     f"Installed from: {SPECTRE_BUILD_DIR}\n"))
    parser.add_argument('--version', action='version', version=SPECTRE_VERSION)

    parser.parse_args()


if __name__ == '__main__':
    main()
