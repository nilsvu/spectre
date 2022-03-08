#!/bin/sh -e

# Distributed under the MIT License.
# See LICENSE.txt for details.

@Python_EXECUTABLE@ -m spectre.tools.CleanOutput --force \
                    --output-dir @CMAKE_BINARY_DIR@/$3 $2
mkdir @CMAKE_BINARY_DIR@/$3
cd @CMAKE_BINARY_DIR@/$3
@SPECTRE_TEST_RUNNER@ @CMAKE_BINARY_DIR@/bin/$1 --input-file $2 ${4} &&
    @Python_EXECUTABLE@ @CMAKE_SOURCE_DIR@/tools/CheckOutputFiles.py \
     --input-file $2 --run-directory @CMAKE_BINARY_DIR@/$3 &&
@Python_EXECUTABLE@ -m spectre.tools.CleanOutput \
                    --output-dir @CMAKE_BINARY_DIR@/$3 $2
