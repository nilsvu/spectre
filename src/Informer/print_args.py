#!/usr/bin/env python
import sys
import argparse

print(" ".join(sys.argv[1:]))
# we need to produce an output file so that the link step does not fail
p = argparse.ArgumentParser()
p.add_argument("compiler")
p.add_argument("-o", required=False)
p.add_argument("-c", required=False)
args, remaining_args = p.parse_known_args()
print(args, remaining_args)
