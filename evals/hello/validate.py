#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from random import Random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", required=True)
    parser.add_argument("--golden", required=True)
    parser.add_argument("--tool", required=True)
    parser.add_argument("--results", type=str, default='results', metavar='DIR')
    args = parser.parse_args()

    double_golden_results = os.listdir(os.path.join(args.results, args.eval, args.golden, 'double'))
    square_golden_results = os.listdir(os.path.join(args.results, args.eval, args.golden, 'square'))
    double_tool_results = os.listdir(os.path.join(args.results, args.eval, args.tool, 'double'))
    square_tool_results = os.listdir(os.path.join(args.results, args.eval, args.tool, 'square'))

    bad = False

    for x in set(double_golden_results) - set(double_tool_results):
        bad = True
        print("Missing tool result: {x}")
    for x in set(double_tool_results) - set(double_golden_results):
        bad = True
        print("Missing golden result: {x}")

    for workload in double_golden_results:
        golden_result = json.load(open(os.path.join(args.results, args.eval, args.golden, 'double', workload, 'output'), 'r'))
        tool_result = json.load(open(os.path.join(args.results, args.eval, args.tool, 'double', workload, 'output'), 'r'))
        if golden_result != tool_result:
            bad = True
            print(f'Mismatch for double, workload={workload}')

    if bad:
        exit(1)
    else:
        exit(0)



if __name__ == "__main__":
    main()
