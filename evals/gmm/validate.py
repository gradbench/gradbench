#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from random import Random
import numpy as np

# Generic checking function.
def check_results(eval_name, eql, results, golden, tool, name):
    bad = False

    objective_golden_results = os.listdir(os.path.join(results, eval_name, golden, name))
    objective_tool_results = os.listdir(os.path.join(results, eval_name, tool, name))

    for x in set(objective_golden_results) - set(objective_tool_results):
        bad = True
        print("Missing tool result: {x}")
    for x in set(objective_tool_results) - set(objective_golden_results):
        bad = True
        print("Missing golden result: {x}")

    for workload in objective_golden_results:
        golden_result = json.load(open(os.path.join(results, eval_name, golden, name, workload, 'output'), 'r'))
        tool_result = json.load(open(os.path.join(results, eval_name, tool, name, workload, 'output'), 'r'))
        if not eql(golden_result, tool_result):
            bad = True
            print(f'Mismatch for {name}, workload={workload}')

    return bad

def eql_objective(a,b):
    try:
        return np.all(np.isclose(a, b))
    except:
        return False

def eql_jacobian(a,b):
    try:
        return np.all(np.isclose(a, b))
    except:
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", required=True)
    parser.add_argument("--tool", required=True)
    parser.add_argument("--results", type=str, default='results', metavar='DIR')
    args = parser.parse_args()

    bad = False

#    bad = check_results('gmm', eql_objective, args.results, args.golden, args.tool, 'calculate_objectiveGMM') or bad
    bad = check_results('gmm', eql_jacobian,args.results, args.golden, args.tool, 'calculate_jacobianGMM') or bad

    if bad:
        exit(1)
    else:
        exit(0)

if __name__ == "__main__":
    main()
