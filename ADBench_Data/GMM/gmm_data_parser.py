import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datafiles", nargs="*")
    args = parser.parse_args()
    

    for datafile in args.datafiles:
        lines = iter(Path(datafile).read_text().splitlines())
        d, k, n = [int(v) for v in next(lines).split()]
        alpha = []
        for _ in range(k):
            alpha.append(float(next(lines)))
        means = []
        for _ in range(k):
            means.append([float(v) for v in next(lines).split()])
        icf = []
        for _ in range(k):
            icf.append([float(v) for v in next(lines).split()])
        x = []
        for _ in range(n):
            x.append([float(v) for v in next(lines).split()])
        last = next(lines).split()
        gamma = float(last[0])
        m = int(last[1])
        print(
            json.dumps(
                {
                    "name": "calculate_jacobianGMM",
                    "input": {
                        "d": d,
                        "k": k,
                        "n": n,
                        "alpha": alpha,
                        "means": means,
                        "icf": icf,
                        "x": x,
                        "gamma": gamma,
                        "m": m,
                    },
                }
            )
        )


if __name__ == "__main__":
    main()