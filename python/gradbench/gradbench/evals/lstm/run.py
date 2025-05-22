import argparse
import math
from typing import Any

import numpy as np
from gradbench import cpp
from gradbench.adbench.lstm_data import LSTMInput
from gradbench.eval import (
    EvaluateResponse,
    SingleModuleValidatedEval,
    approve,
    mismatch,
)


def get_char_bits(text):
    return math.ceil(math.log2(max([ord(c) for c in text])))


def text_to_matrix(text, bits):
    return np.array(
        list(map(lambda c: list(map(int, bin(ord(c))[2:].zfill(bits))), text)),
        dtype=np.double,
    )


def gen_lstm(full_text, layer_count, char_count):
    # Get text extract
    use_text = full_text[:char_count]
    char_bits = get_char_bits(use_text)
    text_mat = text_to_matrix(use_text, char_bits)

    # Randomly generate past state, and parameters
    state = np.random.random((2 * layer_count, char_bits))
    main_params = np.random.random((2 * layer_count, char_bits * 4))
    extra_params = np.random.random((3, char_bits))

    return LSTMInput(main_params, extra_params, state, text_mat)


def read_full_text(filename, char_count):
    return open(filename, encoding="utf8").read(char_count)


def expect(function: str, input: Any) -> EvaluateResponse:
    return cpp.evaluate(
        tool="manual",
        module="lstm",
        function=function,
        input=input | {"min_runs": 1, "min_seconds": 0},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", nargs="+", type=int, default=[2, 4, 6])
    parser.add_argument("-c", nargs="+", type=int, default=[1024, 4096, 8192])
    parser.add_argument("--min-runs", type=int, default=1)
    parser.add_argument("--min-seconds", type=float, default=1)
    parser.add_argument("--no-validation", action="store_true", default=False)
    args = parser.parse_args()

    e = SingleModuleValidatedEval(
        module="lstm", validator=approve if args.no_validation else mismatch(expect)
    )
    e.start(config=vars(args))
    if e.define().success:
        text_file = "evals/lstm/data/lstm_full.txt"
        full_text = read_full_text(text_file, max(args.c))

        combinations = sorted(
            [(l, c) for l in args.l for c in args.c],
            key=lambda v: v[0] * v[1],
        )

        for l, c in combinations:
            input = gen_lstm(full_text, l, c).to_dict()
            e.evaluate(
                function="objective",
                input=input
                | {"min_runs": args.min_runs, "min_seconds": args.min_seconds},
                description=f"l={l},c={c}",
            )
            e.evaluate(
                function="jacobian",
                input=input
                | {"min_runs": args.min_runs, "min_seconds": args.min_seconds},
                description=f"l={l},c={c}",
            )


if __name__ == "__main__":
    try:
        main()
    except (EOFError, BrokenPipeError):
        pass
