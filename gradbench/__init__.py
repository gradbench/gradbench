import json
from bisect import bisect_left
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


# https://stackoverflow.com/a/35857036
class LazySequence(Sequence):
    def __init__(self, f, n):
        """Construct a lazy sequence representing map(f, range(n))"""
        self.f = f
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if not (0 <= i < self.n):
            raise IndexError
        return self.f(i)


def smallest(p: Callable[[int], bool]) -> int:
    """
    Return the smallest nonnegative integer for which the monotonic
    predicate `p` holds true.

    >>> [smallest(lambda i: i >= n) for n in range(10)]
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    i = 1
    while not p(i):
        i *= 2
    return bisect_left(LazySequence(p, i), True)


def smallest_fresh(p: Path, f: Callable[[int], str]) -> Path:
    i = smallest(lambda i: not (p / f(i)).exists())
    return p / f(i)


def write(content: Any):
    folder = Path(datetime.today().strftime("data/%Y/%m/%d"))
    folder.mkdir(parents=True, exist_ok=True)
    file = smallest_fresh(folder, lambda i: f"{i}.json")
    file.write_text(json.dumps(content, indent=2))


def main():
    write({"x": 3e42})
