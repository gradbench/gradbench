import time
from dataclasses import dataclass
from typing import Any, Callable, TypedDict


class Timing(TypedDict):
    name: str
    nanoseconds: int


class OutputAndTimings(TypedDict):
    success: bool
    output: Any
    timings: list[Timing]


@dataclass
class Wrapped:
    function: Callable[[Any], Any]
    wrapped: Callable[[Any], OutputAndTimings]

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


def function(*, pre: Callable[[Any], Any], post: Callable[[Any], Any]):
    def decorator(function: Callable[[Any], Any]) -> Wrapped:
        def wrapped(input_raw: Any) -> OutputAndTimings:
            input = pre(input_raw)
            start = time.perf_counter_ns()
            output = function(input)
            end = time.perf_counter_ns()
            output_raw = post(output)
            timing = Timing(name="evaluate", nanoseconds=end - start)
            return OutputAndTimings(success=True, output=output_raw, timings=[timing])

        return Wrapped(function=function, wrapped=wrapped)

    return decorator


def multiple_runs(
    *,
    min_runs: Callable[[Any], int] = lambda x: x["min_runs"],
    min_seconds: Callable[[Any], float] = lambda x: x["min_seconds"],
    pre: Callable[[Any], Any],
    post: Callable[[Any], Any],
):
    min_runs_f = min_runs
    min_seconds_f = min_seconds

    def decorator(function: Callable[[Any], Any]) -> Wrapped:
        def wrapped(input_raw: Any) -> OutputAndTimings:
            timings: list[Timing] = []
            elapsed = 0
            input = pre(input_raw)
            min_runs = min_runs_f(input_raw)
            min_seconds = min_seconds_f(input_raw)
            while len(timings) < min_runs or elapsed < min_seconds:
                start = time.perf_counter_ns()
                output = function(input)
                end = time.perf_counter_ns()
                timings.append(Timing(name="evaluate", nanoseconds=end - start))
                elapsed += (end - start) / 1e9
            output_raw = post(output)
            return OutputAndTimings(success=True, output=output_raw, timings=timings)

        return Wrapped(function=function, wrapped=wrapped)

    return decorator
