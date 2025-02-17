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
    *, runs=Callable[[Any], int], pre: Callable[[Any], Any], post: Callable[[Any], Any]
):
    def decorator(function: Callable[[Any], Any]) -> Wrapped:
        def wrapped(input_raw: Any) -> OutputAndTimings:
            timings: list[Timing] = []
            input = pre(input_raw)
            for _ in range(runs(input_raw)):
                start = time.perf_counter_ns()
                output = function(input)
                end = time.perf_counter_ns()
                timings.append(Timing(name="evaluate", nanoseconds=end - start))
            output_raw = post(output)
            return OutputAndTimings(success=True, output=output_raw, timings=timings)

        return Wrapped(function=function, wrapped=wrapped)

    return decorator
