import sys
import unittest
from typing import Any, Callable, Literal

from pydantic import BaseModel, TypeAdapter


def assert_equal(a: Any, b: Any, msg: str = None) -> None:
    unittest.TestCase().assertEqual(a, b, msg)  # prints a diff if not equal


class DefineMessage(BaseModel):
    id: int
    kind: Literal["define"]
    module: str


class DefineResponse(BaseModel):
    id: int
    success: bool


class DefineExchange(BaseModel):
    message: DefineMessage
    nanoseconds: int
    response: DefineResponse


class EvaluateMessage(BaseModel):
    id: int
    kind: Literal["evaluate"]
    module: str
    name: str
    input: Any


class EvaluateResponse(BaseModel):
    id: int
    output: Any
    nanoseconds: dict[str, int]


class EvaluateExchange(BaseModel):
    message: EvaluateMessage
    nanoseconds: int
    response: EvaluateResponse


Exchange = DefineExchange | EvaluateExchange


def expect_define(exchange: Exchange) -> DefineExchange:
    assert_equal(exchange.message.kind, "define")
    return exchange


def expect_evaluate(exchange: Exchange) -> EvaluateExchange:
    assert_equal(exchange.message.kind, "evaluate")
    return exchange


Log = list[Exchange]


def validate_fixed(
    *,
    module: str,
    golden: str,
    log: str,
    checker: Callable[[str, Any, Any, Any], None],
) -> None:
    """
    Validate a `log` against a `golden` log for a "fixed" eval.

    In this case "fixed" means that the eval is expected to send the
    same sequence of messages every time. So, as part of validation,
    this function will check that the eval first sends a `"define"`
    message for the given `module` name, where the `golden` `"response"`
    says `"success": true`. If the `"response"` in the other `log` says
    `"success": false`, it will check that there are no other messages
    in that log, and then stop; otherwise, it will check that every
    following message is an `"evaluate"` message with that same `module`
    name, where the `"name"` and `"input"` are the same in both logs.
    Finally, it will check that the `"id"` of every `"response"` matches
    the `"id"` from the corresponding `"message"`, and that there was no
    `"id"` reuse across messages.

    Both the `log` and the `golden` log should be JSON strings.

    The `checker` argument is a function that takes four arguments (the
    `"name"` of the function being evaluated, the `"input"` to that
    function, the `"output"` from the `golden` log, and the `"output"`
    from the `log` being checked). If the two outputs don't match, the
    checker should `raise` an `Exception`.
    """

    log1 = TypeAdapter(Log).validate_json(golden)
    log2 = TypeAdapter(Log).validate_json(log)

    ids1: set[int] = set()
    ids2: set[int] = set()

    first1 = expect_define(log1.pop(0))
    first2 = expect_define(log2.pop(0))
    id1 = first1.message.id
    id2 = first2.message.id
    ids1.add(id1)
    ids2.add(id2)
    assert_equal(first1.response.id, id1)
    assert_equal(first2.response.id, id2)
    assert_equal(first1.message.module, module)
    assert_equal(first2.message.module, module)

    assert first1.response.success
    if not first2.response.success:
        assert not log2
        return

    for i, (exchange1, exchange2) in enumerate(zip(log1, log2, strict=True)):
        try:
            eval1 = expect_evaluate(exchange1)
            eval2 = expect_evaluate(exchange2)
            id1 = eval1.message.id
            id2 = eval2.message.id
            assert id1 not in ids1
            assert id2 not in ids2
            ids1.add(id1)
            ids2.add(id2)
            assert_equal(eval1.response.id, id1)
            assert_equal(eval2.response.id, id2)
            assert_equal(eval1.message.module, module)
            assert_equal(eval2.message.module, module)
            assert_equal(eval2.message.input, eval1.message.input)
            checker(
                eval1.message.name,
                eval1.message.input,
                eval1.response.output,
                eval2.response.output,
            )
        except Exception as e:
            print(f"Error in message at index {i + 1}", file=sys.stderr)
            if id1 and id2:
                print(f"Message ID from golden log: {id1}", file=sys.stderr)
                print(f"Message ID from other  log: {id2}", file=sys.stderr)
            raise e
