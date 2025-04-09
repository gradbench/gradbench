import json
import sys
import traceback
from typing import Any, Callable, Optional

from pydantic import BaseModel

from gradbench.comparison import compare_json_objects


class Timing(BaseModel):
    name: str
    nanoseconds: int


class StartResponse(BaseModel):
    id: int
    tool: Optional[str] = None
    config: Optional[Any] = None


class DefineResponse(BaseModel):
    id: int
    success: bool
    timings: Optional[list[Timing]] = None
    error: Optional[str] = None


class EvaluateResponse(BaseModel):
    id: int
    success: bool
    output: Optional[Any] = None
    timings: Optional[list[Timing]] = None
    error: Optional[str] = None


class AnalysisResponse(BaseModel):
    id: int


class Analysis(BaseModel):
    valid: bool
    error: Optional[str]


Validator = Callable[[str, Any, Any], Analysis]


def approve(function: str, input: Any, output: Any) -> Analysis:
    return Analysis(valid=True, error=None)


def assertion(check: Callable[[str, Any, Any], None]) -> Validator:
    def validator(function: str, input: Any, output: Any) -> Analysis:
        try:
            check(function, input, output)
            return Analysis(valid=True, error=None)
        except Exception as e:
            error = "".join(traceback.format_exception(e))
            return Analysis(valid=False, error=error)

    return validator


def mismatch(
    expect: Callable[[str, Any], EvaluateResponse], max_mismatches=10
) -> Validator:
    def validator(function: str, input: Any, output: Any) -> Analysis:
        expected = expect(function, input)
        if not expected["success"]:
            return Analysis(
                valid=False,
                error=f"golden implementation failed with error:\n{expected['error']}",
            )
        mismatches = compare_json_objects(expected["output"], output)
        if len(mismatches) == 0:
            return Analysis(valid=True, error=None)
        else:
            shown_mismatches = mismatches[0:max_mismatches]
            mismatches_str = "\n".join(shown_mismatches)
            error = f"Found {len(mismatches)} mismatches, showing {len(shown_mismatches)}:\n{mismatches_str}"
            return Analysis(valid=False, error=error)

    return validator


class SingleModuleValidatedEval:
    module: str
    validator: Validator
    id: int
    validations: dict[int, Analysis]

    def __init__(self, *, module: str, validator: Validator):
        self.module = module
        self.validator = validator
        self.id = 0
        self.validations = {}

    def send(self, message: Any) -> Any:
        json.dump({"id": self.id} | message, sys.stdout)
        print(flush=True)
        if message["kind"] == "end":
            return
        l = sys.stdin.readline()  # noqa: E741
        if l == "":
            raise EOFError
        response = json.loads(l)
        if response["id"] != self.id:
            raise ValueError(f"expected message ID {self.id}, got {response['id']}")
        self.id += 1
        return response

    def start(self, *, config: Optional[Any] = None) -> StartResponse:
        message = {"kind": "start", "eval": self.module}
        if config is not None:
            message["config"] = config
        response = StartResponse.model_validate(self.send(message))
        return response

    def define(self) -> DefineResponse:
        message = {"kind": "define", "module": self.module}
        response = DefineResponse.model_validate(self.send(message))
        return response

    def evaluate(
        self, *, function: str, input: Any, description: Optional[str] = None
    ) -> EvaluateResponse:
        message = {
            "kind": "evaluate",
            "module": self.module,
            "function": function,
            "input": input,
        }
        if description is not None:
            message["description"] = description
        id = self.id
        response = EvaluateResponse.model_validate(self.send(message))
        output = response.output
        if output is not None:
            analysis = self.validator(function, input, output)
            self.analysis(of=id, valid=analysis.valid, error=analysis.error)
        return response

    def analysis(self, *, of: int, valid: bool, error: Optional[str]) -> Any:
        request = {"kind": "analysis", "of": of, "valid": valid}
        if error is not None:
            request["error"] = error
        response = AnalysisResponse.model_validate(self.send(request))
        return response
