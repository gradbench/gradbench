import json
import sys
import traceback
from typing import Any, Callable, Optional

from pydantic import BaseModel


class StartResponse(BaseModel):
    id: int


class DefineResponse(BaseModel):
    id: int
    success: bool
    error: Optional[str] = None


class Timing(BaseModel):
    name: str
    nanoseconds: int


class EvaluateResponse(BaseModel):
    id: int
    output: Any
    timings: list[Timing]
    error: Optional[str] = None


class AnalysisResponse(BaseModel):
    id: int


class Analysis(BaseModel):
    valid: bool
    error: Optional[str]


def dump_analysis(analysis: Analysis) -> dict[str, Any]:
    return analysis.model_dump(exclude_none=True)


Validator = Callable[[str, Any, Any], Analysis]


def assertion(check: Callable[[str, Any, Any], None]) -> Validator:
    def validator(function: str, input: Any, output: Any) -> Analysis:
        try:
            check(function, input, output)
            return Analysis(valid=True, error=None)
        except Exception as e:
            error = "".join(traceback.format_exception(e))
            return Analysis(valid=False, error=error)

    return validator


def mismatch(check: Callable[[str, Any, Any], None], max_mismatches=10) -> Validator:
    def validator(function: str, input: Any, output: Any) -> Analysis:
        mismatches = check(function, input, output)
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
        l = sys.stdin.readline()
        if l == "":
            raise EOFError
        response = json.loads(l)
        if response["id"] != self.id:
            raise ValueError(f"expected message ID {self.id}, got {response['id']}")
        self.id += 1
        return response

    def start(self) -> StartResponse:
        message = {"kind": "start"}
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
        if response.error is None:
            analysis = self.validator(function, input, response.output)
            self.analysis(of=id, valid=analysis.valid, error=analysis.error)
        return response

    def analysis(self, *, of: int, valid: bool, error: Optional[str]) -> Any:
        request = {"kind": "analysis", "of": of, "valid": valid}
        if error is not None:
            request["error"] = error
        response = AnalysisResponse.model_validate(self.send(request))
        return response
