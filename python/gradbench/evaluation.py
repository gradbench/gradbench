import json
import sys
from typing import Any, Callable

from pydantic import BaseModel


class DefineResponse(BaseModel):
    id: int
    success: bool


class EvaluateResponse(BaseModel):
    id: int
    output: Any
    nanoseconds: dict[str, int]


class Validation(BaseModel):
    correct: bool


Validator = Callable[[str, Any, Any], Validation]


def correctness(check: Callable[[str, Any, Any], bool]) -> Validator:
    def validator(name: str, input: Any, output: Any) -> Validation:
        return Validation(correct=check(name, input, output))

    return validator


class SingleModuleValidatedEvaluation:
    module: str
    validator: Validator
    id: int
    validations: dict[int, Validation]

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
        response = json.loads(sys.stdin.readline())
        if response["id"] != self.id:
            raise ValueError(f"expected message ID {self.id}, got {response['id']}")
        self.id += 1
        return response

    def define(self, *, source: str) -> DefineResponse:
        message = {"kind": "define", "module": self.module, "source": source}
        response = DefineResponse.model_validate(self.send(message))
        return response

    def evaluate(self, *, name: str, input: Any) -> EvaluateResponse:
        message = {
            "kind": "evaluate",
            "module": self.module,
            "name": name,
            "input": input,
        }
        id = self.id
        response = EvaluateResponse.model_validate(self.send(message))
        self.validations[id] = self.validator(name, input, response.output)
        return response

    def end(self) -> None:
        validations = [
            {"id": id} | validation.model_dump()
            for id, validation in self.validations.items()
        ]
        message = {"kind": "end", "validations": validations}
        self.send(message)
