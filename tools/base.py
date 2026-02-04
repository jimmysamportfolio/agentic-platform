from __future__ import annotations
from pydantic.json_schema import model_json_schema
from pathlib import Path
from typing import Any
from abc import ABC, abstractmethod
from pydantic import BaseModel, ValidationError
from enum import Enum
from dataclasses import dataclass, field

class ToolKind(str, Enum):
    READ = "read"
    WRITE = "write"
    SHELL = "shell"
    NETWORK = "network"
    MEMORY = "memory"
    MCP = "mcp"

@dataclass
class ToolResult:
    success: bool
    output: str
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    truncated: bool = False

    @classmethod
    def error_result(
        cls,
        error: str,
        output: str = "",
        **kwargs: Any,
    ):
        return cls(
            success=False,
            output=output,
            error=error,
            **kwargs
        )

    @classmethod
    def success_result(cls, output: str, **kwargs: Any):
        return cls(
            success=True,
            output=output,
            error = None,
            **kwargs,
        )

    def to_model_output(self) -> str:
        if self.success:
            return self.output
        
        return f"Error: {self.error}\n\nOutput: {self.output}"


@dataclass
class ToolConfirmation:
    tool_name: str
    params: dict[str, Any]
    description: str

@dataclass
class ToolInvocation:
    cwd: Path
    params: dict[str, Any]

class Tool(ABC):
    name: str = "base_tool"
    description: str = "Base tool"
    kind: ToolKind = ToolKind.READ

    def __init__(self) -> None:
        pass

    @property
    def schema(self) -> dict[str, Any] | type["BaseModel"]:
        raise NotImplementedError("Tool must define schema property or class attribute")

    @abstractmethod
    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        pass

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        schema = self.schema
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            try: 
                schema(**params)
            except ValidationError as e:
                errors = []
                for error in e.errors():
                    field = ".".join(str(x) for x in error.get("loc", []))
                    msg = error.get("msg", "Validation error")
                    errors.append(f"Parameter '{field}': {msg}")
                
                return errors
            
            except Exception as e:
                return [str(e)]

        return []

    def is_mutating(self, params: dict[str, Any]) -> bool:
        return self.kind in {
            ToolKind.WRITE,
            ToolKind.SHELL,
            ToolKind.NETWORK,
            ToolKind.MEMORY,
        }

    async def get_confirmation(self, invocation: ToolInvocation) -> ToolInvocation | None:
        if not self.is_mutating(invocation.params):
            return None
        
        return ToolConfirmation(
            tool_name=self.name,
            params=invocation.params,
            description=f"Execute {self.name}",
        )

    def to_openai_schema(self) -> dict[str, Any]:
        schema = self.schema
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            json_schema = model_json_schema(schema, mode="serialization")

            return {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": 'object',
                    "properties": json_schema.get("properties", {}),
                    "required": json_schema.get("requried", [])
                }
            }
        
        if isinstance(schema, dict):
            result = {
                "name": self.name,
                "description": self.description
            }

            if "paramters" in schema:
                result["parameters"] = schema["parameters"]
            else:
                result["parameters"] = schema
            
            return result

        raise ValueError(f'Invalid schema type for tool {self.name}" {type(schema)}')