import json
import logging
from typing import Any

from pydantic import BaseModel, ValidationError

from ..models import ModelInput
from .base import ModelProviderAdapter


class MistralFunctionAdapter(ModelProviderAdapter):
    """Adapter for Mistral models in AWS Bedrock."""

    logger = logging.getLogger(f"{__name__}.MistralFunctionAdapter")

    @classmethod
    def build_tool(cls, output_model: type[BaseModel]) -> dict[str, Any]:
        cls.logger.debug(f"Building tool definition for model: {output_model.__name__}")

        schema = output_model.model_json_schema()
        parameters = {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", []),
        }
        
        # Keep nested definitions if Pydantic generated them
        if "$defs" in schema:
            parameters["$defs"] = schema["$defs"]

        return {
            "type": "function",
            "function": {
                "name": output_model.__name__,
                "description": schema.get("description", ""),
                "parameters": parameters,
            },
        }

    @classmethod
    def prepare_model_input(cls, model_input: ModelInput, output_model: type[BaseModel] | None = None) -> ModelInput:
        cls.logger.debug("Preparing model input for Mistral")

        model_input.anthropic_version = None

        # --- THE SYSTEM PROMPT FIX ---
        # Mistral strictly expects the system prompt inside the messages array
        if getattr(model_input, "system", None):
            if not model_input.messages:
                model_input.messages = []
            
            # Prepend the system prompt to the messages list
            model_input.messages.insert(0, {"role": "system", "content": model_input.system})
            
            # Delete the top-level key so AWS doesn't throw the 'extra_forbidden' error
            model_input.system = None
        # -----------------------------

        # Enforce Mistral's token ceiling
        if getattr(model_input, "max_tokens", None) and model_input.max_tokens > 8192:
            model_input.max_tokens = 8192

        if output_model:
            cls.logger.debug(f"Adding tool definition for {output_model.__name__}")
            model_input.tools = [cls.build_tool(output_model)]
            model_input.tool_choice = "any"
            
        return model_input

    @classmethod
    def validate_result(cls, result: dict[str, Any], output_model: type[BaseModel]) -> BaseModel | None:
        cls.logger.debug(f"Validating result against {output_model.__name__} schema")

        choices = result.get("choices", [])
        if not choices:
            return None

        tools = choices[0].get("message", {}).get("tool_calls", [])
        if not tools:
            return None

        try:
            arguments = tools[0].get("function", {}).get("arguments", {})
            parsed_arguments = json.loads(arguments)
            return output_model(**parsed_arguments)
        except (json.JSONDecodeError, ValidationError) as e:
            cls.logger.debug(f"Validation failed: {e!s}")
            return None