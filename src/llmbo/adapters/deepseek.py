import logging
import json
from typing import Any

from pydantic import BaseModel, ValidationError

from ..models import ModelInput
from .base import ModelProviderAdapter


class DeepSeekAdapter(ModelProviderAdapter):
    """Adapter for DeepSeek models in AWS Bedrock.
    
    This adapter handles:
    1. Building tool definitions in the standard OpenAI-compatible format used by DeepSeek.
    2. Enforcing tool use.
    3. Validating tool-use responses from both Native and Converse API formats.
    """

    logger = logging.getLogger(f"{__name__}.DeepSeekAdapter")

    @classmethod
    def build_tool(cls, output_model: type[BaseModel]) -> dict[str, Any]:
        """Build a tool definition in DeepSeek's native (OpenAI-compatible) format."""
        cls.logger.debug(f"Building tool definition for model: {output_model.__name__}")
        
        tool = {
            "type": "function",
            "function": {
                "name": output_model.__name__,
                "description": output_model.__doc__ or "Please fill in the schema",
                "parameters": output_model.model_json_schema(),
            }
        }
        cls.logger.debug(f"Created tool definition with name: {tool['function']['name']}")
        return tool

    @classmethod
    def prepare_model_input(cls, model_input: ModelInput, output_model: type[BaseModel] | None = None) -> ModelInput:
        """Prepare model input for DeepSeek models."""
        cls.logger.debug("Preparing model input for DeepSeek")

        # Build tool from output_model and add it to model_input
        if output_model:
            cls.logger.debug(f"Adding tool definition for {output_model.__name__}")
            tool = cls.build_tool(output_model)
            model_input.tools = [tool]
            
            # DeepSeek supports the "required" string to force tool execution
            model_input.tool_choice = "required" 

        return model_input

    @classmethod
    def validate_result(cls, result: dict[str, Any], output_model: type[BaseModel]) -> BaseModel | None:
        """Validate and parse output from DeepSeek models."""
        cls.logger.debug(f"Validating result against {output_model.__name__} schema")

        # SCENARIO A: Bedrock Converse API Format
        if "output" in result and "message" in result["output"]:
            content_blocks = result["output"]["message"].get("content", [])
            for block in content_blocks:
                if "toolUse" in block:
                    try:
                        instance = output_model(**block["toolUse"]["input"])
                        cls.logger.debug(f"Successfully validated Converse result as {output_model.__name__}")
                        return instance
                    except ValidationError as e:
                        cls.logger.debug(f"Converse API validation failed: {e!s}")
            return None

        # SCENARIO B: Native OpenAI-Compatible Format (Standard DeepSeek output)
        choices = result.get("choices", [])
        if not choices:
            cls.logger.debug("Result contains no choices (Not Native format)")
            return None

        message = choices[0].get("message", {})
        tool_calls = message.get("tool_calls", [])
        
        if not tool_calls:
            cls.logger.debug("Result contains no tool_calls")
            return None

        try:
            # DeepSeek/OpenAI format passes arguments as a stringified JSON, so we must json.loads() it first
            arguments_str = tool_calls[0].get("function", {}).get("arguments", "{}")
            arguments_dict = json.loads(arguments_str)
            
            instance = output_model(**arguments_dict)
            cls.logger.debug(f"Successfully validated Native result as {output_model.__name__}")
            return instance
        except (json.JSONDecodeError, ValidationError) as e:
            cls.logger.debug(f"Validation failed: {e!s}")
            return None