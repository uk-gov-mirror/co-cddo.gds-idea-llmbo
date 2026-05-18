import json
import logging
from typing import Any

from pydantic import BaseModel, ValidationError

from ..models import ModelInput
from .base import ModelProviderAdapter


class OpenAIAdapter(ModelProviderAdapter):
    """Adapter for OpenAI models in AWS Bedrock.
    
    Handles:
    1. Standard OpenAI function calling format.
    2. Migrating system prompts into the messages array.
    3. Parsing native OpenAI tool_calls responses.
    """

    logger = logging.getLogger(f"{__name__}.OpenAIAdapter")

    @classmethod
    def build_tool(cls, output_model: type[BaseModel]) -> dict[str, Any]:
        cls.logger.debug(f"Building tool definition for model: {output_model.__name__}")

        schema = output_model.model_json_schema()
        properties = schema.get("properties", {})
        
        # --- Inline the nested schemas ---
        # If Pydantic generated a $defs block, we unpack it directly into the properties.
        # This prevents "lazy" models from getting confused by $ref pointers.
        if "$defs" in schema:
            for prop_name, prop_val in properties.items():
                if prop_val.get("type") == "array" and "$ref" in prop_val.get("items", {}):
                    # Extract 'BaseLineTask' from '#/$defs/BaseLineTask'
                    ref_name = prop_val["items"]["$ref"].split("/")[-1]
                    if ref_name in schema["$defs"]:
                        # Replace the pointer with the actual dictionary shape!
                        prop_val["items"] = schema["$defs"][ref_name]
        # ------------------------------------------

        parameters = {
            "type": "object",
            "properties": properties,
            "required": schema.get("required", []),
        }

        return {
            "type": "function",
            "function": {
                "name": output_model.__name__,
                "description": schema.get("description", ""),
                "parameters": parameters,
            }
        }

    @classmethod
    def prepare_model_input(cls, model_input: ModelInput, output_model: type[BaseModel] | None = None) -> ModelInput:
        cls.logger.debug("Preparing model input for OpenAI OSS")

        # Clean up Anthropic-specific keys
        model_input.anthropic_version = None

        # OpenAI strictly expects the system prompt as the first message with role "system"
        if getattr(model_input, "system", None):
            if not model_input.messages:
                model_input.messages = []
            model_input.messages.insert(0, {"role": "system", "content": model_input.system})
            model_input.system = None  # Delete the top-level key to prevent validation errors

        if output_model:
            cls.logger.debug(f"Adding tool definition for {output_model.__name__}")
            model_input.tools = [cls.build_tool(output_model)]
            
            # OpenAI supports 'required' to force the model to use the provided function
            model_input.tool_choice = "required" 

        return model_input

    @classmethod
    def validate_result(cls, result: dict[str, Any], output_model: type[BaseModel]) -> BaseModel | None:
        cls.logger.debug(f"Validating result against {output_model.__name__} schema")

        # SCENARIO A: Bedrock Converse API Format
        if "output" in result and "message" in result["output"]:
            content_blocks = result["output"]["message"].get("content", [])
            for block in content_blocks:
                if "toolUse" in block:
                    try:
                        return output_model(**block["toolUse"]["input"])
                    except ValidationError as e:
                        cls.logger.debug(f"Converse API validation failed: {e!s}")
            return None

        # SCENARIO B: Native OpenAI Format
        choices = result.get("choices", [])
        if not choices:
            cls.logger.debug("No 'choices' array found.")
            return None

        message = choices[0].get("message", {})
        tool_calls = message.get("tool_calls", [])
        
        if not tool_calls:
            cls.logger.debug("Result contains no tool_calls. The model hallucinated text.")
            return None

        try:
            # OpenAI passes arguments as a stringified JSON, so we must parse it first
            arguments_str = tool_calls[0].get("function", {}).get("arguments", "{}")
            arguments_dict = json.loads(arguments_str)
            return output_model(**arguments_dict)
        except (json.JSONDecodeError, ValidationError) as e:
            cls.logger.debug(f"Validation failed: {e!s}")
            return None