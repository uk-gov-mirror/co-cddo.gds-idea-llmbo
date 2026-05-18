import logging
from typing import Any

from pydantic import BaseModel, ValidationError

from ..models import ModelInput
from .base import ModelProviderAdapter


class NovaAdapter(ModelProviderAdapter):
    """Adapter for Amazon Nova models natively using the Bedrock Converse API."""

    logger = logging.getLogger(f"{__name__}.NovaAdapter")

    @classmethod
    def build_tool(cls, output_model: type[BaseModel]) -> dict[str, Any]:
        cls.logger.debug(f"Building tool definition for model: {output_model.__name__}")

        schema = output_model.model_json_schema()
        properties = schema.get("properties", {})
        
        # Inline the $defs to prevent the "lazy pointer" issue we saw with OSS models
        if "$defs" in schema:
            for prop_name, prop_val in properties.items():
                if prop_val.get("type") == "array" and "$ref" in prop_val.get("items", {}):
                    ref_name = prop_val["items"]["$ref"].split("/")[-1]
                    if ref_name in schema["$defs"]:
                        prop_val["items"] = schema["$defs"][ref_name]

        parameters = {
            "type": "object",
            "properties": properties,
            "required": schema.get("required", []),
        }

        # Converse API strictly rejects zero-length strings for descriptions.
        description = schema.get("description", "").strip()
        if not description:
            description = f"Extract structured data matching the {output_model.__name__} schema."

        # Converse API wraps tools inside a "toolSpec" and "inputSchema -> json"
        return {
            "toolSpec": {
                "name": output_model.__name__,
                "description": description,
                "inputSchema": {
                    "json": parameters
                }
            }
        }

    @classmethod
    def prepare_model_input(cls, model_input: ModelInput, output_model: type[BaseModel] | None = None) -> ModelInput:
        cls.logger.debug("Preparing model input for Amazon Nova")

        # Clean up legacy artifacts
        model_input.anthropic_version = None

        # 1. Reshape the User Messages into Converse JSONArrays
        if model_input.messages:
            for msg in model_input.messages:
                if isinstance(msg.get("content"), str):
                    msg["content"] = [{"text": msg["content"]}]

        # 2. Reshape the System Prompt into a Converse JSONArray
        if getattr(model_input, "system", None) and isinstance(model_input.system, str):
            model_input.system = [{"text": model_input.system}]

        # 3. Reshape max_tokens into inferenceConfig
        if getattr(model_input, "max_tokens", None):
            setattr(model_input, "inferenceConfig", {"maxTokens": model_input.max_tokens})
            model_input.max_tokens = None

        # 4. Reshape tools into toolConfig
        if output_model:
            cls.logger.debug(f"Adding tool definition for {output_model.__name__}")
            tool_config = {
                "tools": [cls.build_tool(output_model)],
                "toolChoice": {
                    "tool": {"name": output_model.__name__}
                }
            }
            # Dynamically attach the toolConfig and delete the legacy keys
            setattr(model_input, "toolConfig", tool_config)
            model_input.tools = None
            model_input.tool_choice = None

        return model_input

    @classmethod
    def validate_result(cls, result: dict[str, Any], output_model: type[BaseModel]) -> BaseModel | None:
        cls.logger.debug(f"Validating result against {output_model.__name__} schema")

        # Converse API returns data in: output -> message -> content -> toolUse
        output = result.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])

        for block in content_blocks:
            if "toolUse" in block:
                try:
                    return output_model(**block["toolUse"]["input"])
                except ValidationError as e:
                    cls.logger.debug(f"Validation failed: {e!s}")
                    return None

        cls.logger.debug("Result contains no toolUse. The model hallucinated text.")
        return None