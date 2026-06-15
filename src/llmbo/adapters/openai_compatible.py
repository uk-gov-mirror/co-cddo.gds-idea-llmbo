import json
import logging
from typing import Any

from pydantic import BaseModel, ValidationError

from ..models import ModelInput
from .base import ModelProviderAdapter


class OpenAICompatibleAdapter(ModelProviderAdapter):
    """Base adapter for models that follow the OpenAI function-calling
    convention in AWS Bedrock.

    This adapter handles:
    1. Building tool definitions in the standard OpenAI-compatible format.
    2. Migrating the system prompt into the messages array.
    3. Parsing responses from both the native OpenAI format and the
       Bedrock Converse API format.

    Subclasses need only set ``logger`` and ``_provider_name`` and can
    override individual methods where the provider diverges from the
    standard pattern.
    """

    logger = logging.getLogger(f"{__name__}.OpenAICompatibleAdapter")
    _provider_name: str = "OpenAI-compatible"

    @staticmethod
    def _inline_defs(
        schema: dict[str, Any], properties: dict[str, Any]
    ) -> dict[str, Any]:
        """Resolve ``$ref`` pointers by inlining definitions from ``$defs``.

        Pydantic can generate ``$defs`` for nested models. Some providers
        cannot follow ``$ref`` pointers, so we inline the referenced
        definitions directly into the property.

        Args:
            schema (dict[str, Any]): The full JSON schema from Pydantic.
            properties (dict[str, Any]): The top-level properties dict
                (mutated in place).

        Returns:
            dict[str, Any]: The same ``properties`` dict, with any
                ``$ref`` items replaced by the resolved definition.
        """
        defs = schema.get("$defs", {})
        if not defs:
            return properties

        for prop_val in properties.values():
            if prop_val.get("type") == "array" and "$ref" in prop_val.get(
                "items", {}
            ):
                ref_name = prop_val["items"]["$ref"].split("/")[-1]
                if ref_name in defs:
                    prop_val["items"] = defs[ref_name]

        return properties

    @classmethod
    def build_tool(cls, output_model: type[BaseModel]) -> dict[str, Any]:
        """Build a tool definition in the OpenAI function-calling format.

        Args:
            output_model (type[BaseModel]): The Pydantic model to convert.

        Returns:
            dict[str, Any]: A tool definition dict with inlined ``$defs``.
        """
        cls.logger.debug(
            f"Building tool definition for model: {output_model.__name__}"
        )

        schema = output_model.model_json_schema()
        properties = cls._inline_defs(schema, schema.get("properties", {}))

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
            },
        }

    @classmethod
    def prepare_model_input(
        cls,
        model_input: ModelInput,
        output_model: type[BaseModel] | None = None,
    ) -> ModelInput:
        """Prepare model input for an OpenAI-compatible provider.

        Args:
            model_input (ModelInput): The original model input.
            output_model (type[BaseModel] | None): Optional Pydantic
                model defining the expected output structure.

        Returns:
            ModelInput: Modified model input with provider-specific
                configurations applied.
        """
        cls.logger.debug(f"Preparing model input for {cls._provider_name}")

        model_input.anthropic_version = None

        # OpenAI-compatible models expect the system prompt as the
        # first message with role "system"
        if model_input.system and isinstance(model_input.system, str):
            if not model_input.messages:
                model_input.messages = []
            model_input.messages.insert(
                0, {"role": "system", "content": model_input.system}
            )
            model_input.system = None

        if output_model:
            cls.logger.debug(
                f"Adding tool definition for {output_model.__name__}"
            )
            model_input.tools = [cls.build_tool(output_model)]
            model_input.tool_choice = "required"

        return model_input

    @classmethod
    def validate_result(
        cls, result: dict[str, Any], output_model: type[BaseModel]
    ) -> BaseModel | None:
        """Validate and parse output from an OpenAI-compatible model.

        Supports both the Bedrock Converse API format and the native
        OpenAI function-calling format.

        Args:
            result (dict[str, Any]): Raw model output.
            output_model (type[BaseModel]): Pydantic model to validate
                against.

        Returns:
            BaseModel | None: Validated model instance, or None if
                validation fails.
        """
        cls.logger.debug(
            f"Validating result against {output_model.__name__} schema"
        )

        # Scenario A: Bedrock Converse API format
        if "output" in result and "message" in result["output"]:
            content_blocks = result["output"]["message"].get("content", [])
            for block in content_blocks:
                if "toolUse" in block:
                    try:
                        return output_model(**block["toolUse"]["input"])
                    except ValidationError as e:
                        cls.logger.debug(
                            f"Converse API validation failed: {e!s}"
                        )
            return None

        # Scenario B: Native OpenAI-compatible format
        choices = result.get("choices", [])
        if not choices:
            cls.logger.debug("No 'choices' array found.")
            return None

        message = choices[0].get("message", {})
        tool_calls = message.get("tool_calls", [])

        if not tool_calls:
            cls.logger.debug(
                "Result contains no tool_calls. The model hallucinated text."
            )
            return None

        try:
            arguments_str = (
                tool_calls[0].get("function", {}).get("arguments", "{}")
            )
            arguments_dict = json.loads(arguments_str)
            return output_model(**arguments_dict)
        except (json.JSONDecodeError, ValidationError) as e:
            cls.logger.debug(f"Validation failed: {e!s}")
            return None
