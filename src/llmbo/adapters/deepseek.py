import logging
from typing import Any

from pydantic import BaseModel

from ..models import ModelInput
from .openai_compatible import OpenAICompatibleAdapter


class DeepSeekAdapter(OpenAICompatibleAdapter):
    """Adapter for DeepSeek models in AWS Bedrock.

    DeepSeek follows the OpenAI function-calling convention but differs
    in two ways:

    - ``build_tool`` passes the raw Pydantic schema as parameters
      (no ``$defs`` inlining needed).
    - ``prepare_model_input`` does not null ``anthropic_version`` or
      migrate the system prompt into the messages array.
    """

    logger = logging.getLogger(f"{__name__}.DeepSeekAdapter")
    _provider_name = "DeepSeek"

    @classmethod
    def build_tool(cls, output_model: type[BaseModel]) -> dict[str, Any]:
        """Build a tool definition using the raw Pydantic schema.

        Args:
            output_model (type[BaseModel]): The Pydantic model to convert.

        Returns:
            dict[str, Any]: A tool definition dict.
        """
        cls.logger.debug(
            f"Building tool definition for model: {output_model.__name__}"
        )

        return {
            "type": "function",
            "function": {
                "name": output_model.__name__,
                "description": (
                    output_model.__doc__ or "Please fill in the schema"
                ),
                "parameters": output_model.model_json_schema(),
            },
        }

    @classmethod
    def prepare_model_input(
        cls,
        model_input: ModelInput,
        output_model: type[BaseModel] | None = None,
    ) -> ModelInput:
        """Prepare model input for DeepSeek models.

        Unlike other OpenAI-compatible providers, DeepSeek does not
        require the system prompt to be moved into the messages array.

        Args:
            model_input (ModelInput): The original model input.
            output_model (type[BaseModel] | None): Optional Pydantic
                model defining the expected output structure.

        Returns:
            ModelInput: Modified model input.
        """
        cls.logger.debug(f"Preparing model input for {cls._provider_name}")

        if output_model:
            cls.logger.debug(
                f"Adding tool definition for {output_model.__name__}"
            )
            model_input.tools = [cls.build_tool(output_model)]
            model_input.tool_choice = "required"

        return model_input
