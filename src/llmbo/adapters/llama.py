import json
import logging
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from ..models import ModelInput
from .base import ModelProviderAdapter


class LlamaAdapter(ModelProviderAdapter):
    """Adapter for Meta Llama models (Llama 3 / 4) in AWS Bedrock.
    
    This adapter handles:
    1. Formatting the prompt using Meta's specific header tokens.
    2. Enforcing JSON-only schema outputs.
    3. Translating 'max_tokens' to Llama's native 'max_gen_len'.
    """

    logger = logging.getLogger(f"{__name__}.LlamaAdapter")

    @staticmethod
    def format_llama_prompt(user_prompt: str, system_prompt: str | None = None, tools: str | None = None) -> str:
        """Formats the prompt using Meta's special tokens."""
        prompt_parts = ["<|begin_of_text|>"]
        
        if system_prompt or tools:
            prompt_parts.append("<|start_header_id|>system<|end_header_id|>\n\n")
            if system_prompt:
                prompt_parts.append(f"{system_prompt}\n\n")
            if tools:
                prompt_parts.append(
                    "You must respond ONLY with a valid JSON object. "
                    "Do not include any conversational text, markdown formatting, or preamble. "
                    f"The JSON object must follow this exact schema:\n{tools}\n"
                )
            prompt_parts.append("<|eot_id|>")
        
        prompt_parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|>")
        prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        
        return "".join(prompt_parts)

    @staticmethod
    def _schema_to_string(output_model: type[BaseModel]) -> str:
        """Serialise a Pydantic model's JSON schema for prompt injection."""
        schema = output_model.model_json_schema()
        return json.dumps(schema, indent=2)

    @classmethod
    def prepare_model_input(cls, model_input: ModelInput, output_model: type[BaseModel] | None = None) -> ModelInput:
        cls.logger.debug("Preparing model input for Meta Llama")

        original_prompt = model_input.messages[0].get("content", "") if model_input.messages else ""
        tool = cls._schema_to_string(output_model) if output_model else None
        
        # Build the native prompt string
        system = model_input.system if isinstance(model_input.system, str) else None
        formatted_prompt = cls.format_llama_prompt(original_prompt, system, tool)
        
        # Inject the native Llama keys dynamically
        requested_tokens = model_input.max_tokens or 2048
        safe_max_gen_len = min(requested_tokens, 8192) 
        
        model_input.prompt = formatted_prompt
        model_input.max_gen_len = safe_max_gen_len

        # Nullify the Anthropic/Converse keys so Bedrock doesn't reject them
        model_input.messages = None
        model_input.system = None
        model_input.max_tokens = None
        model_input.anthropic_version = None
        model_input.tools = None
        model_input.tool_choice = None
        
        return model_input

    @classmethod
    def validate_result(cls, result: dict[str, Any], output_model: type[BaseModel]) -> BaseModel | None:
        """Extracts JSON from the Llama 'generation' field."""
        cls.logger.debug(f"Validating result against {output_model.__name__} schema")

        # Bedrock's native Llama endpoint returns the text in a 'generation' key
        generation = result.get("generation", "")
        if not generation:
            cls.logger.debug("No 'generation' key found in result.")
            return None

        # Hunt for JSON brackets
        match = re.search(r"\{.*\}", generation, re.DOTALL)
        if not match:
            cls.logger.debug("Did not find anything that looked like JSON in the response")
            return None

        try:
            arguments = match.group(0)
            parsed_arguments = json.loads(arguments)
            return output_model(**parsed_arguments)
        except (json.JSONDecodeError, ValidationError) as e:
            cls.logger.debug(f"Validation failed: {e!s}")
            return None