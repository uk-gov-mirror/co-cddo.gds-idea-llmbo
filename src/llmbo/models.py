import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class Manifest(BaseModel):
    """Job manifest details.

    Uses ``extra="allow"`` so that new fields returned by the AWS Bedrock API
    (e.g. ``inputAudioSecond``) are captured in ``model_extra`` instead of
    raising ``TypeError``.
    """

    model_config = ConfigDict(extra="allow")

    totalRecordCount: int
    processedRecordCount: int
    successRecordCount: int
    errorRecordCount: int
    inputTokenCount: int | None = None
    outputTokenCount: int | None = None


@dataclass
class ToolChoice:
    """Toolchoice details."""

    type: Literal["any", "tool", "auto"]
    name: str | None = None


@dataclass
class ModelInput:
    """Configuration class for AWS Bedrock model inputs.

    This class defines the structure and parameters for model invocation requests
    following AWS Bedrock's expected format. Provider-specific adapters may
    reshape or nullify fields as needed for their API.

    See https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html

    Attributes:
        messages (list[dict] | None): List of message objects with role
            and content. Defaults to None.
        anthropic_version (str | None): Version string for Anthropic
            models. Defaults to "bedrock-2023-05-31".
        max_tokens (int | None): Maximum number of tokens in the
            response. Defaults to 2000.
        system (str | list[dict[str, Any]] | None): System message for
            the model. A string for most providers; reshaped to a list
            of content blocks for Converse API models (e.g. Nova).
        stop_sequences (list[str] | None): Custom stop sequences.
        temperature (float | None): Sampling temperature.
        top_p (float | None): Nucleus sampling parameter.
        top_k (int | None): Top-k sampling parameter.
        tools (list[dict] | None): Tool definitions for structured
            outputs.
        tool_choice (ToolChoice | str | None): Tool selection
            configuration.
        prompt (str | None): Native text prompt for models that use a
            single string instead of a messages array (e.g. Llama).
        max_gen_len (int | None): Maximum generation length for models
            that use this parameter instead of max_tokens (e.g. Llama).
        inferenceConfig (dict[str, Any] | None): Inference
            configuration for Converse API models (e.g. Nova).
        toolConfig (dict[str, Any] | None): Tool configuration for
            Converse API models (e.g. Nova).
    """


    # These are required
    messages: list[dict] | None = None
    anthropic_version: str | None = "bedrock-2023-05-31"
    max_tokens: int | None = 2000

    system: str | list[dict[str, Any]] | None = None
    stop_sequences: list[str] | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None

    tools: list[dict] | None = None
    tool_choice: ToolChoice | str | None = None

    # Provider-specific fields (used by adapters that diverge from the
    # Anthropic / Messages API shape, e.g. Llama, Nova)
    prompt: str | None = None
    max_gen_len: int | None = None
    inferenceConfig: dict[str, Any] | None = None
    toolConfig: dict[str, Any] | None = None


    def to_dict(self):
        """Convert to dict."""
        result = {k: v for k, v in self.__dict__.items() if v is not None}
        if isinstance(self.tool_choice, ToolChoice):
            result["tool_choice"] = self.tool_choice.__dict__
        return result

    def to_json(self):
        """Convert to json string."""
        return json.dumps(self.to_dict())
