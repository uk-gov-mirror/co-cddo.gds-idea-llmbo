import logging

from .openai_compatible import OpenAICompatibleAdapter


class OpenAIAdapter(OpenAICompatibleAdapter):
    """Adapter for OpenAI models in AWS Bedrock."""

    logger = logging.getLogger(f"{__name__}.OpenAIAdapter")
    _provider_name = "OpenAI"
