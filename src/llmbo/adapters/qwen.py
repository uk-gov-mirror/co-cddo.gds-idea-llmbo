import logging

from .openai_compatible import OpenAICompatibleAdapter


class QwenAdapter(OpenAICompatibleAdapter):
    """Adapter for Alibaba Qwen models in AWS Bedrock."""

    logger = logging.getLogger(f"{__name__}.QwenAdapter")
    _provider_name = "Qwen"
