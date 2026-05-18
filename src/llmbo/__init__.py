from .adapters import (
    AnthropicAdapter, 
    MistralFunctionAdapter, 
    DeepSeekAdapter, 
    LlamaAdapter,
    OpenAIAdapter,
    QwenAdapter,
    NovaAdapter,
)
from .batch_inferer import BatchInferer
from .models import (
    Manifest,
    ModelInput,
    ToolChoice,
)
from .registry import ModelAdapterRegistry
from .structured_batch_inferer import StructuredBatchInferer

try:
    from ._version import __version__
except ImportError:  # pragma: no cover
    __version__ = "0.0.0.dev0"

# Register the model adapters
ModelAdapterRegistry.register(r"(anthropic|claude)", AnthropicAdapter)
ModelAdapterRegistry.register(r"(mistral|mixtral)", MistralFunctionAdapter)
ModelAdapterRegistry.register(r"deepseek", DeepSeekAdapter)
ModelAdapterRegistry.register(r"(meta\.llama)", LlamaAdapter)
ModelAdapterRegistry.register(r"openai", OpenAIAdapter)
ModelAdapterRegistry.register(r"qwen", QwenAdapter)
ModelAdapterRegistry.register(r"(amazon\.nova)", NovaAdapter)


__all__ = [
    "BatchInferer",
    "Manifest",
    "ModelAdapterRegistry",
    "ModelInput",
    "StructuredBatchInferer",
    "ToolChoice",
    "__version__",
]
