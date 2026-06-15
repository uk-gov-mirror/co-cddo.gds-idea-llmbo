from pydantic import BaseModel

from conftest import ExampleOutput

from llmbo.adapters import DeepSeekAdapter
from llmbo.models import ModelInput


def test_build_tool():
    """Test building a tool definition for DeepSeek."""
    tool = DeepSeekAdapter.build_tool(ExampleOutput)

    assert tool["type"] == "function"
    assert tool["function"]["name"] == "ExampleOutput"
    assert "parameters" in tool["function"]
    assert "properties" in tool["function"]["parameters"]


def test_build_tool_uses_docstring():
    """Test that build_tool uses the model's docstring as description."""
    tool = DeepSeekAdapter.build_tool(ExampleOutput)

    assert tool["function"]["description"] == ExampleOutput.__doc__


def test_build_tool_fallback_description():
    """Test build_tool falls back when model has no docstring."""

    class NoDocModel(BaseModel):
        value: str

    NoDocModel.__doc__ = None

    tool = DeepSeekAdapter.build_tool(NoDocModel)
    assert tool["function"]["description"] == "Please fill in the schema"


def test_prepare_model_input_preserves_fields():
    """Test that DeepSeek does not null anthropic_version or migrate system."""
    model_input = ModelInput(
        messages=[{"role": "user", "content": "Test"}],
        anthropic_version="bedrock-2023-05-31",
        system="You are helpful.",
    )

    result = DeepSeekAdapter.prepare_model_input(model_input)

    assert result.anthropic_version == "bedrock-2023-05-31"
    assert result.system == "You are helpful."


def test_prepare_model_input_with_tool():
    """Test preparing model input with an output model."""
    model_input = ModelInput(
        messages=[{"role": "user", "content": "Test"}],
    )

    result = DeepSeekAdapter.prepare_model_input(model_input, ExampleOutput)

    assert result.tools is not None
    assert len(result.tools) == 1
    assert result.tool_choice == "required"


def test_validate_result_valid():
    """Test validate_result with a valid native response."""
    valid_result = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "arguments": '{"name": "John", "age": 30}',
                                "name": "ExampleOutput",
                            },
                            "type": "function",
                        }
                    ],
                },
            }
        ],
    }

    result = DeepSeekAdapter.validate_result(valid_result, ExampleOutput)
    assert isinstance(result, ExampleOutput)
    assert result.name == "John"
    assert result.age == 30


def test_validate_result_empty():
    """Test validate_result with empty result."""
    result = DeepSeekAdapter.validate_result({}, ExampleOutput)
    assert result is None