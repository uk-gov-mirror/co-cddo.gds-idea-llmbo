from pydantic import BaseModel

from conftest import ExampleOutput

from llmbo.adapters import NovaAdapter
from llmbo.models import ModelInput


def test_build_tool():
    """Test building a tool definition in Converse API toolSpec format."""
    tool = NovaAdapter.build_tool(ExampleOutput)

    assert "toolSpec" in tool
    assert tool["toolSpec"]["name"] == "ExampleOutput"
    assert "inputSchema" in tool["toolSpec"]
    assert "json" in tool["toolSpec"]["inputSchema"]
    params = tool["toolSpec"]["inputSchema"]["json"]
    assert "name" in params["properties"]
    assert "age" in params["properties"]


def test_build_tool_fallback_description():
    """Test build_tool provides a fallback when description is empty."""

    class NoDocModel(BaseModel):
        value: str

    NoDocModel.__doc__ = ""

    tool = NovaAdapter.build_tool(NoDocModel)
    assert "Extract structured data matching the NoDocModel schema." in (
        tool["toolSpec"]["description"]
    )


def test_build_tool_inlines_defs():
    """Test that build_tool inlines $defs for nested models."""

    class Inner(BaseModel):
        """An inner item."""

        value: str

    class Outer(BaseModel):
        """An outer container."""

        items: list[Inner]

    tool = NovaAdapter.build_tool(Outer)
    items_prop = (
        tool["toolSpec"]["inputSchema"]["json"]["properties"]["items"]
    )

    assert "$ref" not in items_prop.get("items", {})


def test_prepare_model_input():
    """Test preparing model input reshapes all fields for Converse API."""
    model_input = ModelInput(
        messages=[{"role": "user", "content": "Test"}],
        anthropic_version="bedrock-2023-05-31",
        system="Be helpful.",
        max_tokens=1024,
    )

    result = NovaAdapter.prepare_model_input(model_input)

    assert result.anthropic_version is None
    assert result.messages[0]["content"] == [{"text": "Test"}]
    assert result.system == [{"text": "Be helpful."}]
    assert result.inferenceConfig == {"maxTokens": 1024}
    assert result.max_tokens is None


def test_prepare_model_input_with_tool():
    """Test preparing model input with an output model adds toolConfig."""
    model_input = ModelInput(
        messages=[{"role": "user", "content": "Test"}],
    )

    result = NovaAdapter.prepare_model_input(model_input, ExampleOutput)

    assert result.toolConfig is not None
    assert len(result.toolConfig["tools"]) == 1
    assert (
        result.toolConfig["toolChoice"]["tool"]["name"] == "ExampleOutput"
    )
    assert result.tools is None
    assert result.tool_choice is None


def test_validate_result_valid():
    """Test validate_result with a valid Converse API response."""
    valid_result = {
        "output": {
            "message": {
                "content": [
                    {
                        "toolUse": {
                            "input": {"name": "John", "age": 30},
                            "name": "ExampleOutput",
                        }
                    }
                ],
            }
        },
    }

    result = NovaAdapter.validate_result(valid_result, ExampleOutput)
    assert isinstance(result, ExampleOutput)
    assert result.name == "John"
    assert result.age == 30


def test_validate_result_no_tool_use(caplog):
    """Test validate_result with no toolUse block."""
    invalid_result = {
        "output": {
            "message": {
                "content": [{"text": "Hallucinated text."}],
            }
        },
    }

    with caplog.at_level("DEBUG"):
        result = NovaAdapter.validate_result(invalid_result, ExampleOutput)

    assert result is None
    assert "no toolUse" in caplog.text


def test_validate_result_validation_error(caplog):
    """Test validate_result with schema validation failure."""
    invalid_result = {
        "output": {
            "message": {
                "content": [
                    {
                        "toolUse": {
                            "input": {"name": "John", "age": "thirty"},
                        }
                    }
                ],
            }
        },
    }

    with caplog.at_level("DEBUG"):
        result = NovaAdapter.validate_result(invalid_result, ExampleOutput)

    assert result is None
    assert "Validation failed:" in caplog.text


def test_validate_result_empty_output():
    """Test validate_result with empty result."""
    result = NovaAdapter.validate_result({}, ExampleOutput)
    assert result is None