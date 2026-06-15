from conftest import ExampleOutput
from pydantic import BaseModel

from llmbo.adapters.openai_compatible import OpenAICompatibleAdapter
from llmbo.models import ModelInput


class NestedItem(BaseModel):
    """A nested item."""

    value: str


class NestedOutput(BaseModel):
    """Output with nested items."""

    items: list[NestedItem]


def test_inline_defs_resolves_refs():
    """Test that _inline_defs replaces $ref pointers with definitions."""
    schema = {
        "$defs": {
            "NestedItem": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            }
        },
        "properties": {
            "items": {
                "type": "array",
                "items": {"$ref": "#/$defs/NestedItem"},
            }
        },
    }
    properties = schema["properties"]
    result = OpenAICompatibleAdapter._inline_defs(schema, properties)

    assert "$ref" not in result["items"]["items"]
    assert result["items"]["items"]["type"] == "object"
    assert "value" in result["items"]["items"]["properties"]


def test_inline_defs_no_defs():
    """Test that _inline_defs is a no-op when no $defs are present."""
    properties = {"name": {"type": "string"}}
    schema = {"properties": properties}
    result = OpenAICompatibleAdapter._inline_defs(schema, properties)

    assert result == properties


def test_build_tool():
    """Test building a tool definition in OpenAI format."""
    tool = OpenAICompatibleAdapter.build_tool(ExampleOutput)

    assert tool["type"] == "function"
    assert tool["function"]["name"] == "ExampleOutput"
    assert "parameters" in tool["function"]
    assert "name" in tool["function"]["parameters"]["properties"]
    assert "age" in tool["function"]["parameters"]["properties"]


def test_build_tool_inlines_defs():
    """Test that build_tool inlines $defs for nested models."""
    tool = OpenAICompatibleAdapter.build_tool(NestedOutput)
    items_schema = tool["function"]["parameters"]["properties"]["items"]

    assert "$ref" not in items_schema.get("items", {})


def test_prepare_model_input():
    """Test system prompt migration and anthropic_version removal."""
    model_input = ModelInput(
        messages=[{"role": "user", "content": "Test"}],
        anthropic_version="bedrock-2023-05-31",
        system="You are helpful.",
    )

    result = OpenAICompatibleAdapter.prepare_model_input(model_input)

    assert result.anthropic_version is None
    assert result.system is None
    assert result.messages[0] == {
        "role": "system",
        "content": "You are helpful.",
    }
    assert result.messages[1] == {"role": "user", "content": "Test"}


def test_prepare_model_input_no_system():
    """Test that messages are untouched when no system prompt is set."""
    model_input = ModelInput(
        messages=[{"role": "user", "content": "Test"}],
    )

    result = OpenAICompatibleAdapter.prepare_model_input(model_input)

    assert len(result.messages) == 1
    assert result.messages[0] == {"role": "user", "content": "Test"}


def test_prepare_model_input_with_tool():
    """Test that tool definition and tool_choice are added."""
    model_input = ModelInput(
        messages=[{"role": "user", "content": "Test"}],
    )

    result = OpenAICompatibleAdapter.prepare_model_input(model_input, ExampleOutput)

    assert result.tools is not None
    assert len(result.tools) == 1
    assert result.tools[0]["type"] == "function"
    assert result.tool_choice == "required"


def test_validate_result_valid_native():
    """Test validate_result with a valid native OpenAI format response."""
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

    result = OpenAICompatibleAdapter.validate_result(valid_result, ExampleOutput)
    assert isinstance(result, ExampleOutput)
    assert result.name == "John"
    assert result.age == 30


def test_validate_result_valid_converse():
    """Test validate_result with a valid Converse API format response."""
    valid_result = {
        "output": {
            "message": {
                "content": [
                    {
                        "toolUse": {
                            "input": {"name": "Jane", "age": 25},
                            "name": "ExampleOutput",
                        }
                    }
                ],
            }
        },
    }

    result = OpenAICompatibleAdapter.validate_result(valid_result, ExampleOutput)
    assert isinstance(result, ExampleOutput)
    assert result.name == "Jane"
    assert result.age == 25


def test_validate_result_no_choices(caplog):
    """Test validate_result with no choices in native format."""
    with caplog.at_level("DEBUG"):
        result = OpenAICompatibleAdapter.validate_result({"choices": []}, ExampleOutput)

    assert result is None
    assert "No 'choices' array found." in caplog.text


def test_validate_result_no_tool_calls(caplog):
    """Test validate_result with no tool_calls in message."""
    invalid_result = {
        "choices": [{"message": {"content": "No tools here."}}],
    }
    with caplog.at_level("DEBUG"):
        result = OpenAICompatibleAdapter.validate_result(invalid_result, ExampleOutput)

    assert result is None
    assert "no tool_calls" in caplog.text


def test_validate_result_invalid_json(caplog):
    """Test validate_result with unparseable JSON arguments."""
    invalid_result = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "arguments": '{"broken json}',
                                "name": "ExampleOutput",
                            },
                        }
                    ],
                },
            }
        ],
    }
    with caplog.at_level("DEBUG"):
        result = OpenAICompatibleAdapter.validate_result(invalid_result, ExampleOutput)

    assert result is None
    assert "Validation failed:" in caplog.text


def test_validate_result_invalid_schema(caplog):
    """Test validate_result with schema validation failure."""
    invalid_result = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "arguments": '{"name": "John", "age": "thirty"}',
                                "name": "ExampleOutput",
                            },
                        }
                    ],
                },
            }
        ],
    }
    with caplog.at_level("DEBUG"):
        result = OpenAICompatibleAdapter.validate_result(invalid_result, ExampleOutput)

    assert result is None
    assert "Validation failed:" in caplog.text


def test_validate_result_converse_validation_error(caplog):
    """Test validate_result with Converse format but invalid data."""
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
        result = OpenAICompatibleAdapter.validate_result(invalid_result, ExampleOutput)

    assert result is None
    assert "Converse API validation failed:" in caplog.text


def test_validate_result_converse_no_tool_use():
    """Test validate_result with Converse format but no toolUse block."""
    invalid_result = {
        "output": {
            "message": {
                "content": [{"text": "Hallucinated text."}],
            }
        },
    }

    result = OpenAICompatibleAdapter.validate_result(invalid_result, ExampleOutput)
    assert result is None
