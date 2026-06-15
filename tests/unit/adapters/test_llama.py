import json

from conftest import ExampleOutput

from llmbo.adapters import LlamaAdapter
from llmbo.models import ModelInput


def test_format_llama_prompt_basic():
    """Test basic prompt formatting without system or tools."""
    result = LlamaAdapter.format_llama_prompt("Hello")

    assert "<|begin_of_text|>" in result
    assert "<|start_header_id|>user<|end_header_id|>" in result
    assert "Hello" in result
    assert "<|start_header_id|>assistant<|end_header_id|>" in result
    # No system header when neither system nor tools are provided
    assert "system<|end_header_id|>" not in result


def test_format_llama_prompt_with_system():
    """Test prompt formatting includes system block."""
    result = LlamaAdapter.format_llama_prompt(
        "Hello", system_prompt="Be helpful."
    )

    assert "<|start_header_id|>system<|end_header_id|>" in result
    assert "Be helpful." in result


def test_format_llama_prompt_with_tools():
    """Test prompt formatting injects schema constraint."""
    result = LlamaAdapter.format_llama_prompt(
        "Hello", tools='{"type": "object"}'
    )

    assert "You must respond ONLY with a valid JSON object." in result
    assert '{"type": "object"}' in result


def test_schema_to_string():
    """Test that _schema_to_string returns a valid JSON string."""
    result = LlamaAdapter._schema_to_string(ExampleOutput)
    parsed = json.loads(result)

    assert "properties" in parsed
    assert "name" in parsed["properties"]
    assert "age" in parsed["properties"]


def test_prepare_model_input():
    """Test preparing model input populates Llama fields and nulls others."""
    model_input = ModelInput(
        messages=[{"role": "user", "content": "Test"}],
        anthropic_version="bedrock-2023-05-31",
        system="Be helpful.",
        max_tokens=1024,
    )

    result = LlamaAdapter.prepare_model_input(model_input)

    assert result.prompt is not None
    assert "<|begin_of_text|>" in result.prompt
    assert "Be helpful." in result.prompt
    assert result.max_gen_len == 1024

    assert result.messages is None
    assert result.system is None
    assert result.max_tokens is None
    assert result.anthropic_version is None
    assert result.tools is None
    assert result.tool_choice is None


def test_prepare_model_input_caps_tokens():
    """Test that max_gen_len is capped at 8192."""
    model_input = ModelInput(
        messages=[{"role": "user", "content": "Test"}],
        max_tokens=20000,
    )

    result = LlamaAdapter.prepare_model_input(model_input)
    assert result.max_gen_len == 8192


def test_prepare_model_input_default_tokens():
    """Test that max_gen_len defaults to 2048 when max_tokens is None."""
    model_input = ModelInput(
        messages=[{"role": "user", "content": "Test"}],
        max_tokens=None,
    )

    result = LlamaAdapter.prepare_model_input(model_input)
    assert result.max_gen_len == 2048


def test_prepare_model_input_with_tool():
    """Test that output model schema is injected into the prompt."""
    model_input = ModelInput(
        messages=[{"role": "user", "content": "Test"}],
    )

    result = LlamaAdapter.prepare_model_input(model_input, ExampleOutput)

    assert "You must respond ONLY with a valid JSON object." in result.prompt


def test_validate_result_valid():
    """Test validate_result with a valid generation containing JSON."""
    valid_result = {
        "generation": 'Here is the result: {"name": "John", "age": 30}',
    }

    result = LlamaAdapter.validate_result(valid_result, ExampleOutput)
    assert isinstance(result, ExampleOutput)
    assert result.name == "John"
    assert result.age == 30


def test_validate_result_no_generation(caplog):
    """Test validate_result with missing generation key."""
    with caplog.at_level("DEBUG"):
        result = LlamaAdapter.validate_result({}, ExampleOutput)

    assert result is None
    assert "No 'generation' key found in result." in caplog.text


def test_validate_result_no_json(caplog):
    """Test validate_result with no JSON in generation."""
    invalid_result = {"generation": "Just text, no JSON here."}

    with caplog.at_level("DEBUG"):
        result = LlamaAdapter.validate_result(invalid_result, ExampleOutput)

    assert result is None
    assert "Did not find anything that looked like JSON" in caplog.text


def test_validate_result_invalid_json(caplog):
    """Test validate_result with malformed JSON."""
    invalid_result = {"generation": '{"name": "broken json}'}

    with caplog.at_level("DEBUG"):
        result = LlamaAdapter.validate_result(invalid_result, ExampleOutput)

    assert result is None
    assert "Validation failed:" in caplog.text


def test_validate_result_invalid_schema(caplog):
    """Test validate_result with schema validation failure."""
    invalid_result = {
        "generation": '{"name": "John", "age": "thirty"}',
    }

    with caplog.at_level("DEBUG"):
        result = LlamaAdapter.validate_result(invalid_result, ExampleOutput)

    assert result is None
    assert "Validation failed:" in caplog.text
