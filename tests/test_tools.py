from __future__ import annotations

import pytest

from insight_agent.core.llm import ToolCall
from insight_agent.core.tools import ToolError, ToolRegistry, build_parameters, tool


@tool
def greet(name: str, excited: bool = False) -> str:
    """Greet someone by name."""

    return f"hi {name}{'!' if excited else ''}"


@tool
async def slow_echo(text: str) -> str:
    """Echo text back."""

    return text


def test_parameters_track_annotations_and_defaults() -> None:
    params = build_parameters(greet.func)
    assert params["properties"]["name"] == {"type": "string"}
    assert params["properties"]["excited"] == {"type": "boolean"}
    assert params["required"] == ["name"]


def test_description_comes_from_docstring() -> None:
    assert greet.description == "Greet someone by name."


def test_schema_shape() -> None:
    assert greet.schema() == {
        "name": "greet",
        "description": "Greet someone by name.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "excited": {"type": "boolean"}},
            "required": ["name"],
        },
    }


async def test_invoke_sync_tool() -> None:
    result = await greet.invoke(ToolCall(id="1", name="greet", arguments={"name": "ada"}))
    # Strings are passed through verbatim; only structured results are encoded.
    assert result.content == "hi ada"
    assert result.is_error is False


async def test_invoke_async_tool() -> None:
    result = await slow_echo.invoke(ToolCall(id="2", name="slow_echo", arguments={"text": "x"}))
    assert result.content == "x"


async def test_tool_error_is_reported_to_the_model() -> None:
    @tool
    def picky(flag: bool) -> str:
        """Require a flag."""

        if not flag:
            raise ToolError("flag must be true")
        return "ok"

    result = await picky.invoke(ToolCall(id="3", name="picky", arguments={"flag": False}))
    assert result.is_error is True
    assert result.content == "flag must be true"


async def test_unexpected_exception_is_not_swallowed() -> None:
    @tool
    def boom() -> str:
        """Always fail."""

        raise RuntimeError("kaboom")

    result = await boom.invoke(ToolCall(id="4", name="boom", arguments={}))
    assert result.is_error is True
    assert "kaboom" in result.content


async def test_unknown_tool_returns_error_result() -> None:
    registry = ToolRegistry([greet])
    result = await registry.run(ToolCall(id="5", name="nope", arguments={}))
    assert result.is_error is True
    assert "Unknown tool" in result.content


async def test_run_all_preserves_order() -> None:
    registry = ToolRegistry([greet])
    results = await registry.run_all(
        [
            ToolCall(id="a", name="greet", arguments={"name": "x"}),
            ToolCall(id="b", name="greet", arguments={"name": "y"}),
        ]
    )
    assert [r.call_id for r in results] == ["a", "b"]


def test_registry_membership() -> None:
    registry = ToolRegistry([greet, slow_echo])
    assert "greet" in registry
    assert "missing" not in registry
    assert len(registry) == 2
    assert {s["name"] for s in registry.schemas()} == {"greet", "slow_echo"}


def test_custom_name_and_description() -> None:
    @tool(name="aliased", description="Custom text.")
    def original() -> None:
        """Ignored."""

    assert original.name == "aliased"
    assert original.description == "Custom text."


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        (str, "string"),
        (int, "integer"),
        (float, "number"),
        (bool, "boolean"),
        (list[str], "array"),
        (dict[str, int], "object"),
    ],
)
def test_json_type_mapping(annotation: object, expected: str) -> None:
    def f(x: object) -> None:  # noqa: ANN401
        pass

    f.__annotations__["x"] = annotation
    assert build_parameters(f)["properties"]["x"] == {"type": expected}
