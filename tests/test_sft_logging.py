import json
from pathlib import Path

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.tools import ToolDefinition

from pine.agent.llm import _tool_to_schema, messages_to_chat


def test_tool_to_schema_extraction() -> None:
    tool_def = ToolDefinition(
        name="get_outline",
        description="Retrieve the document section hierarchy.",
        parameters_json_schema={
            "type": "object",
            "properties": {"max_depth": {"type": "integer"}},
        },
    )
    schema = _tool_to_schema(tool_def)
    assert schema["name"] == "get_outline"
    assert schema["description"] == "Retrieve the document section hierarchy."
    assert schema["parameters"] == {
        "type": "object",
        "properties": {"max_depth": {"type": "integer"}},
    }


def test_messages_to_chat_single_leading_system_prompt() -> None:
    # Simulating a multi-turn conversation where Pydantic AI resends instructions on turn 2
    turn1_req = ModelRequest(
        instructions="System Instruction Version 1",
        parts=[UserPromptPart("What is on page 1?")],
    )
    turn1_resp = ModelResponse(
        parts=[
            ToolCallPart(
                tool_name="read_page",
                args={"page": 1},
                tool_call_id="call_abc123",
            )
        ]
    )
    turn2_req = ModelRequest(
        instructions="System Instruction Version 1",
        parts=[
            SystemPromptPart("System Instruction Version 1"),
            ToolReturnPart(
                tool_name="read_page",
                content="Page 1 text content...",
                tool_call_id="call_abc123",
            ),
        ],
    )
    turn2_resp = ModelResponse(parts=[TextPart("Page 1 describes NARL.")])

    messages = [turn1_req, turn1_resp, turn2_req, turn2_resp]
    chat = messages_to_chat(messages)

    # 1. First message must be the single system prompt
    assert chat[0]["role"] == "system"
    assert chat[0]["content"] == "System Instruction Version 1"

    # 2. Verify no secondary system prompt appears later in the conversation
    roles = [msg["role"] for msg in chat]
    assert roles.count("system") == 1
    assert roles == ["system", "user", "assistant", "tool", "assistant"]


def test_messages_to_chat_preserves_tool_call_ids() -> None:
    turn1_resp = ModelResponse(
        parts=[
            ToolCallPart(
                tool_name="search_pages",
                args={"query": "NARL"},
                tool_call_id="call_search_999",
            )
        ]
    )
    turn2_req = ModelRequest(
        instructions="You are an assistant.",
        parts=[
            ToolReturnPart(
                tool_name="search_pages",
                content='[{"page": 2}]',
                tool_call_id="call_search_999",
            )
        ],
    )

    chat = messages_to_chat([turn1_resp, turn2_req])

    # Assistant tool call retains id
    assistant_msg = chat[1]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["tool_calls"][0]["id"] == "call_search_999"

    # Tool return retains tool_call_id
    tool_msg = chat[2]
    assert tool_msg["role"] == "tool"
    assert tool_msg["name"] == "search_pages"
    assert tool_msg.get("tool_call_id") == "call_search_999"
