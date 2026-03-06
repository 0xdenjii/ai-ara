"""Tool agent — structured tool calls for desktop automation and system commands.

Handles user requests that require executing actions on the computer:
opening apps, file operations, system commands, media control, etc.

Architecture:
    - LLM generates structured tool calls (JSON format)
    - Permission system gates which tools can be executed
    - Confirmation required for dangerous operations (configurable)
    - Results fed back to LLM for natural language response

Flow:
    1. Parse user request to identify desired action
    2. LLM generates structured ToolCall(s)
    3. Check permissions for each tool call
    4. Execute tool calls via ToolRegistry
    5. Feed results back to LLM for response generation

Usage:
    from ara.tools.registry import ToolRegistry
    registry = ToolRegistry(settings)
    agent = ToolAgent(settings, llm_client, tool_registry=registry)
    await agent.connect()
    result_state = await agent.execute(state)
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from ara.agents.base import AgentCapability, AgentPermission, BaseAgent
from ara.agents.state import tool_call_to_dict, tool_result_to_dict
from ara.utils.errors import ToolPermissionError
from ara.utils.logging import get_logger
from ara.utils.types import (
    ChatMessage,
    IntentType,
    MessageRole,
    ToolCall,
    ToolResult,
)

if TYPE_CHECKING:
    from ara.agents.state import AgentState
    from ara.config.settings import AraSettings
    from ara.core.llm import LLMClient
    from ara.tools.registry import ToolRegistry

log = get_logger("tool_agent")


# ── Capability Declaration ───────────────────────────────────────────────────

_TOOL_CAPABILITY = AgentCapability(
    name="tool_agent",
    display_name="Desktop Automation",
    description=(
        "Executes actions on the computer: opening applications, file operations, "
        "system commands, media control, and other desktop automation tasks."
    ),
    supported_intents=frozenset(
        {
            IntentType.DESKTOP,
        }
    ),
    required_permissions=frozenset(
        {
            AgentPermission.LLM_GENERATE,
            AgentPermission.DESKTOP_CONTROL,
            AgentPermission.SYSTEM_EXECUTE,
            AgentPermission.FILE_READ,
            AgentPermission.FILE_WRITE,
        }
    ),
    priority=100,
)

# Base prompt template — tool list is injected dynamically from the registry.
_TOOL_GENERATION_PROMPT_TEMPLATE = """You are ARA's tool execution agent. The user wants you to perform an action on their Windows computer.

{tool_list}

Rules:
1. Return a JSON array of tool calls: [{{"tool_name": "...", "parameters": {{...}}}}]
2. Use the MINIMUM number of tool calls needed
3. If the request is dangerous (deleting files, system changes), include a confirmation step
4. If you can't determine the right tool, return: [{{"tool_name": "clarify", "parameters": {{"message": "..."}}}}]
5. ONLY return the JSON array, no other text"""

# Fallback tool list when no registry is provided (backwards compatibility).
_FALLBACK_TOOL_LIST = """Available tools:
- open_application: Opens an application. Parameters: {"name": "app_name"}
- close_application: Closes an application. Parameters: {"name": "app_name"}
- type_text: Types text at cursor position. Parameters: {"text": "content"}
- press_key: Presses a keyboard shortcut. Parameters: {"keys": "ctrl+c"}
- click: Clicks at screen position. Parameters: {"x": 100, "y": 200, "button": "left"}
- screenshot: Takes a screenshot. Parameters: {}
- run_command: Runs a PowerShell command. Parameters: {"command": "Get-Process"}
- open_url: Opens a URL in browser. Parameters: {"url": "https://..."}
- file_read: Reads a file. Parameters: {"path": "C:/Users/.../file.txt"}
- file_write: Writes to a file. Parameters: {"path": "C:/Users/.../file.txt", "content": "..."}
- volume_control: Controls system volume. Parameters: {"action": "mute|unmute|up|down", "level": 50}
- media_control: Controls media playback. Parameters: {"action": "play|pause|next|previous|stop"}"""


class ToolAgent(BaseAgent):
    """Tool agent — generates and executes structured tool calls.

    Uses a ToolRegistry for:
        - Dynamic LLM prompt generation (tool list from registry schemas)
        - Actual tool execution (registry.execute handles permissions + audit)
        - Confirmation gate (based on ToolRiskLevel from schemas)

    Falls back to a static tool list and stub results if no registry is provided.

    Args:
        settings: Application settings.
        llm_client: LLM for generating tool calls and responses.
        tool_registry: Optional ToolRegistry for execution and prompt generation.
        tool_executor: Optional callable (legacy) — prefer tool_registry instead.
    """

    def __init__(
        self,
        settings: AraSettings,
        llm_client: LLMClient,
        *,
        tool_registry: ToolRegistry | None = None,
        tool_executor: Any | None = None,
    ) -> None:
        super().__init__(settings)
        self._llm = llm_client
        self._registry = tool_registry
        # Use registry.execute if available, otherwise fall back to legacy executor
        self._executor = tool_registry.execute if tool_registry else tool_executor

    @property
    def capability(self) -> AgentCapability:
        return _TOOL_CAPABILITY

    async def _execute(self, state: AgentState) -> AgentState:
        """Execute tool agent: parse → generate calls → execute → respond.

        Args:
            state: Current orchestrator state.

        Returns:
            Updated state with tool calls, results, and response.
        """
        query = state.get("query", "")
        if not query.strip():
            return {"response": "", "error": "Empty query"}  # type: ignore[return-value]

        # Step 1: Generate tool calls via LLM
        tool_calls = await self._generate_tool_calls(query)

        if not tool_calls:
            return {  # type: ignore[return-value]
                "response": "I couldn't determine what action to take. Could you be more specific?",
                "tool_calls": [],
                "tool_results": [],
            }

        # Step 2: Check for clarification requests
        if len(tool_calls) == 1 and tool_calls[0].tool_name == "clarify":
            message = tool_calls[0].parameters.get(
                "message", "Could you clarify what you'd like me to do?"
            )
            return {  # type: ignore[return-value]
                "response": message,
                "tool_calls": [tool_call_to_dict(tc) for tc in tool_calls],
                "tool_results": [],
            }

        # Step 3: Check confirmation requirements (use registry risk levels if available)
        needs_confirmation = self._check_needs_confirmation(tool_calls)

        if needs_confirmation and self._settings.desktop.require_confirmation:
            # Format confirmation message
            actions = [f"- {tc.tool_name}: {json.dumps(tc.parameters)}" for tc in tool_calls]
            confirm_msg = (
                "I need to perform these actions:\n" + "\n".join(actions) + "\n\nShall I go ahead?"
            )
            return {  # type: ignore[return-value]
                "response": confirm_msg,
                "tool_calls": [tool_call_to_dict(tc) for tc in tool_calls],
                "tool_results": [],
                "metadata": {"awaiting_confirmation": True},
            }

        # Step 4: Execute tool calls (if executor available)
        tool_results: list[ToolResult] = []
        if self._executor is not None:
            for tc in tool_calls:
                try:
                    result = await self._executor(tc)
                    tool_results.append(result)
                except ToolPermissionError:
                    raise
                except Exception as exc:
                    tool_results.append(
                        ToolResult(
                            call_id=tc.call_id,
                            tool_name=tc.tool_name,
                            success=False,
                            output="",
                            error=str(exc),
                        )
                    )
        else:
            # No executor yet (Phase 6) — return calls for orchestrator
            tool_results.extend(
                ToolResult(
                    call_id=tc.call_id,
                    tool_name=tc.tool_name,
                    success=True,
                    output=f"Tool '{tc.tool_name}' queued for execution",
                )
                for tc in tool_calls
            )

        # Step 5: Generate natural language response from results
        response = await self._generate_response(query, tool_calls, tool_results)

        return {  # type: ignore[return-value]
            "response": response,
            "tool_calls": [tool_call_to_dict(tc) for tc in tool_calls],
            "tool_results": [tool_result_to_dict(tr) for tr in tool_results],
        }

    def _check_needs_confirmation(self, tool_calls: list[ToolCall]) -> bool:
        """Check if any tool call requires user confirmation.

        Uses registry tool schemas (risk level) if available, otherwise
        falls back to a static set of known-dangerous tool names.

        Args:
            tool_calls: List of tool calls to check.

        Returns:
            True if at least one call needs confirmation.
        """
        # Legacy fallback names when no registry is available
        _legacy_confirm = frozenset({"run_command", "file_write", "close_application"})

        for tc in tool_calls:
            if self._registry is not None:
                tool = self._registry.get_tool(tc.tool_name)
                if tool is not None:
                    from ara.tools.registry import ToolRiskLevel

                    if tool.schema.risk_level in {ToolRiskLevel.MODERATE, ToolRiskLevel.DANGEROUS}:
                        return True
            elif tc.tool_name in _legacy_confirm:
                return True
        return False

    async def _generate_tool_calls(self, query: str) -> list[ToolCall]:
        """Use LLM to generate structured tool calls from a natural language request.

        Generates the tool prompt dynamically from the registry if available,
        otherwise uses the fallback static tool list.

        Args:
            query: The user's request.

        Returns:
            List of ToolCall objects.
        """
        if self._registry is not None and self._registry.tool_count > 0:
            tool_list = self._registry.generate_tool_prompt()
        else:
            tool_list = _FALLBACK_TOOL_LIST

        prompt = _TOOL_GENERATION_PROMPT_TEMPLATE.format(tool_list=tool_list)

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=prompt),
            ChatMessage(role=MessageRole.USER, content=query),
        ]

        raw = await self._llm.generate(messages, temperature=0.1, max_tokens=500)
        return self._parse_tool_calls(raw)

    def _parse_tool_calls(self, raw_response: str) -> list[ToolCall]:
        """Parse LLM response into ToolCall objects.

        Args:
            raw_response: Raw LLM output (expected JSON array).

        Returns:
            List of ToolCall objects.
        """
        cleaned = raw_response.strip()
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
        cleaned = cleaned.strip()

        # Try to find JSON array
        array_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if array_match:
            cleaned = array_match.group(0)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            log.warning("tool_call_parse_failed", raw=raw_response[:200])
            return []

        if not isinstance(parsed, list):
            parsed = [parsed]

        calls: list[ToolCall] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            tool_name = item.get("tool_name", "")
            parameters = item.get("parameters", {})
            if tool_name:
                calls.append(ToolCall(tool_name=str(tool_name), parameters=dict(parameters)))

        return calls

    async def _generate_response(
        self,
        query: str,
        tool_calls: list[ToolCall],
        tool_results: list[ToolResult],
    ) -> str:
        """Generate a natural language response summarizing tool execution.

        Args:
            query: Original user request.
            tool_calls: Tools that were called.
            tool_results: Results of tool execution.

        Returns:
            Human-readable response.
        """
        results_summary = []
        for tc, tr in zip(tool_calls, tool_results, strict=False):
            status = "succeeded" if tr.success else f"failed: {tr.error}"
            results_summary.append(f"- {tc.tool_name}: {status}")
            if tr.output and tr.success:
                results_summary.append(f"  Output: {tr.output[:200]}")

        context = "\n".join(results_summary)

        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=(
                    "You are ARA. The user asked you to perform a task, and you've executed tool calls. "
                    "Summarize what happened in a natural, conversational way. Be brief."
                ),
            ),
            ChatMessage(role=MessageRole.USER, content=query),
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=f"I executed the following actions:\n{context}",
            ),
            ChatMessage(
                role=MessageRole.USER,
                content="Now give me a brief, natural summary of what you did.",
            ),
        ]

        return await self._llm.generate(messages, max_tokens=300)
