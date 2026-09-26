"""Minimal OpenAI-compatible LLM client (Responses or Chat Completions API).

When an MCP tool is passed, the client starts the MCP server over stdio,
exposes its tools as function tools and runs the tool-calling loop locally.
Works with any OpenAI-compatible endpoint (KI:Connect, OpenAI, vLLM, ...).
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import httpx


_MAX_CONCURRENT_REQUESTS = int(os.getenv("EVAL_MAX_CONCURRENT_REQUESTS", "2"))
_REQUEST_SPACING_SECONDS = float(os.getenv("EVAL_REQUEST_SPACING_SECONDS", "0.5"))
_REQUEST_SEMAPHORE = (
    threading.BoundedSemaphore(_MAX_CONCURRENT_REQUESTS)
    if _MAX_CONCURRENT_REQUESTS > 0
    else None
)
_REQUEST_SPACING_LOCK = threading.Lock()
_NEXT_REQUEST_AT = 0.0


@dataclass
class EvaluationResponse:
    output_text: str
    usage: dict[str, Any] = field(default_factory=dict)
    output_items: list[dict[str, Any]] = field(default_factory=list)
    raw_response: dict[str, Any] = field(default_factory=dict)

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        return {
            "usage": self.usage,
            "output": self.output_items,
            "raw_response": self.raw_response,
        }


class ResponsesAPI:
    """OpenAI-like client.responses.create(...) interface."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def create(self, **kwargs: Any) -> EvaluationResponse:
        if self._client.completions_mode:
            return self._client.chat.create(**kwargs)
        return asyncio.run(self.acreate(**kwargs))

    async def acreate(self, **kwargs: Any) -> EvaluationResponse:
        # Check if any tools list has a local MCP server transport
        tools_list = kwargs.get("tools") or []
        local_mcp_tool = None
        for tool in tools_list:
            if isinstance(tool, dict) and tool.get("type") == "mcp":
                transport = tool.get("server_transport") or tool.get("server_url")
                if transport and not isinstance(transport, str):
                    local_mcp_tool = tool
                    break

        if local_mcp_tool is not None:
            transport = local_mcp_tool.get("server_transport") or local_mcp_tool.get("server_url")
            try:
                from fastmcp import Client as FastMCPClient
            except Exception as exc:
                raise RuntimeError("fastmcp is required for local MCP tools.") from exc

            async with FastMCPClient(transport) as mcp_client:
                local_tools = await mcp_client.list_tools()
                
                async def call_tool(name: str, arguments: dict[str, Any]) -> Any:
                    return await mcp_client.call_tool(name, arguments or {})

                return await self._run_responses_loop(
                    instructions=kwargs.get("instructions", ""),
                    prompt=kwargs.get("input", ""),
                    schema=kwargs.get("text", {}).get("format") if isinstance(kwargs.get("text"), dict) else kwargs.get("schema"),
                    tools=local_tools,
                    tool_caller=call_tool,
                    reasoning_effort=kwargs.get("reasoning_effort"),
                    reasoning=kwargs.get("reasoning"),
                )

        tool_caller = kwargs.get("tool_caller")
        return await self._run_responses_loop(
            instructions=kwargs.get("instructions", ""),
            prompt=kwargs.get("input", ""),
            schema=kwargs.get("text", {}).get("format") if isinstance(kwargs.get("text"), dict) else kwargs.get("schema"),
            tools=kwargs.get("tools"),
            tool_caller=tool_caller,
            reasoning_effort=kwargs.get("reasoning_effort"),
            reasoning=kwargs.get("reasoning"),
        )

    async def _run_responses_loop(
        self,
        *,
        instructions: str,
        prompt: str,
        schema: dict[str, Any] | None = None,
        tools: list[Any] | None = None,
        tool_caller: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None,
        reasoning_effort: str | None = None,
        reasoning: dict[str, Any] | None = None,
    ) -> EvaluationResponse:
        response_input: str | list[dict[str, Any]] = prompt
        previous_response_id: str | None = None
        output_items: list[dict[str, Any]] = []
        usage_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        latest_response: dict[str, Any] = {}

        async with httpx.AsyncClient(timeout=self._client.timeout_seconds) as http:
            for _ in range(self._client.max_tool_rounds + 1):
                payload: dict[str, Any] = {
                    "model": self._client.model,
                    "instructions": instructions,
                    "input": response_input,
                }
                if previous_response_id:
                    payload["previous_response_id"] = previous_response_id
                
                effort = reasoning_effort or self._client.reasoning_effort
                if effort:
                    payload["reasoning"] = {"effort": effort}
                elif reasoning:
                    payload["reasoning"] = reasoning

                if tools:
                    payload["tools"] = [
                        tool if isinstance(tool, dict) and tool.get("type") in {"mcp", "code_interpreter"}
                        else _to_responses_tool(tool)
                        for tool in tools
                    ]
                    payload["tool_choice"] = "auto"
                if schema:
                    payload["text"] = {"format": _to_responses_text_format(schema)}

                data = await self._client._post(http, payload)
                latest_response = data
                previous_response_id = data.get("id") or previous_response_id
                _merge_usage(usage_total, data.get("usage") or {})

                calls = _extract_function_calls(data)
                output_items.extend(data.get("output") or [])
                if calls and tool_caller:
                    response_input = []
                    for call in calls:
                        try:
                            result = await tool_caller(call["name"], call["arguments"])
                            tool_output = _tool_result_to_text(result)
                            status = "completed"
                            error = None
                        except Exception as exc:
                            tool_output = f"{type(exc).__name__}: {exc}"
                            status = "error"
                            error = tool_output

                        output_items.append(
                            {
                                "type": "function_call",
                                "id": call.get("id"),
                                "call_id": call.get("call_id"),
                                "name": call["name"],
                                "arguments": json.dumps(call["arguments"], ensure_ascii=False),
                                "output": tool_output,
                                "status": status,
                                "error": error,
                            }
                        )
                        response_input.append(
                            {
                                "type": "function_call_output",
                                "call_id": call["call_id"],
                                "output": tool_output,
                            }
                        )
                    continue

                return EvaluationResponse(
                    output_text=_extract_output_text(data),
                    usage={k: (v or None) for k, v in usage_total.items()},
                    output_items=output_items,
                    raw_response=latest_response,
                )

        raise RuntimeError(f"Maximum tool rounds reached ({self._client.max_tool_rounds}).")


class ChatCompletionsAPI:
    """Chat Completions (/v1/chat/completions) client with local MCP tool-calling loop."""

    def __init__(self, client: "LLMClient") -> None:
        self._client = client

    def create(self, **kwargs: Any) -> EvaluationResponse:
        return asyncio.run(self.acreate(**kwargs))

    async def acreate(self, **kwargs: Any) -> EvaluationResponse:
        # Same local MCP transport detection as ResponsesAPI
        tools_list = kwargs.get("tools") or []
        local_mcp_tool = None
        for tool in tools_list:
            if isinstance(tool, dict) and tool.get("type") == "mcp":
                transport = tool.get("server_transport") or tool.get("server_url")
                if transport and not isinstance(transport, str):
                    local_mcp_tool = tool
                    break

        if local_mcp_tool is not None:
            transport = local_mcp_tool.get("server_transport") or local_mcp_tool.get("server_url")
            try:
                from fastmcp import Client as FastMCPClient
            except Exception as exc:
                raise RuntimeError("fastmcp is required for local MCP tools.") from exc

            async with FastMCPClient(transport) as mcp_client:
                local_tools = await mcp_client.list_tools()

                async def call_tool(name: str, arguments: dict[str, Any]) -> Any:
                    return await mcp_client.call_tool(name, arguments or {})

                return await self._run_chat_loop(
                    instructions=kwargs.get("instructions", ""),
                    prompt=kwargs.get("input", ""),
                    schema=kwargs.get("schema"),
                    tools=local_tools,
                    tool_caller=call_tool,
                )

        return await self._run_chat_loop(
            instructions=kwargs.get("instructions", ""),
            prompt=kwargs.get("input", ""),
            schema=kwargs.get("schema"),
            tools=None,
            tool_caller=kwargs.get("tool_caller"),
        )

    async def _run_chat_loop(
        self,
        *,
        instructions: str,
        prompt: str,
        schema: dict[str, Any] | None = None,
        tools: list[Any] | None = None,
        tool_caller: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None,
    ) -> EvaluationResponse:
        messages: list[dict[str, Any]] = []
        if instructions:
            messages.append({"role": "system", "content": instructions})
        messages.append({"role": "user", "content": prompt})

        # When a JSON schema is required, inject an explicit system instruction.
        # Some models ignore response_format silently, so this ensures JSON output.
        if schema:
            schema_inner = schema.get("schema") or {}
            schema_str = json.dumps(schema_inner, ensure_ascii=False, indent=2)
            json_instruction = (
                "Du musst AUSSCHLIESSLICH ein valides JSON-Objekt zurückgeben, "
                f"das folgendem Schema entspricht:\n```json\n{schema_str}\n```\n"
                "Gib keinen anderen Text aus – nur das rohe JSON-Objekt, ohne Markdown-Codeblöcke."
            )
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] = messages[0]["content"] + "\n\n" + json_instruction
            else:
                messages.insert(0, {"role": "system", "content": json_instruction})

        output_items: list[dict[str, Any]] = []
        usage_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        latest_response: dict[str, Any] = {}

        chat_tools = [_to_chat_tool(t) for t in tools] if tools else None

        async with httpx.AsyncClient(timeout=self._client.timeout_seconds) as http:
            for _ in range(self._client.max_tool_rounds + 1):
                payload: dict[str, Any] = {
                    "model": self._client.model,
                    "messages": messages,
                }
                if schema:
                    schema_inner = schema.get("schema") or {}
                    # Try json_schema first (OpenAI-style strict enforcement);
                    # falls back gracefully to json_object on unsupported models.
                    try:
                        payload["response_format"] = {
                            "type": "json_schema",
                            "json_schema": {
                                "name": schema.get("name", "evaluation_response"),
                                "strict": schema.get("strict", True),
                                "schema": schema_inner,
                            },
                        }
                    except Exception:
                        payload["response_format"] = {"type": "json_object"}
                if chat_tools:
                    payload["tools"] = chat_tools
                    payload["tool_choice"] = "auto"

                data = await self._client._post(http, payload, url=self._client.completions_url)
                latest_response = data

                usage = data.get("usage") or {}
                _merge_usage(usage_total, {
                    "input_tokens": usage.get("prompt_tokens"),
                    "output_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                })

                choice = (data.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                content = message.get("content") or ""
                tc_list = message.get("tool_calls") or []

                if tc_list and tool_caller:
                    messages.append({"role": "assistant", "content": content, "tool_calls": tc_list})
                    for tc in tc_list:
                        fn = tc.get("function") or {}
                        name = fn.get("name")
                        raw_args = fn.get("arguments") or "{}"
                        tc_id = tc.get("id", "")
                        try:
                            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                        except json.JSONDecodeError:
                            args = {}
                        try:
                            result = await tool_caller(name, args if isinstance(args, dict) else {})
                            tool_output = _tool_result_to_text(result)
                            status = "completed"
                            error = None
                        except Exception as exc:
                            tool_output = f"{type(exc).__name__}: {exc}"
                            status = "error"
                            error = tool_output

                        output_items.append({
                            "type": "function_call",
                            "id": tc_id,
                            "call_id": tc_id,
                            "name": name,
                            "arguments": raw_args,
                            "output": tool_output,
                            "status": status,
                            "error": error,
                        })
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": tool_output,
                        })
                    continue

                return EvaluationResponse(
                    output_text=content,
                    usage={k: (v or None) for k, v in usage_total.items()},
                    output_items=output_items,
                    raw_response=latest_response,
                )

        raise RuntimeError(f"Maximum tool rounds reached ({self._client.max_tool_rounds}).")


class LLMClient:
    """OpenAI-compatible client exposing `client.responses.create(...)`."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        completions_mode: bool = False,
        timeout_seconds: int = 300,
        max_tool_rounds: int = 80,
        reasoning_effort: str | None = None,
    ):
        if not api_key:
            raise RuntimeError("EVAL_API_KEY is not set (see evaluation/.env.example).")
        if not model:
            raise RuntimeError("No model set (--model or EVAL_MODEL).")
        self.api_key = api_key
        self.model = model
        base_url = base_url.rstrip("/")
        self.api_url = f"{base_url}/responses"
        self.completions_url = f"{base_url}/chat/completions"
        self.completions_mode = completions_mode
        self.timeout_seconds = timeout_seconds
        self.max_tool_rounds = max_tool_rounds
        self.reasoning_effort = reasoning_effort
        self.max_api_retries = 8
        self.retry_base_seconds = 2.0
        self.responses = ResponsesAPI(self)
        self.chat = ChatCompletionsAPI(self)

    @property
    def provider(self) -> str:
        return "chat_completions" if self.completions_mode else "responses"

    async def _post(self, http: httpx.AsyncClient, payload: dict[str, Any], url: str | None = None) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        target_url = url or self.api_url
        retryable_statuses = {429, 500, 502, 503, 504}
        response: httpx.Response | None = None

        for attempt in range(self.max_api_retries + 1):
            await _throttle_request()
            if _REQUEST_SEMAPHORE:
                _REQUEST_SEMAPHORE.acquire()
            try:
                response = await http.post(target_url, headers=headers, json=payload)
            finally:
                if _REQUEST_SEMAPHORE:
                    _REQUEST_SEMAPHORE.release()

            if response.status_code not in retryable_statuses or attempt >= self.max_api_retries:
                break

            retry_after = response.headers.get("retry-after")
            if retry_after:
                try:
                    delay = float(retry_after)
                except ValueError:
                    delay = self.retry_base_seconds
            else:
                delay = min(30.0, self.retry_base_seconds * (2 ** attempt))
            await asyncio.sleep(delay + random.uniform(0.0, 1.5))

        assert response is not None
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"LLM API error {response.status_code}: {response.text[:1000]}"
            ) from exc
        return response.json()


async def _throttle_request() -> None:
    global _NEXT_REQUEST_AT
    if _REQUEST_SPACING_SECONDS <= 0:
        return
    with _REQUEST_SPACING_LOCK:
        now = time.monotonic()
        scheduled_at = max(now, _NEXT_REQUEST_AT)
        _NEXT_REQUEST_AT = scheduled_at + _REQUEST_SPACING_SECONDS
        wait_seconds = scheduled_at - now
    if wait_seconds > 0:
        await asyncio.sleep(wait_seconds)


def _to_chat_tool(tool: Any) -> dict[str, Any]:
    """Convert a tool to the Chat Completions function format."""
    name = _tool_name(tool)
    description = getattr(tool, "description", None) if not isinstance(tool, dict) else tool.get("description")
    parameters = (
        getattr(tool, "inputSchema", None)
        if not isinstance(tool, dict)
        else tool.get("inputSchema") or tool.get("parameters")
    )
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description or "",
            "parameters": parameters or {"type": "object", "properties": {}},
        },
    }


def _to_responses_tool(tool: Any) -> dict[str, Any]:
    name = _tool_name(tool)
    description = getattr(tool, "description", None) if not isinstance(tool, dict) else tool.get("description")
    parameters = (
        getattr(tool, "inputSchema", None)
        if not isinstance(tool, dict)
        else tool.get("inputSchema") or tool.get("parameters")
    )
    return {
        "type": "function",
        "name": name,
        "description": description or "",
        "parameters": parameters or {"type": "object", "properties": {}},
    }


def _tool_name(tool: Any) -> str:
    name = getattr(tool, "name", None) if not isinstance(tool, dict) else tool.get("name")
    if not name:
        raise ValueError(f"Tool ohne Namen: {tool!r}")
    return name


def _to_responses_text_format(schema: dict[str, Any]) -> dict[str, Any]:
    if schema.get("type") != "json_schema":
        return schema
    return {
        "type": "json_schema",
        "name": schema.get("name", "evaluation_response"),
        "strict": schema.get("strict", True),
        "schema": schema.get("schema") or {},
    }


def _extract_function_calls(data: dict[str, Any]) -> list[dict[str, Any]]:
    calls = []
    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") not in {"function_call", "tool_call"}:
            continue
        name = item.get("name")
        raw_args = item.get("arguments") or "{}"
        if not name:
            continue
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            args = {}
        calls.append(
            {
                "id": item.get("id"),
                "call_id": item.get("call_id") or item.get("id"),
                "name": name,
                "arguments": args if isinstance(args, dict) else {},
            }
        )
    return calls


def _extract_output_text(data: dict[str, Any]) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    parts: list[str] = []
    for item in data.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if text is None:
                text = content.get("output_text")
            if text is not None:
                parts.append(str(text))
    return "".join(parts)


def _tool_result_to_text(result: Any) -> str:
    data = _tool_result_to_jsonable(result)
    if isinstance(data, str):
        return data
    return json.dumps(data, ensure_ascii=False)


def _tool_result_to_jsonable(result: Any) -> Any:
    if result is None or isinstance(result, (str, int, float, bool)):
        return result
    if isinstance(result, (dict, list)):
        return _json_safe(result)

    for attr in ("structured_content", "structuredContent"):
        value = getattr(result, attr, None)
        if value is not None:
            if isinstance(value, dict) and set(value.keys()) == {"result"}:
                return _json_safe(value["result"])
            return _json_safe(value)

    content = getattr(result, "content", None)
    if isinstance(content, list):
        text_parts = []
        for item in content:
            text = getattr(item, "text", None)
            if text is None and isinstance(item, dict):
                text = item.get("text")
            if text is not None:
                text_parts.append(str(text))
        if text_parts:
            return "\n".join(text_parts)

    value = getattr(result, "data", None)
    if value is not None:
        return _json_safe(value)

    if hasattr(result, "model_dump"):
        try:
            return _json_safe(result.model_dump(mode="json"))
        except Exception:
            pass
    return str(result)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump(mode="json"))
        except Exception:
            pass
    return str(value)


def _merge_usage(total: dict[str, int], usage: dict[str, Any]) -> None:
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens")) or 0
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens")) or 0
    total_tokens = usage.get("total_tokens") or (input_tokens + output_tokens)
    total["input_tokens"] += int(input_tokens or 0)
    total["output_tokens"] += int(output_tokens or 0)
    total["total_tokens"] += int(total_tokens or 0)

