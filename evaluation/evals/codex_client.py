from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx

from evals.codex_oauth import CodexOAuthTokens, ensure_fresh_tokens
from evals.kiconnect import (
    EvaluationResponse,
    _merge_usage,
    _to_responses_text_format,
    _to_responses_tool,
    _tool_result_to_text,
)


DEFAULT_CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
DEFAULT_USER_AGENT = "BAEvaluationCodexOAuth/0.1 (+local eval)"

_MAX_CONCURRENT_REQUESTS = int(os.getenv("CODEX_OAUTH_MAX_CONCURRENT_REQUESTS", "2"))
_REQUEST_SPACING_SECONDS = float(os.getenv("CODEX_OAUTH_REQUEST_SPACING_SECONDS", "0.5"))
_REQUEST_SEMAPHORE = (
    threading.BoundedSemaphore(_MAX_CONCURRENT_REQUESTS)
    if _MAX_CONCURRENT_REQUESTS > 0
    else None
)
_REQUEST_SPACING_LOCK = threading.Lock()
_NEXT_REQUEST_AT = 0.0


class CodexOAuthResponsesAPI:
    """OpenAI-like client.responses.create(...) wrapper for Codex OAuth."""

    def __init__(self, client: "CodexOAuthResponsesClient") -> None:
        self._client = client

    def create(self, **kwargs: Any) -> EvaluationResponse:
        return asyncio.run(self.acreate(**kwargs))

    async def acreate(self, **kwargs: Any) -> EvaluationResponse:
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
                raise RuntimeError("fastmcp ist fuer lokale MCP-Tools erforderlich.") from exc

            async with FastMCPClient(transport) as mcp_client:
                local_tools = await mcp_client.list_tools()

                async def call_tool(name: str, arguments: dict[str, Any]) -> Any:
                    return await mcp_client.call_tool(name, arguments or {})

                return await self._run_responses_loop(
                    instructions=kwargs.get("instructions", ""),
                    prompt=kwargs.get("input", ""),
                    schema=(
                        kwargs.get("text", {}).get("format")
                        if isinstance(kwargs.get("text"), dict)
                        else kwargs.get("schema")
                    ),
                    tools=local_tools,
                    tool_caller=call_tool,
                    reasoning_effort=kwargs.get("reasoning_effort"),
                    reasoning=kwargs.get("reasoning"),
                )

        return await self._run_responses_loop(
            instructions=kwargs.get("instructions", ""),
            prompt=kwargs.get("input", ""),
            schema=(
                kwargs.get("text", {}).get("format")
                if isinstance(kwargs.get("text"), dict)
                else kwargs.get("schema")
            ),
            tools=kwargs.get("tools"),
            tool_caller=kwargs.get("tool_caller"),
            reasoning_effort=kwargs.get("reasoning_effort"),
            reasoning=kwargs.get("reasoning"),
        )

    async def _run_responses_loop(
        self,
        *,
        instructions: str,
        prompt: str | dict[str, Any] | list[dict[str, Any]],
        schema: dict[str, Any] | None = None,
        tools: list[Any] | None = None,
        tool_caller: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None,
        reasoning_effort: str | None = None,
        reasoning: dict[str, Any] | None = None,
    ) -> EvaluationResponse:
        initial_input = _initial_input(prompt)
        followup_initial_input = _followup_input_without_images(initial_input)
        response_input: str | list[dict[str, Any]] = initial_input
        conversation_items: list[dict[str, Any]] = [*followup_initial_input]
        output_items: list[dict[str, Any]] = []
        usage_total = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "reasoning_output_tokens": 0,
        }
        latest_response: dict[str, Any] = {}

        async with httpx.AsyncClient(timeout=self._client.timeout_seconds) as http:
            for _ in range(self._client.max_tool_rounds + 1):
                payload: dict[str, Any] = {
                    "model": self._client.model,
                    "instructions": instructions,
                    "input": response_input,
                    "store": False,
                    "stream": True,
                }

                effort = reasoning_effort or self._client.reasoning_effort
                if effort:
                    payload["reasoning"] = {"effort": effort}
                elif reasoning:
                    payload["reasoning"] = reasoning

                if tools:
                    payload["tools"] = [
                        tool
                        if isinstance(tool, dict) and tool.get("type") in {"mcp", "code_interpreter"}
                        else _to_responses_tool(tool)
                        for tool in tools
                    ]
                    payload["tool_choice"] = "auto"
                if schema:
                    payload["text"] = {"format": _to_responses_text_format(schema)}

                data = await self._client._post(http, payload)
                latest_response = data
                _merge_usage(usage_total, _normalize_usage(data.get("usage") or {}))

                calls = _extract_function_calls(data)
                response_items = data.get("output") or []
                output_items.extend(response_items)
                conversation_items.extend(
                    item
                    for item in response_items
                    if isinstance(item, dict) and item.get("type") in {"function_call", "tool_call"}
                )
                if calls and tool_caller:
                    tool_outputs = []
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
                        tool_outputs.append(
                            {
                                "type": "function_call_output",
                                "call_id": call["call_id"],
                                "output": tool_output,
                            }
                        )
                    conversation_items.extend(tool_outputs)
                    response_input = [*conversation_items]
                    continue

                return EvaluationResponse(
                    output_text=_extract_output_text(data),
                    usage={k: (v or None) for k, v in usage_total.items()},
                    output_items=output_items,
                    raw_response=latest_response,
                )

        raise RuntimeError(f"Maximale Tool-Runden erreicht ({self._client.max_tool_rounds}).")


class CodexOAuthResponsesClient:
    """Responses client using a Codex ChatGPT OAuth token."""

    def __init__(
        self,
        *,
        auth_file: Path,
        model: str,
        api_url: str = DEFAULT_CODEX_RESPONSES_URL,
        include_global_auth: bool = True,
        timeout_seconds: int = 300,
        max_tool_rounds: int = 80,
        reasoning_effort: str | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        if not model:
            raise RuntimeError("EVAL_MODEL ist nicht gesetzt.")
        self.auth_file = auth_file
        self.include_global_auth = include_global_auth
        self.model = model
        self.api_url = api_url
        self.timeout_seconds = timeout_seconds
        self.max_tool_rounds = max_tool_rounds
        self.reasoning_effort = reasoning_effort
        self.user_agent = user_agent
        self.max_api_retries = int(os.getenv("CODEX_OAUTH_MAX_API_RETRIES", "8"))
        self.retry_base_seconds = float(os.getenv("CODEX_OAUTH_RETRY_BASE_SECONDS", "2.0"))
        self.responses = CodexOAuthResponsesAPI(self)

    @property
    def provider(self) -> str:
        return "codex_oauth"

    async def _post(self, http: httpx.AsyncClient, payload: dict[str, Any]) -> dict[str, Any]:
        retryable_statuses = {429, 500, 502, 503, 504}
        response: httpx.Response | None = None
        body = b""

        for attempt in range(self.max_api_retries + 1):
            tokens = self._load_tokens()
            await _throttle_request()
            if _REQUEST_SEMAPHORE:
                _REQUEST_SEMAPHORE.acquire()
            try:
                response = await http.post(
                    self.api_url,
                    headers=self._headers(tokens),
                    json=payload,
                )
                body = response.content
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
        text = body.decode("utf-8", errors="replace")
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"Codex OAuth Responses API error {response.status_code}: {text[:1000]}") from exc

        data = _parse_response_body(text)
        if not data:
            raise RuntimeError("Codex OAuth Responses API returned an empty response.")
        return data

    def _load_tokens(self) -> CodexOAuthTokens:
        return ensure_fresh_tokens(
            self.auth_file,
            include_global=self.include_global_auth,
        )

    def _headers(self, tokens: CodexOAuthTokens) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {tokens.access}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": self.user_agent,
        }
        if tokens.account_id:
            headers["ChatGPT-Account-Id"] = tokens.account_id
        return headers


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


def _image_data_url(path: str) -> str:
    image_path = Path(path).expanduser()
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _initial_input(prompt: str | dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(prompt, list):
        return prompt
    if isinstance(prompt, dict):
        text = str(prompt.get("text") or "")
        image_paths = prompt.get("image_paths") or []
        content: list[dict[str, Any]] = [{"type": "input_text", "text": text}]
        for image_path in image_paths:
            content.append(
                {
                    "type": "input_image",
                    "image_url": _image_data_url(str(image_path)),
                    "detail": "high",
                }
            )
        return [{"role": "user", "content": content}]
    return [{"role": "user", "content": prompt}]


def _followup_input_without_images(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            compacted.append(item)
            continue
        content = item.get("content")
        if not isinstance(content, list):
            compacted.append(dict(item))
            continue

        omitted = 0
        compact_content: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "input_image":
                omitted += 1
                continue
            compact_content.append(part)
        if omitted:
            compact_content.append(
                {
                    "type": "input_text",
                    "text": (
                        f"[{omitted} image attachment(s) were included in the "
                        "initial request and omitted from follow-up tool rounds.]"
                    ),
                }
            )
        compacted_item = dict(item)
        compacted_item["content"] = compact_content
        compacted.append(compacted_item)
    return compacted


def _parse_response_body(text: str) -> dict[str, Any]:
    stripped = text.lstrip()
    if stripped.startswith("{"):
        data = json.loads(stripped)
        data.setdefault("output_text", _extract_output_text(data))
        data.setdefault("usage", _normalize_usage(data.get("usage") or {}))
        return data
    return _parse_sse_response(text)


def _parse_sse_response(text: str) -> dict[str, Any]:
    output_texts: list[str] = []
    events: list[dict[str, Any]] = []
    final_response: dict[str, Any] | None = None

    for block in re.split(r"\r?\n\r?\n", text):
        event_name = ""
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].strip())
        if not data_lines:
            continue
        raw_data = "\n".join(data_lines)
        if raw_data == "[DONE]":
            continue
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError:
            continue
        data["_event"] = event_name
        events.append(data)
        if data.get("type") == "response.output_text.done" and isinstance(data.get("text"), str):
            output_texts.append(data["text"])
        elif data.get("type") == "response.completed" and isinstance(data.get("response"), dict):
            final_response = data["response"]

    output: list[dict[str, Any]] = []
    for event in events:
        if event.get("type") == "response.output_item.done" and isinstance(event.get("item"), dict):
            output.append(event["item"])
    if not output and final_response and isinstance(final_response.get("output"), list):
        output = final_response["output"]

    if output_texts and not _extract_output_text({"output": output}):
        output.append(
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "\n".join(output_texts)}],
            }
        )

    usage = _normalize_usage((final_response or {}).get("usage") or {})
    return {
        "id": (final_response or {}).get("id"),
        "output_text": "\n".join(output_texts).strip(),
        "output": output,
        "usage": usage,
        "response": final_response or {},
        "events": events,
    }


def _normalize_usage(usage: dict[str, Any]) -> dict[str, Any]:
    if not usage:
        return {}
    output_tokens = usage.get("output_tokens")
    details = usage.get("output_tokens_details") or {}
    reasoning_tokens = details.get("reasoning_tokens")
    return {
        "input_tokens": usage.get("input_tokens") or usage.get("prompt_tokens"),
        "output_tokens": output_tokens or usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "reasoning_output_tokens": reasoning_tokens or usage.get("reasoning_output_tokens"),
    }


def _extract_output_text(data: dict[str, Any]) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    chunks: list[str] = []
    for item in data.get("output", []) or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if isinstance(content, dict) and content.get("type") in {"output_text", "text"}:
                chunks.append(str(content.get("text", "")))
    return "\n".join(chunks).strip()


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
        if not isinstance(args, dict):
            args = {}
        calls.append(
            {
                "id": item.get("id"),
                "call_id": item.get("call_id") or item.get("id"),
                "name": name,
                "arguments": args,
            }
        )
    return calls
