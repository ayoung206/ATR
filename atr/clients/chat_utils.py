"""LLM client used by every ATR component that talks to a chat model.

One entry point, :func:`get_chat_result`, dispatches on ``llm_config["provider"]``:

``openai``
    Any OpenAI-compatible endpoint: the managed API, or a local vLLM server
    serving an open-weight backbone.
``vertex_openai_compat``
    Google Vertex AI's OpenAI-compatible surface. The bearer token is minted
    from a service-account JSON on each call, so tokens never sit in config.
``anthropic_vertex``
    Claude models served through Vertex, via the Anthropic SDK.

The return value is always an object exposing ``.content`` so callers can stay
provider-agnostic.

Set ``LLM_USAGE_LOG=<path>`` to append one JSON line of token accounting per
call; :mod:`atr.evaluate` and the cost analysis in the paper read that file.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

_TOKEN_TTL_SEC = 45 * 60          # Vertex access tokens live one hour
_token_cache: Dict[str, tuple] = {}


# ──────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────
def init_logger(
    name: str = "atr",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Return a logger writing to stderr and, if given, to ``log_file`` too.

    Re-calling with the same ``name`` replaces the handlers rather than adding
    a second copy, so repeated CLI entry points do not duplicate every line.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    for stale in list(logger.handlers):
        logger.removeHandler(stale)

    line = logging.Formatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setFormatter(line)
    logger.addHandler(console)

    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)) or ".", exist_ok=True)
        to_file = logging.FileHandler(log_file, encoding="utf-8")
        to_file.setFormatter(line)
        logger.addHandler(to_file)

    return logger


# ──────────────────────────────────────────────────────────────────────────
# Credentials
# ──────────────────────────────────────────────────────────────────────────
def _service_account_field(credentials_path: str, field: str) -> str:
    with open(credentials_path, "r", encoding="utf-8") as handle:
        value = json.load(handle).get(field, "")
    if not value:
        raise ValueError(f"'{field}' missing from {credentials_path}")
    return value


def _vertex_bearer_token(credentials_path: str) -> str:
    """Mint (and briefly cache) an OAuth token for a Vertex service account."""
    cached = _token_cache.get(credentials_path)
    if cached and time.time() < cached[1]:
        return cached[0]

    from google.auth.transport.requests import Request
    from google.oauth2 import service_account

    creds = service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    creds.refresh(Request())
    _token_cache[credentials_path] = (creds.token, time.time() + _TOKEN_TTL_SEC)
    return creds.token


def _resolve_api_key(llm_config: Dict[str, Any]) -> str:
    """Credential for one backbone: a fresh Vertex token, or a static key."""
    path = llm_config.get("credentials_path")
    if llm_config.get("provider") == "vertex_openai_compat" and path:
        return _vertex_bearer_token(path)
    return llm_config.get("api_key", "")


# ──────────────────────────────────────────────────────────────────────────
# Token accounting
# ──────────────────────────────────────────────────────────────────────────
def _record_usage(model: str, usage: Any) -> None:
    path = os.getenv("LLM_USAGE_LOG", "").strip()
    if not path or usage is None:
        return
    details = getattr(usage, "completion_tokens_details", None)
    row = {
        "model": model,
        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
        "completion_tokens": getattr(usage, "completion_tokens", 0),
        "reasoning_tokens": getattr(details, "reasoning_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0),
        "ts": time.time(),
    }
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
    except OSError:
        pass          # accounting must never break inference


class _Reply:
    """Minimal stand-in for an OpenAI message object."""

    __slots__ = ("content", "tool_calls")

    def __init__(self, content: str, tool_calls: Any = None) -> None:
        self.content = content
        self.tool_calls = tool_calls


# ──────────────────────────────────────────────────────────────────────────
# Per-backbone request shaping
# ──────────────────────────────────────────────────────────────────────────
def _decoding_kwargs(model: str) -> Dict[str, Any]:
    """Sampling settings ATR relies on, keyed off the model family.

    Every route emits structured JSON, so the goal throughout is stable
    formatting rather than diverse text.
    """
    if model.startswith("gpt-5"):
        # Reasoning models reject an explicit temperature.
        effort = os.getenv("GPT5_REASONING_EFFORT", "high")
        return {"reasoning_effort": effort} if effort else {}
    if model.startswith("Llama-4"):
        return {"temperature": 0.0}          # greedy keeps Scout's JSON well-formed
    if model.startswith("Qwen3"):
        return {
            "temperature": 0.1,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
    return {"temperature": 0.1}


def _claude_reply(
    messages: List[Dict],
    tools: Any,
    llm_config: Dict[str, Any],
) -> _Reply:
    """Call Claude on Vertex and flatten the reply to plain text."""
    from anthropic import AnthropicVertex

    client = AnthropicVertex(
        project_id=_service_account_field(llm_config["credentials_path"], "project_id"),
        region=llm_config.get("region", "global"),
    )

    system_prompt = "".join(
        m.get("content", "") for m in messages if m.get("role") == "system"
    )
    turns = [m for m in messages if m.get("role") != "system"]

    budget = int(os.getenv("CLAUDE_MAX_TOKENS", "8192"))
    kwargs: Dict[str, Any] = {
        "model": llm_config["model"],
        "messages": turns,
        "max_tokens": budget,
        "temperature": 0.1,
    }
    if system_prompt:
        kwargs["system"] = system_prompt
    if tools:
        kwargs["tools"] = tools

    effort = llm_config.get("thinking_effort") or os.getenv("CLAUDE_THINKING_EFFORT", "")
    if effort:
        kwargs["thinking"] = {"type": "adaptive"}
        kwargs["extra_body"] = {"output_config": {"effort": effort}}
        kwargs["max_tokens"] = max(budget, 16384)
        kwargs.pop("temperature", None)      # incompatible with adaptive thinking

    reply = client.messages.create(**kwargs)
    text = "".join(
        block.text for block in reply.content if getattr(block, "type", "") == "text"
    )
    return _Reply(text)


# ──────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────
def get_chat_result(
    messages: List[Dict],
    tools: Any = None,
    tool_choice: Any = None,
    llm_config: Optional[Dict[str, Any]] = None,
):
    """Send ``messages`` to the backbone named by ``llm_config``.

    Returns an object with a ``.content`` string. Transient failures are
    retried with exponential backoff; the last error is re-raised so a
    misconfigured backbone fails loudly instead of silently answering "".
    """
    cfg = dict(llm_config or {})
    if cfg.get("provider") == "anthropic_vertex":
        return _claude_reply(messages, tools, cfg)

    from openai import OpenAI

    model = cfg.get("model", "gpt-4o")
    attempts = max(1, int(os.getenv("LLM_MAX_RETRIES", "3")))
    last_error: Optional[BaseException] = None

    for attempt in range(attempts):
        try:
            client = OpenAI(api_key=_resolve_api_key(cfg), base_url=cfg.get("url", ""))
            request: Dict[str, Any] = {"messages": messages, "model": model}
            request.update(_decoding_kwargs(model))
            if tools:
                request["tools"] = tools
                if tool_choice:
                    request["tool_choice"] = tool_choice

            completion = client.chat.completions.create(**request)
            _record_usage(model, getattr(completion, "usage", None))
            return completion.choices[0].message
        except Exception as error:            # noqa: BLE001 - retry anything transient
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(2 ** attempt)

    raise RuntimeError(f"LLM call to '{model}' failed after {attempts} attempts") from last_error
