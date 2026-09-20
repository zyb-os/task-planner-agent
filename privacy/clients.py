"""
privacy/clients.py — Privacy-aware LLM proxy client.

AnthropicPrivacyClient is a thin wrapper around the orchestrator's internal
LLM proxy endpoint (POST /api/v1/llm/complete).  In this codebase neither
TaskPlanner nor EmergentStepRunner calls the Anthropic SDK directly — all LLM
traffic is centralised through the orchestrator proxy, which in turn dispatches
to Anthropic / OpenAI / Gemini.  The wrapper therefore intercepts the HTTP
payload rather than an SDK client call, but the privacy semantics are identical:

    redact request  →  call real backend  →  restore response

Streaming is *not* used in the current codebase (both callers read the full
JSON response in one shot), so no streaming buffer is needed here.

Usage
-----
    client = AnthropicPrivacyClient(
        proxy_url="http://localhost:8000/api/v1/llm/complete",
        agent_id=my_agent_id,
        privacy_proxy=proxy,          # PrivacyProxy singleton
    )

    text = await client.complete(payload, privacy_ctx=ctx)
"""
from __future__ import annotations

import logging

import httpx

from .context import PrivacyContext
from .proxy import PrivacyProxy

logger = logging.getLogger(__name__)


class AnthropicPrivacyClient:
    """
    Drop-in replacement for the inline ``httpx.AsyncClient`` calls made by
    ``TaskPlanner._proxy_complete`` and ``EmergentStepRunner._llm_call``.

    When a PrivacyContext is supplied and the proxy is enabled the client:
      1. Deep-copies the payload, redacts all message / segment content, and
         injects the privacy note into the system prompt.
      2. POSTs the sanitised payload to the orchestrator LLM proxy.
      3. Extracts the text from the first ``type=text`` content block.
      4. Restores all placeholder tokens in the response text before returning.

    When the proxy is disabled (default) or no context is passed the client
    behaves exactly like the original inline httpx call.
    """

    def __init__(
        self,
        proxy_url: str,
        agent_id: str,
        privacy_proxy: PrivacyProxy,
    ) -> None:
        self._proxy_url = proxy_url
        self._agent_id = agent_id
        self._privacy = privacy_proxy

    async def complete(
        self,
        payload: dict,
        *,
        privacy_ctx: PrivacyContext | None = None,
        timeout: float = 180.0,
    ) -> str:
        """
        Send *payload* to the LLM proxy and return the response text.

        Parameters
        ----------
        payload:
            Full request body for ``POST /api/v1/llm/complete``.
            Supported keys: ``provider``, ``model``, ``messages``, ``system``,
            ``max_tokens``, ``prompt_segments``.
        privacy_ctx:
            When provided (and ``self._privacy.enabled`` is True) the payload
            is redacted before sending and the response is restored before
            returning.
        timeout:
            httpx request timeout in seconds.  Callers should set this to
            match the original per-call timeout (180 s for planning calls,
            120 s for emergent tool-loop calls).
        """
        # ── Optionally redact ─────────────────────────────────────────────────
        if self._privacy.enabled and privacy_ctx is not None:
            payload = self._privacy.apply_to_payload(payload, privacy_ctx)

        # ── HTTP call ─────────────────────────────────────────────────────────
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(
                self._proxy_url,
                headers={"X-Agent-Id": self._agent_id},
                json=payload,
            )
            r.raise_for_status()
            data = r.json()
            text_blocks = [b["text"] for b in data["content"] if b.get("type") == "text" and b.get("text")]
            if not text_blocks:
                raise ValueError(
                    f"LLM response contained no usable text content "
                    f"(stop_reason={data.get('stop_reason')!r}, "
                    f"model={data.get('model')!r}). "
                    "If using a Qwen3/thinking model, the response may have been "
                    "entirely a reasoning block with no answer after </think>."
                )
            text = text_blocks[0]

        # ── Optionally restore ────────────────────────────────────────────────
        if self._privacy.enabled and privacy_ctx is not None:
            text = self._privacy.unwrap_response(text, privacy_ctx)

        return text

    async def complete_structured(
        self,
        payload: dict,
        *,
        privacy_ctx: PrivacyContext | None = None,
        timeout: float = 180.0,
    ) -> dict:
        """
        Like ``complete()`` but returns the full response dict
        (``{"content": [...], "stop_reason": str, ...}``) instead of just
        the first text block.  Required for tool-use responses which contain
        ``type=tool_use`` blocks alongside (or instead of) text blocks.

        Privacy redaction is applied to the request payload exactly as in
        ``complete()``.  The response is returned as-is (tool_use blocks do
        not contain sensitive text that was substituted, so no un-redaction
        is needed for the structured path).
        """
        if self._privacy.enabled and privacy_ctx is not None:
            payload = self._privacy.apply_to_payload(payload, privacy_ctx)

        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(
                self._proxy_url,
                headers={"X-Agent-Id": self._agent_id},
                json=payload,
            )
            r.raise_for_status()
            return r.json()
