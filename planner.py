"""
planner.py — LLM-based workflow planner.

Discovers available agent capabilities (cached with TTL to minimise REST calls),
then issues a single Anthropic API call to generate a structured WorkflowPlan
from a natural-language goal string.
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx

from models import WorkflowPlan, WorkflowStep
from privacy import AnthropicPrivacyClient, PrivacyContext, PrivacyProxy

logger = logging.getLogger(__name__)

# ── Capability cache TTL ───────────────────────────────────────────────────────
_CAPS_CACHE_TTL_S: float = 60.0   # re-fetch agents at most once per minute
_COMPACT_CAP_LIMIT: int = 10
_MAX_MEMORY_CONTEXT_CHARS: int = 1200
# Micro type-codes for the dense compact format.  String — the modal field
# type — carries no code at all (its absence implies string), so only the
# rarer non-string types cost a character.  Each code is a single character
# to stay one BPE token.
_TYPE_MICRO: dict[str, str] = {
    "string": "", "integer": "i", "number": "n",
    "boolean": "b", "object": "o", "array": "a",
}
# Boilerplate openers LLM-authored capability descriptions tend to start with.
# Stripping them is lossless for planning and saves 3–6 tokens per capability.
_DESC_PREFIX_RE = re.compile(
    r"^(use this (capability |tool )?to |this (capability |tool |agent )|"
    r"allows? (you )?to |enables? (you )?to |lets? you |used to )",
    re.IGNORECASE,
)


def _micro_field(name: str, properties: dict, required: bool) -> str:
    """Render one field as a micro-token: ``name`` (+ ``:code`` for non-string)
    (+ ``!`` when required).  Examples: ``task!``  ``channel!``  ``limit:i``."""
    raw_type = (properties.get(name) or {}).get("type", "")
    if isinstance(raw_type, list):
        raw_type = raw_type[0] if raw_type else ""
    code = "" if not raw_type else _TYPE_MICRO.get(raw_type, "?")
    token = f"{name}:{code}" if code else name
    return f"{token}!" if required else token
_SENSITIVE_KEYWORDS = (
    "password", "passcode", "otp", "token", "api key", "secret", "ssn",
    "social security", "credit card", "card number", "cvv", "bank account",
    "routing number", "dob", "date of birth", "medical", "diagnosis",
)

_PROMPT_FILE = Path(__file__).parent / "prompts" / "system_prompt.md"

# ── Tool-discovery mode constants ──────────────────────────────────────────────
_TOOL_DISCOVERY_MAX_TURNS: int = 6   # cap on search+plan round-trips

_TOOL_DISCOVERY_SYSTEM_PROMPT = """\
You are a workflow planning AI. Your job is to plan a sequence of agent capability calls that accomplishes the user's goal.

Process:
1. Call search_capabilities with 2–4 keyword phrases that describe the kinds of agents/actions you need.
   You may call it multiple times to explore different parts of the goal.
2. Once you have found the exact capabilities you need, call generate_plan with the complete workflow.

Rules for generate_plan:
- Only use capabilities returned by search_capabilities; never invent capability names.
- The `capability` field must be the exact capability identifier shown in search results
  (e.g. `execute_code`), NOT the agent name.
- Steps run sequentially. Reference earlier outputs with {{steps[N].output.field}} (0-indexed).
- Keep step count minimal — avoid unnecessary steps.
- Resolve relative dates against CURRENT_UTC. schedule_task.scheduled_at must be ISO 8601 UTC.
- Prefer lower-cost capabilities when quality is equivalent.
- Slack output: always insert a format_step_output step before send_slack_message when referencing capability output. browse_web returns only {summary: str} — never reference .order_number, .delivery_date, .price, or any structured field from browse_web output directly in send_slack_message.text.
- Do NOT add a final messaging/notification step (send_slack_message, send_email, etc.) to deliver your result — the orchestration layer handles delivery to the originating channel automatically after all steps complete.
- memory_entries (optional, max 3): stable user facts. Omit transient details.
"""

_SEARCH_CAPABILITIES_TOOL: dict = {
    "name": "search_capabilities",
    "description": (
        "Search for agent capabilities by keyword phrases. "
        "Returns matching capabilities with their input schemas and costs. "
        "Call this before generate_plan to discover what agents are available. "
        "You can call it multiple times with different keywords."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "2–4 natural-language keyword phrases describing the capability you need. "
                    "Examples: ['send email gmail'], ['browse web scrape'], "
                    "['execute python code'], ['search web news'], ['slack message notify']"
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Max number of capabilities to return (default: 6, max: 12).",
                "default": 6,
            },
        },
        "required": ["keywords"],
    },
}

_GENERATE_PLAN_TOOL: dict = {
    "name": "generate_plan",
    "description": (
        "Generate the final workflow plan. Call this once you have found all needed capabilities "
        "via search_capabilities. The plan will be executed step-by-step by the task executor."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Short workflow title (max 60 chars).",
            },
            "description": {
                "type": "string",
                "description": "One-sentence description of what this workflow accomplishes.",
            },
            "steps": {
                "type": "array",
                "description": "Ordered list of steps. Each step calls one agent capability.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Step name (max 40 chars)."},
                        "goal": {"type": "string", "description": "Why this step is needed."},
                        "description": {"type": "string", "description": "What this step does."},
                        "capability": {
                            "type": "string",
                            "description": "Exact capability identifier from search_capabilities results.",
                        },
                        "input_data": {
                            "type": "object",
                            "description": "Input values for the capability. Use {{steps[N].output.field}} for chaining.",
                        },
                    },
                    "required": ["name", "goal", "description", "capability", "input_data"],
                },
            },
            "memory_entries": {
                "type": "array",
                "description": "Optional stable user facts to persist (max 3).",
                "items": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": ["Facts", "Preferences", "Patterns", "Instructions"],
                        },
                        "content": {"type": "string"},
                    },
                    "required": ["category", "content"],
                },
            },
        },
        "required": ["title", "description", "steps"],
    },
}


def _load_plan_system_prompt() -> str:
    try:
        return _PROMPT_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    # inline fallback (same content as the MD file)
    return """\
You are a workflow planning AI. Output valid JSON only — no extra text:
{
  "title": "Short title (max 60 chars)",
  "description": "One-sentence description",
  "steps": [
    {
      "name": "Step name (max 40 chars)",
      "goal": "Why this step is needed",
      "description": "What this step does",
      "capability": "exact_capability_name",
      "input_data": { "key": "value" }
    }
  ],
  "memory_entries": [
    { "category": "Facts|Preferences|Patterns|Instructions", "content": "concise fact" }
  ]
}

Rules:
- Conversation context: In multi-turn requests, read the prior assistant message first
  to interpret the user's reply. Map each answer to the question that prompted it
  (e.g. "Tesla" after "What car do you drive?" = user drives a Tesla). Never treat a
  contextual reply as a standalone goal.
- Only use capabilities from the provided list; use concrete input_data (no placeholders).
  CRITICAL: the `capability` field must be the capability identifier shown indented under
  the agent header — e.g. `execute_code`, NOT the agent name `code-execution-agent`.
  Agent names appear in brackets like `[code-execution-agent]` and are never valid
  capability values. Using an agent name will cause the step to fail.
- Steps run sequentially. Reference earlier outputs with {{steps[N].output.field}} (0-indexed).
- Keep step count minimal.
- Resolve relative dates against CURRENT_UTC. schedule_task.scheduled_at must be ISO 8601 UTC in the future.
- Prefer lower-cost capabilities when quality is equivalent.
- Slack output: never hardcode field refs in send_slack_message. Insert format_step_output
  before it (input_data.data={{steps[N].output}}, input_data.capability_name=<step N cap>);
  the send step uses {{steps[K].output.text}}. Only use it for explicit user-requested
  messages (e.g. "post this to #channel") — never add it as an auto-completion step.
- Do NOT add a final messaging/notification step (send_slack_message, send_email, etc.)
  to deliver the result — the orchestration layer routes the result back to the originating
  channel automatically after all steps complete. Only include a messaging step when the
  user explicitly asks to send a message to a specific target (channel, email address, etc.).
- If you include summarize_content steps, pass through request preferences into
  input_data when provided: persona -> input_data.persona, DELIVERY_CHANNEL ->
  input_data.delivery_channel, SUMMARY_FORMAT -> input_data.output_format.
- memory_entries (optional, max 3): stable user facts — preferences, standing facts,
  recurring patterns, explicit instructions. Omit transient details and duplicates.
- Memory queries ("Do you know my X?"): if memory has the answer report it; if not,
  ask the user to share it in one step.
- User-provided info (e.g. "Tesla" answering "What car?"): one step — confirm and store
  in memory_entries. Do not research or act on the info further.
"""


_PLAN_SYSTEM_PROMPT: str = _load_plan_system_prompt()

_SESSION_HISTORY_MAX_CHARS: int = 1500  # keep injected history compact


def _format_session_history(history: list[dict]) -> str:
    """
    Format recent conversation turns into a compact context block for the LLM.
    *history* is a list of {"sender": str, "text": str} dicts, oldest first.
    Returns an empty string when history is empty.
    """
    if not history:
        return ""
    lines: list[str] = ["Recent conversation context (use this to avoid redundant questions and build on prior context):"]
    for turn in history:
        sender = str(turn.get("sender", "user")).strip()
        text   = str(turn.get("text", "")).strip()
        if not text:
            continue
        role = "User" if sender == "user" else "Assistant"
        # Truncate very long turns
        if len(text) > 400:
            text = text[:397] + "…"
        lines.append(f"[{role}]: {text}")
    if len(lines) == 1:
        return ""  # nothing but the header
    block = "\n".join(lines)
    # Hard cap to keep total prompt size reasonable
    if len(block) > _SESSION_HISTORY_MAX_CHARS:
        block = block[:_SESSION_HISTORY_MAX_CHARS] + "\n[...earlier history truncated]"
    return "\n" + block + "\n"


_CLARIFICATION_SYSTEM_PROMPT = """\
Assess whether the goal needs clarification before planning. Output JSON only:

Clear: {"needs_clarification": false}
Unclear: {"needs_clarification": true, "questions": ["Q1?", "Q2?"], "understood_as": "restatement"}

Rules:
- Prior-question replies: read prior questions before judging. A short answer is resolved
  by the question that prompted it — output false. Do not re-ask for info already given.
- Max 3 questions. Ask only for genuinely missing info (targets, scope, constraints).
  Do not ask about things that can be inferred or that capabilities make unnecessary.
- Memory context = confirmed facts. Do not ask about anything already in memory.
- Session history = this conversation's prior exchanges. Do NOT ask about anything
  the user already told you in recent messages. Extract facts from prior turns.
- If the goal asks what you know about the user (e.g. "Do you know my car?"),
  output false — the planner resolves it via memory lookup.
"""


_READINESS_SYSTEM_PROMPT = """\
Assess whether a clarification conversation has provided enough information to generate
a fully executable workflow plan. Output JSON only:

Ready:     {"ready": true}
Not ready: {"ready": false, "follow_up_questions": ["Q1?", "Q2?"], "understood_as": "restatement"}

Rules:
- "ready" = true when: targets/recipients/constraints are concrete, every plan step can
  be expressed with real capability inputs (no placeholders or guesses required), and
  there is no ambiguity about scope or desired output format.
- "ready" = false ONLY when missing information would cause a step to fail at runtime
  (e.g. unknown recipient email, unspecified date, ambiguous target system).
- Maximum 2 follow-up questions per round. Ask only for what is genuinely missing.
- Do NOT re-ask anything already answered in the conversation history.
- If answers are vague but workable with reasonable defaults, lean toward ready=true.
- Memory context = confirmed facts. Do not ask about anything already in memory.
"""

_FOLLOWUP_MEMORY_SYSTEM_PROMPT = """\
You answer whether a follow-up question can be resolved from provided memory context only.
Return strict JSON only:
{"found": true|false, "answer": "string", "confidence": 0.0-1.0}

Rules:
- Use ONLY the provided memory context; do not infer beyond it.
- If context is insufficient, return {"found": false, "answer": "", "confidence": 0.0}.
- Keep answer concise and directly usable as a tool input value.
"""


_DECOMPOSE_SYSTEM_PROMPT = """\
Analyze a goal and identify the capability domains and execution phases needed.
Output strict JSON only:
{
  "complexity": "simple" | "moderate" | "complex",
  "phases": [
    {
      "name": "Short phase name",
      "description": "What this phase accomplishes",
      "search_queries": ["query1", "query2"]
    }
  ]
}

Rules:
- simple: 1-2 steps, single capability domain (e.g. just send email). 1 phase.
- moderate: 3-5 steps, 2-3 domains (e.g. search + summarise + notify). 2-3 phases.
- complex: 6+ steps or 3+ distinct capability domains. 3-4 phases.
- search_queries: 2-3 natural-language phrases that describe what kind of agent
  capability is needed for that phase (used for semantic capability retrieval).
  Be specific: "read emails from gmail", "search the web for news", "send slack message".
- Keep phases to the minimum needed; merge closely related steps.
"""


class TaskPlanner:
    """Discovers capabilities (cached) and uses one LLM call to produce a plan."""

    def __init__(
        self,
        orchestrator_base_url: str,
        agent_id: str = "",
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 2048,
        privacy_proxy: PrivacyProxy | None = None,
    ) -> None:
        self._base = orchestrator_base_url.rstrip("/")
        self._agent_id = agent_id
        self._proxy_url = f"{self._base}/api/v1/llm/complete"
        self._default_model = model
        self._model = model
        self._provider = "anthropic"
        self._max_tokens = max_tokens

        # Per-task-type model overrides.
        # _model_fast   — cheap/fast model for classification, decomposition,
        #                 clarification checks, and memory lookups.
        #                 Falls back to _model when not set.
        # _model_plan   — powerful model for actual plan generation.
        #                 Falls back to _model when not set.
        self._model_fast: str = ""
        self._model_plan: str = ""

        # Privacy proxy — set once here; enabled flag toggled via set_privacy_proxy / settings_push
        _privacy = privacy_proxy or PrivacyProxy()
        self._llm_client = AnthropicPrivacyClient(
            proxy_url=self._proxy_url,
            agent_id=self._agent_id,
            privacy_proxy=_privacy,
        )

        # System prompt (hot-reloadable via update_prompt)
        self._plan_system_prompt: str = _PLAN_SYSTEM_PROMPT

        # Capability cache: (fetch_time, agents_list)
        self._caps_cache_time: float = 0.0
        self._caps_cache: list[dict] = []

        # Agent metadata cache: agent_name → metadata dict (refreshed with capability cache)
        self._agent_meta_cache: dict[str, dict] = {}

        # Vector search settings — updated live via update_settings()
        self._vector_search_enabled: bool = False
        self._vector_search_top_k: int = 8
        self._vector_search_multiphase: bool = False

        # Tool-discovery mode — LLM drives capability lookup via search_capabilities tool
        self._tool_discovery_enabled: bool = False

        # Skill learning — replay past plans instead of re-planning (zero tokens)
        self._skill_learning_enabled: bool = False
        self._skill_replay_threshold: float = 0.75

    def update_prompt(self, content: str) -> None:
        """Hot-reload the planning system prompt (called on prompt_push from orchestrator)."""
        if content and content.strip():
            self._plan_system_prompt = content.strip()
            logger.info("Plan system prompt updated (%d chars)", len(self._plan_system_prompt))

    def update_settings(self, common_settings: dict, agent_settings: dict | None = None) -> None:
        """Apply common_settings and agent-specific settings pushed from the orchestrator."""
        agent_settings = agent_settings or {}

        # ── Model / provider resolution ────────────────────────────────────────
        # Priority: agent setting → global default → compiled-in default
        for candidate in (
            str(agent_settings.get("planner_model", "")).strip(),
            str(common_settings.get("default_model", "")).strip(),
            self._default_model,
        ):
            if candidate:
                self._model = candidate
                break

        # Per-task-type model overrides (empty string = fall back to _model).
        # planner_model_fast  → quick tasks: classification, decomposition,
        #                       clarification, memory lookup.
        # planner_model_plan  → heavy tasks: actual plan/workflow generation.
        self._model_fast = (
            str(agent_settings.get("planner_model_fast", "")).strip()
            or str(common_settings.get("planner_model_fast", "")).strip()
        )
        self._model_plan = (
            str(agent_settings.get("planner_model_plan", "")).strip()
            or str(common_settings.get("planner_model_plan", "")).strip()
        )

        for candidate in (
            str(agent_settings.get("planner_provider", "")).strip(),
            str(common_settings.get("default_provider", "")).strip(),
        ):
            if candidate:
                self._provider = candidate
                break
        else:
            self._provider = "anthropic"

        # ── Vector search ──────────────────────────────────────────────────────
        raw_enabled = str(common_settings.get("vector_search_enabled", "false")).lower().strip()
        self._vector_search_enabled = raw_enabled in ("true", "1", "yes")

        raw_multiphase = str(common_settings.get("vector_search_multiphase", "false")).lower().strip()
        self._vector_search_multiphase = raw_multiphase in ("true", "1", "yes")

        try:
            self._vector_search_top_k = max(3, int(common_settings.get("vector_search_top_k", 8)))
        except (ValueError, TypeError):
            self._vector_search_top_k = 8

        raw_tool_discovery = str(common_settings.get("tool_discovery_enabled", "false")).lower().strip()
        self._tool_discovery_enabled = raw_tool_discovery in ("true", "1", "yes")

        # ── Skill learning ─────────────────────────────────────────────────────
        raw_skill = str(common_settings.get("skill_learning_enabled", "false")).lower().strip()
        self._skill_learning_enabled = raw_skill in ("true", "1", "yes")
        try:
            self._skill_replay_threshold = float(
                common_settings.get("skill_replay_threshold", 0.75)
            )
        except (ValueError, TypeError):
            self._skill_replay_threshold = 0.75

        logger.info(
            "Planner settings updated: model=%s (fast=%s plan=%s) provider=%s "
            "vector_search=%s top_k=%d multiphase=%s tool_discovery=%s "
            "skill_learning=%s threshold=%.2f",
            self._model,
            self._model_fast or "(same)",
            self._model_plan or "(same)",
            self._provider,
            self._vector_search_enabled, self._vector_search_top_k,
            self._vector_search_multiphase, self._tool_discovery_enabled,
            self._skill_learning_enabled, self._skill_replay_threshold,
        )

    # ── Skill learning helpers ────────────────────────────────────────────────

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        """Token-overlap similarity between two goal strings (0.0–1.0)."""
        ta = set(a.lower().split())
        tb = set(b.lower().split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    async def _find_skill(self, goal: str) -> dict | None:
        """
        Search the learned-skill store for a past plan that matches *goal*.
        Returns the plan dict if similarity ≥ threshold, else None.
        Silently returns None on any network or store error.
        """
        if not self._skill_learning_enabled:
            return None
        try:
            # Build a safe FTS5 query from alpha-only tokens
            tokens = [w for w in goal.split() if w.isalnum() or len(w) > 2]
            q = " ".join(tokens[:12])  # cap at 12 tokens to keep query tight
            async with httpx.AsyncClient() as http:
                r = await http.get(
                    f"{self._base}/api/v1/skills/learned/search",
                    params={"q": q, "limit": 3},
                    timeout=5,
                )
                r.raise_for_status()
                results = r.json().get("results", [])
        except Exception as exc:
            logger.debug("Skill search failed (non-fatal): %s", exc)
            return None

        for hit in results:
            score = self._jaccard(goal, hit.get("goal", ""))
            if score >= self._skill_replay_threshold:
                logger.info(
                    "Skill match: score=%.2f  skill_id=%s  goal=%r",
                    score, hit.get("id"), hit.get("goal", "")[:80],
                )
                # Fire-and-forget use counter (don't block planning)
                asyncio.create_task(self._record_skill_use(hit["id"]))
                return hit.get("plan")
        return None

    async def _record_skill_use(self, skill_id: str) -> None:
        try:
            async with httpx.AsyncClient() as http:
                await http.post(
                    f"{self._base}/api/v1/skills/learned/{skill_id}/use", timeout=3
                )
        except Exception:
            pass

    def set_privacy_proxy(self, proxy: PrivacyProxy) -> None:
        """Swap the PrivacyProxy instance used by the LLM client (called once from OrchestratorClient)."""
        self._llm_client = AnthropicPrivacyClient(
            proxy_url=self._proxy_url,
            agent_id=self._agent_id,
            privacy_proxy=proxy,
        )

    async def _proxy_complete(
        self,
        messages: list[dict],
        system: str,
        max_tokens: int,
        segments: list[dict] | None = None,
        *,
        privacy_ctx: PrivacyContext | None = None,
        model: str = "",
    ) -> str:
        """
        Send a completion request to the orchestrator LLM proxy and return the response text.

        When *segments* is provided the request uses prompt_segments for cache-aware
        assembly; *system* and *messages* are ignored in that case.

        When *privacy_ctx* is provided (and the privacy proxy is enabled) the
        payload is redacted before sending and the response is restored before
        returning.

        When *model* is provided it overrides ``self._model`` for this single call.
        Pass ``self._model_fast`` for lightweight classification/decomposition tasks,
        or ``self._model_plan`` for full plan generation.
        """
        effective_model = model or self._model
        if segments:
            payload: dict = {
                "provider": self._provider,
                "model": effective_model,
                "max_tokens": max_tokens,
                "prompt_segments": segments,
            }
        else:
            payload = {
                "provider": self._provider,
                "model": effective_model,
                "messages": messages,
                "system": system,
                "max_tokens": max_tokens,
            }
        return await self._llm_client.complete(
            payload, privacy_ctx=privacy_ctx, timeout=180.0
        )

    async def _proxy_complete_structured(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
        max_tokens: int,
        *,
        privacy_ctx: PrivacyContext | None = None,
        model: str = "",
    ) -> dict:
        """
        Like _proxy_complete but returns the full response dict
        (content list + stop_reason) for tool-use responses.

        When *model* is provided it overrides ``self._model`` for this call.
        """
        payload: dict = {
            "provider": self._provider,
            "model": model or self._model,
            "messages": messages,
            "system": system,
            "tools": tools,
            "tool_choice": {"type": "auto"},
            "max_tokens": max_tokens,
        }
        return await self._llm_client.complete_structured(
            payload, privacy_ctx=privacy_ctx, timeout=180.0
        )

    async def _execute_search_capabilities(self, tool_input: dict) -> str:
        """
        Execute a search_capabilities tool call from the LLM.
        Uses the vector search index when available; falls back to
        keyword-ranked full-context search so the mode works even without
        the vector index.
        """
        keywords: list[str] = tool_input.get("keywords") or []
        limit: int = min(int(tool_input.get("limit") or 6), 12)

        if not keywords:
            return "(no keywords provided — please supply at least one keyword phrase)"

        query = " ".join(keywords)
        logger.info("Tool-discovery search: %r (limit=%d)", query[:80], limit)

        # Try vector search first
        if self._vector_search_enabled or True:   # always try; falls back gracefully
            results = await self.discover_capabilities_semantic(query, limit=limit)
            if results:
                formatted = self._format_semantic_caps(results, goal=query)
                logger.info("Tool-discovery: vector search returned %d capabilities", len(results))
                return formatted

        # Fallback: keyword-ranked selection from full capability cache
        agents = await self.discover_capabilities()
        selected = self._select_capabilities(agents, query, limit=limit)
        if not selected:
            return "(no matching capabilities found — try different keywords)"
        formatted = self._format_caps_compact(selected)
        logger.info("Tool-discovery: keyword fallback returned %d capabilities", len(selected))
        return formatted

    async def _plan_with_tool_discovery(
        self,
        goal: str,
        requester_id: str,
        user_msg: str,
        now_utc,
        user_id: str = "",
        privacy_ctx: PrivacyContext | None = None,
    ) -> tuple[dict, str]:
        """
        Agentic planning loop: the LLM calls search_capabilities to discover
        what agents are available, then calls generate_plan with the workflow.

        Returns (plan_dict, planning_mode_label).

        Compared to the full-context path this approach:
        - Sends a much smaller system prompt (~70 tokens vs ~1,800)
        - Only injects the capabilities the LLM actually asks for
        - Uses native tool_use for structured plan output (no JSON text-parsing)
        """
        tools = [_SEARCH_CAPABILITIES_TOOL, _GENERATE_PLAN_TOOL]
        messages: list[dict] = [{"role": "user", "content": user_msg}]
        searches_made: int = 0

        for turn in range(_TOOL_DISCOVERY_MAX_TURNS):
            response = await self._proxy_complete_structured(
                messages=messages,
                system=_TOOL_DISCOVERY_SYSTEM_PROMPT,
                tools=tools,
                max_tokens=self._max_tokens,
                privacy_ctx=privacy_ctx,
            )
            content: list[dict] = response.get("content", [])
            stop_reason: str = response.get("stop_reason", "end_turn")

            messages.append({"role": "assistant", "content": content})

            # Collect tool calls
            tool_results: list[dict] = []
            plan_dict: dict | None = None

            for block in content:
                if block.get("type") != "tool_use":
                    continue
                tool_name = block["name"]
                tool_id = block["id"]
                tool_input = block.get("input") or {}

                if tool_name == "search_capabilities":
                    searches_made += 1
                    result_text = await self._execute_search_capabilities(tool_input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result_text,
                    })

                elif tool_name == "generate_plan":
                    # LLM has produced the plan — extract and return immediately
                    plan_dict = tool_input
                    # Acknowledge the tool call (required by Anthropic API)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": "Plan accepted.",
                    })
                    break  # no need to process further blocks

            if plan_dict is not None:
                mode_label = f"tool-discovery(searches={searches_made})"
                logger.info(
                    "Tool-discovery planning complete: %d search(es), %d step(s)",
                    searches_made, len(plan_dict.get("steps", [])),
                )
                self._normalise_schedule_times(plan_dict, now_utc)
                return plan_dict, mode_label

            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            elif stop_reason == "end_turn":
                # LLM ended without calling generate_plan — extract JSON from text if any
                text_parts = [b.get("text", "") for b in content if b.get("type") == "text"]
                text = " ".join(text_parts).strip()
                if text:
                    logger.warning(
                        "Tool-discovery: LLM ended without generate_plan call — "
                        "attempting text JSON extraction"
                    )
                    plan_dict = self._extract_json(text)
                    self._normalise_schedule_times(plan_dict, now_utc)
                    return plan_dict, f"tool-discovery-text-fallback(searches={searches_made})"
                break

        raise ValueError(
            f"Tool-discovery loop exhausted after {_TOOL_DISCOVERY_MAX_TURNS} turns "
            f"without a generate_plan call"
        )

    async def fetch_memory_context(self, user_id: str = "") -> str:
        """
        Fetch Cortex memory to inject into the planning prompt.
        Pulls global memory (which includes user profile facts from all channels).
        Fails silently — memory is advisory context, never critical.
        """
        parts: list[str] = []
        try:
            async with httpx.AsyncClient(timeout=5.0) as http:
                # Global memory — facts/prefs visible to all agents
                try:
                    r = await http.get(f"{self._base}/api/v1/cortex/global")
                    if r.status_code == 200:
                        content = r.json().get("content", "").strip()
                        if content and "## " in content:
                            parts.append(f"=== Global Memory ===\n{content}")
                except Exception as exc:
                    logger.debug("Could not fetch global Cortex memory: %s", exc)

                # User profile facts are stored in global memory (channel-agnostic)
        except Exception as exc:
            logger.debug("Memory context fetch failed: %s", exc)

        return "\n\n".join(parts)

    async def write_memory_entries(
        self,
        user_entries: list[dict],
        user_id: str = "",
        planner_entries: list[dict] | None = None,
    ) -> None:
        """
        Write memory entries to Cortex. All writes are best-effort — failures are logged and silently ignored.
        user_entries: [{category, content}] → written to __global__ (channel-agnostic, visible to all agents)
        planner_entries: [{category, content}] → written to task-planner-agent
        """
        async with httpx.AsyncClient(timeout=5.0) as http:
            if user_entries:
                for entry in user_entries:
                    try:
                        await http.post(
                            f"{self._base}/api/v1/cortex/agents/__global__/entries",
                            json={
                                "category": entry.get("category", "Facts"),
                                "content": entry["content"],
                            },
                        )
                        logger.debug(
                            "Wrote user memory entry to global [%s]: %s",
                            entry.get("category"), entry["content"][:60],
                        )
                    except Exception as exc:
                        logger.debug("Failed to write user memory entry to global: %s", exc)

            if planner_entries:
                for entry in planner_entries:
                    try:
                        await http.post(
                            f"{self._base}/api/v1/cortex/agents/task-planner-agent/entries",
                            json={
                                "category": entry.get("category", "Patterns"),
                                "content": entry["content"],
                            },
                        )
                        logger.debug(
                            "Wrote planner memory entry [%s]: %s",
                            entry.get("category"), entry["content"][:60],
                        )
                    except Exception as exc:
                        logger.debug("Failed to write planner memory entry: %s", exc)

    async def discover_capabilities(self, force: bool = False) -> list[dict]:
        """
        Return full agent capability schemas from the orchestrator.
        Results are cached for _CAPS_CACHE_TTL_S seconds to avoid repeated
        REST calls when several plans are requested in quick succession.
        """
        now = time.monotonic()
        if not force and (now - self._caps_cache_time) < _CAPS_CACHE_TTL_S:
            logger.debug("Using cached capabilities (age=%.1fs)", now - self._caps_cache_time)
            return self._caps_cache

        try:
            async with httpx.AsyncClient(timeout=10.0) as http:
                list_resp = await http.get(f"{self._base}/api/v1/agents", params={"active": "true"})
                if list_resp.status_code != 200:
                    return self._caps_cache  # return stale on error

                agents_list = list_resp.json()
                full_agents: list[dict] = []
                for agent_summary in agents_list:
                    agent_id = agent_summary.get("id")
                    if not agent_id:
                        continue
                    detail_resp = await http.get(f"{self._base}/api/v1/agents/{agent_id}")
                    if detail_resp.status_code == 200:
                        full_agents.append(detail_resp.json())

            self._caps_cache = full_agents
            self._caps_cache_time = time.monotonic()
            # Refresh agent metadata cache for path-constraint injection
            self._agent_meta_cache = {
                a.get("name", ""): a.get("metadata") or {}
                for a in full_agents
            }
            logger.info("Capability cache refreshed: %d agent(s)", len(full_agents))
            return full_agents

        except Exception as exc:
            logger.warning("Failed to discover agents: %s", exc)

        return self._caps_cache  # return stale on network error

    # ── Vector search ──────────────────────────────────────────────────────

    async def discover_capabilities_semantic(
        self, goal: str, limit: Optional[int] = None
    ) -> list[dict]:
        """
        Call the orchestrator's TF-IDF vector index to retrieve the top-K
        capabilities most relevant to *goal*.  Falls back to an empty list on
        any error so the caller can degrade gracefully to the full-context path.
        """
        k = limit or self._vector_search_top_k
        try:
            async with httpx.AsyncClient(timeout=10.0) as http:
                resp = await http.post(
                    f"{self._base}/api/v1/discover/semantic",
                    json={"goal": goal, "limit": k},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    logger.info(
                        "Vector search returned %d/%d capabilities (top_k=%d)",
                        data.get("count", 0), data.get("total_indexed", 0), k,
                    )
                    return data.get("capabilities", [])
                logger.warning("Semantic search HTTP %d — falling back", resp.status_code)
        except Exception as exc:
            logger.warning("Semantic search failed (%s) — falling back", exc)
        return []

    def _format_semantic_caps(self, results: list[dict], goal: str = "") -> str:
        """Format vector-search results using the compact single-line style."""
        skip_agents = {"task-planner-agent", "task-executor-agent"}
        from collections import defaultdict
        by_agent: dict[str, list[dict]] = defaultdict(list)
        seen: set[tuple[str, str]] = set()

        for item in results:
            agent_name = item.get("agent_name", "unknown")
            if agent_name in skip_agents:
                continue
            cap_dict = item.get("capability") or {}
            cap_name = cap_dict.get("name") or item.get("capability_name", "")
            key = (agent_name, cap_name)
            if key in seen or not cap_name:
                continue
            seen.add(key)
            by_agent[agent_name].append((cap_name, cap_dict, item.get("score", 0.0)))

        if not by_agent:
            return "  (no relevant capabilities found)"

        lines: list[str] = []
        for agent_name, agent_caps in by_agent.items():
            lines.append(f"[{agent_name}]")
            for cap_name, cap_dict, score in agent_caps:
                desc = self._trim_desc(cap_dict.get("description") or "")
                cost = cap_dict.get("cost") or {}
                cost_usd = cost.get("estimated_cost_usd")
                cost_str = f"${cost_usd:.4f}" if cost_usd else ""
                schema = cap_dict.get("input_schema") or {}
                props = schema.get("properties") or {}
                required = list(schema.get("required") or [])
                optional = [f for f in props if f not in required]

                # Same drop-inferred micro-encoding as _format_caps_compact.
                field_parts = [_micro_field(f, props, True) for f in required]
                field_parts += [_micro_field(f, props, False) for f in optional]

                meta = f"${cost_str[1:]}, " if cost_str else ""
                line = f"  {cap_name} ({meta}rel:{score:.2f})"
                if desc:
                    line += f": {desc}"
                if field_parts:
                    line += f" → {', '.join(field_parts)}"
                lines.append(line)

                agent_meta = self._agent_meta_cache.get(agent_name, {})
                path_constraint = agent_meta.get("fs_allowed_paths") or None
                if path_constraint:
                    roots_str = ", ".join(str(p) for p in path_constraint)
                    lines.append(f"    [paths⊂{roots_str}]")
        return "\n".join(lines)

    # ── Multi-phase goal decomposition ─────────────────────────────────────

    async def decompose_goal(
        self,
        goal: str,
        memory_context: str = "",
        privacy_ctx: PrivacyContext | None = None,
    ) -> dict:
        """
        Lightweight LLM call that classifies the goal complexity and returns
        per-phase search queries for targeted capability retrieval.

        Returns {"complexity": str, "phases": [{"name", "description", "search_queries"}]}.
        Falls back to a single-phase structure on any error.
        """
        compact_memory = self._compact_memory_context(memory_context)
        memory_section = (
            f"\nMemory context:\n{compact_memory}\n" if compact_memory else ""
        )
        try:
            raw = await self._proxy_complete(
                messages=[],
                system="",
                max_tokens=512,
                segments=[
                    {
                        "name": "system_prompt",
                        "type": "system",
                        "content": _DECOMPOSE_SYSTEM_PROMPT,
                        "cacheable": True,
                    },
                    {
                        "name": "request",
                        "type": "messages",
                        "content": [{
                            "role": "user",
                            "content": f"Goal: {goal}\n{memory_section}",
                        }],
                        "cacheable": False,
                    },
                ],
                privacy_ctx=privacy_ctx,
                # Decomposition is a lightweight classification — use fast model if configured.
                model=self._model_fast or self._model,
            )
            result = self._extract_json(raw)
            phases = result.get("phases") or []
            if not phases:
                raise ValueError("Empty phases")
            logger.info(
                "Goal decomposed: complexity=%s phases=%d",
                result.get("complexity", "?"), len(phases),
            )
            return result
        except Exception as exc:
            logger.warning("Goal decomposition failed (%s) — using single-phase fallback", exc)
            return {
                "complexity": "simple",
                "phases": [{"name": "Execute", "description": goal, "search_queries": [goal]}],
            }

    async def _gather_multiphase_caps(self, goal: str, phases: list[dict]) -> str:
        """
        For each phase, run vector searches and union the results (deduplicated).
        Returns formatted capability text ready for the planning LLM prompt.
        """
        seen_keys: set[tuple[str, str]] = set()
        combined: list[dict] = []
        per_phase_k = max(4, self._vector_search_top_k // max(len(phases), 1))

        for phase in phases:
            queries = phase.get("search_queries") or [phase.get("description", goal)]
            for q in queries:
                search_query = f"{goal} {q}".strip()
                results = await self.discover_capabilities_semantic(search_query, limit=per_phase_k)
                for item in results:
                    key = (item.get("agent_name", ""), item.get("capability_name", ""))
                    if key not in seen_keys:
                        seen_keys.add(key)
                        combined.append(item)

        logger.info(
            "Multi-phase search: %d unique capabilities gathered across %d phase(s)",
            len(combined), len(phases),
        )
        return self._format_semantic_caps(combined, goal=goal)

    async def check_needs_clarification(
        self,
        goal: str,
        agents: list[dict],
        memory_context: str = "",
        session_history: list[dict] | None = None,
        privacy_ctx: PrivacyContext | None = None,
    ) -> dict:
        """
        Quick LLM call (max_tokens=512) to check if goal needs clarification.
        Returns {"needs_clarification": bool, "questions": list[str], "understood_as": str}.
        Fails open — returns {"needs_clarification": False} on any error.

        memory_context: pre-fetched Cortex content; answers already present there
        are treated as known and suppress the corresponding clarification questions.
        session_history: recent conversation turns — answers already given in
        this session also suppress redundant clarification questions.
        """
        caps_text = self._format_capabilities(agents, goal=goal, compact=True)
        compact_memory = self._compact_memory_context(memory_context)
        memory_section = (
            f"\nPersonalisation memory (confirmed facts about this user — "
            f"do NOT ask about anything already answered here):\n{compact_memory}\n"
            if compact_memory else ""
        )
        session_section = _format_session_history(session_history or [])
        try:
            raw = await self._proxy_complete(
                messages=[],
                system="",
                max_tokens=512,
                segments=[
                    {
                        "name": "system_prompt",
                        "type": "system",
                        "content": _CLARIFICATION_SYSTEM_PROMPT,
                        "cacheable": True,
                    },
                    {
                        "name": "request",
                        "type": "messages",
                        "content": [{
                            "role": "user",
                            "content": (
                                f"Goal: {goal}\n"
                                f"{memory_section}"
                                f"{session_section}\n"
                                f"Available capabilities:\n{caps_text}"
                            ),
                        }],
                        "cacheable": False,
                    },
                ],
                privacy_ctx=privacy_ctx,
                # Clarification check is a yes/no decision — use fast model if configured.
                model=self._model_fast or self._model,
            )
            logger.debug("Clarification check response: %s", raw[:300])
            return self._extract_json(raw)
        except Exception as exc:
            logger.warning("Clarification check failed (fail open): %s", exc)
            return {"needs_clarification": False}

    async def assess_planning_readiness(
        self,
        goal: str,
        conversation: list[dict],
        agents: list[dict],
        memory_context: str = "",
        privacy_ctx: PrivacyContext | None = None,
    ) -> dict:
        """
        After one or more clarification rounds, decide whether the combined
        goal + conversation supplies enough information for a fully executable plan.

        conversation — ordered list of {"role": "assistant"|"user", "content": str}
        turns representing all prior Q&A rounds for this goal.

        Returns {"ready": bool, "follow_up_questions": list[str], "understood_as": str}.
        Fails open (ready=True) so a transient LLM error never blocks execution.
        """
        caps_text = self._format_capabilities(agents, goal=goal, compact=True)
        compact_memory = self._compact_memory_context(memory_context)
        memory_section = (
            f"\nConfirmed user facts (do NOT re-ask):\n{compact_memory}\n"
            if compact_memory else ""
        )
        turns_text = "\n".join(
            f"{'Planner' if t['role'] == 'assistant' else 'User'}: {t['content']}"
            for t in conversation
        )
        try:
            raw = await self._proxy_complete(
                messages=[{"role": "user", "content": (
                    f"Goal: {goal}\n"
                    f"{memory_section}"
                    f"\nConversation so far:\n{turns_text}\n\n"
                    f"Available capabilities:\n{caps_text}"
                )}],
                system=_READINESS_SYSTEM_PROMPT,
                max_tokens=256,
                model=self._model_fast or self._model,
                privacy_ctx=privacy_ctx,
            )
            logger.debug("Planning readiness response: %s", raw[:200])
            return self._extract_json(raw)
        except Exception as exc:
            logger.warning("assess_planning_readiness failed (fail open): %s", exc)
            return {"ready": True}

    async def answer_followup_from_memory(
        self,
        question: str,
        memory_context: str = "",
        privacy_ctx: PrivacyContext | None = None,
    ) -> dict:
        """
        Try to answer a follow-up question from memory context only.
        Returns {"found": bool, "answer": str, "confidence": float}.
        """
        compact_memory = self._compact_memory_context(memory_context)
        if not compact_memory.strip():
            return {"found": False, "answer": "", "confidence": 0.0}
        try:
            raw = await self._proxy_complete(
                messages=[],
                system="",
                max_tokens=256,
                segments=[
                    {
                        "name": "system_prompt",
                        "type": "system",
                        "content": _FOLLOWUP_MEMORY_SYSTEM_PROMPT,
                        "cacheable": True,
                    },
                    {
                        "name": "request",
                        "type": "messages",
                        "content": [{
                            "role": "user",
                            "content": (
                                f"Question:\n{question}\n\n"
                                f"Memory context:\n{compact_memory}"
                            ),
                        }],
                        "cacheable": False,
                    },
                ],
                privacy_ctx=privacy_ctx,
                # Memory lookup is a simple retrieval task — use fast model if configured.
                model=self._model_fast or self._model,
            )
            parsed = self._extract_json(raw)
            found = bool(parsed.get("found"))
            answer = str(parsed.get("answer", "")).strip()
            try:
                confidence = float(parsed.get("confidence", 0.0))
            except Exception:
                confidence = 0.0
            if not found or not answer:
                return {"found": False, "answer": "", "confidence": max(0.0, confidence)}
            return {"found": True, "answer": answer, "confidence": max(0.0, min(1.0, confidence))}
        except Exception as exc:
            logger.warning("Follow-up memory answer failed (fail closed): %s", exc)
            return {"found": False, "answer": "", "confidence": 0.0}

    def _flatten_capabilities(self, agents: list[dict]) -> list[dict]:
        caps: list[dict] = []
        skip_agents = {"task-planner-agent", "task-executor-agent"}
        for agent in agents:
            agent_name = agent.get("name", "unknown")
            if agent_name in skip_agents:
                continue
            if agent.get("disabled"):
                continue
            agent_metadata = agent.get("metadata") or {}
            for cap in agent.get("capabilities", []):
                if isinstance(cap, dict):
                    cap_name = cap.get("name", "")
                    cap_desc = cap.get("description", "")
                    schema = cap.get("input_schema", {}) or {}
                    required_fields = list(schema.get("required", []) or [])
                    properties = dict(schema.get("properties", {}) or {})
                    tags = list(cap.get("tags", []) or [])
                    cost = cap.get("cost", {}) or {}
                    cost_type = cost.get("type", "free")
                    cost_usd = cost.get("estimated_cost_usd")
                    cost_note = cost.get("notes", "")
                else:
                    cap_name = str(cap)
                    cap_desc = ""
                    required_fields = []
                    properties = {}
                    tags = []
                    cost_type = "free"
                    cost_usd = None
                    cost_note = ""

                if not cap_name:
                    continue
                optional_fields = [f for f in properties.keys() if f not in required_fields]
                # Carry path constraint from agent metadata (set by filesystem-agent)
                path_constraint = agent_metadata.get("fs_allowed_paths") or None
                caps.append({
                    "agent_name": agent_name,
                    "capability_name": cap_name,
                    "description": cap_desc,
                    "required_fields": required_fields,
                    "optional_fields": optional_fields,
                    "tags": tags,
                    "cost_type": cost_type,
                    "cost_usd": cost_usd,
                    "cost_note": cost_note,
                    "properties": properties,
                    "path_constraint": path_constraint,
                })
        return caps

    @staticmethod
    def _goal_tokens(goal: str) -> set[str]:
        return {t for t in re.split(r"[^a-z0-9]+", (goal or "").lower()) if len(t) >= 3}

    def _cap_score(self, cap: dict, goal_tokens: set[str]) -> int:
        text = " ".join([
            cap.get("capability_name", ""),
            cap.get("agent_name", ""),
            cap.get("description", ""),
            " ".join(cap.get("tags", [])),
            " ".join(cap.get("required_fields", [])),
            " ".join(cap.get("optional_fields", [])),
        ]).lower()
        score = sum(1 for tok in goal_tokens if tok in text)
        name = cap.get("capability_name", "")
        if name in {"send_slack_message", "send_email", "send_whatsapp_message", "send_telegram_message"}:
            score += 1  # keep at least one low-cost completion channel near top
        if name == "schedule_task" and {"schedule", "remind", "later", "tomorrow", "week"} & goal_tokens:
            score += 3
        return score

    def _select_capabilities(self, agents: list[dict], goal: str, limit: int = _COMPACT_CAP_LIMIT) -> list[dict]:
        return self._rank_and_select(self._flatten_capabilities(agents), goal, limit)

    def _rank_and_select(self, all_caps: list[dict], goal: str, limit: int = _COMPACT_CAP_LIMIT) -> list[dict]:
        """Rank an already-flattened capability list by goal relevance and return
        the top *limit*, always pinning a messaging channel (and the scheduler
        when the goal is time-based) so plans can still complete."""
        if len(all_caps) <= limit:
            return all_caps

        goal_tokens = self._goal_tokens(goal)
        ranked = sorted(
            all_caps,
            key=lambda c: (
                self._cap_score(c, goal_tokens),
                -len(c.get("required_fields", [])),
            ),
            reverse=True,
        )
        selected: list[dict] = []
        used: set[tuple[str, str]] = set()

        def _add(cap: dict) -> None:
            key = (cap["agent_name"], cap["capability_name"])
            if key in used:
                return
            selected.append(cap)
            used.add(key)

        # Ensure final-step messaging capability stays available.
        for cap in ranked:
            if cap["capability_name"] in {"send_slack_message", "send_email", "send_whatsapp_message", "send_telegram_message"}:
                _add(cap)
                break

        # If goal sounds time-based, keep scheduler capability visible.
        if {"schedule", "remind", "later", "tomorrow", "week"} & goal_tokens:
            for cap in ranked:
                if cap["capability_name"] == "schedule_task":
                    _add(cap)
                    break

        # Pin the top-ranked capability from any agent explicitly named in the goal.
        # This prevents e.g. "use heygen to create a video" from pushing create_video
        # into the names-only tail because many heygen caps split the relevance score.
        # Match by checking whether any goal token is a prefix-word of the agent name
        # (e.g. goal token "heygen" matches agent "heygen-agent").
        for cap in ranked:
            agent_words = set(re.split(r"[^a-z0-9]+", cap["agent_name"].lower()))
            if agent_words & goal_tokens:
                _add(cap)
                break

        for cap in ranked:
            if len(selected) >= limit:
                break
            _add(cap)
        return selected[:limit]

    def _format_capabilities(self, agents: list[dict], goal: str = "", compact: bool = False) -> str:
        all_caps = self._flatten_capabilities(agents)
        if not all_caps:
            return "  (no agents currently available)"
        if compact:
            return self._format_caps_tiered(all_caps, goal)
        return self._format_caps_verbose(all_caps)

    def _format_caps_tiered(self, all_caps: list[dict], goal: str, detail_limit: int = _COMPACT_CAP_LIMIT) -> str:
        """Progressive disclosure: keep **every** capability visible, but spend
        full-schema tokens only on the goal-relevant ones.  The remainder are
        listed name-only (grouped by agent) so the planner still knows they
        exist and can surface them via tool-discovery / re-ranking, without
        paying full field-schema cost for each.

        This replaces the old hard top-N drop — nothing is removed from the
        catalogue, the long tail is just compressed."""
        if len(all_caps) <= detail_limit:
            return self._format_caps_compact(all_caps)

        detailed = self._rank_and_select(all_caps, goal, detail_limit)
        detailed_keys = {(c["agent_name"], c["capability_name"]) for c in detailed}
        remainder = [
            c for c in all_caps
            if (c["agent_name"], c["capability_name"]) not in detailed_keys
        ]

        parts = [self._format_caps_compact(detailed)]
        if remainder:
            from collections import defaultdict
            by_agent: dict[str, list[str]] = defaultdict(list)
            for c in remainder:
                by_agent[c["agent_name"]].append(c["capability_name"])
            parts.append(
                "\nMore capabilities available (names only — request full schema if relevant):"
            )
            for agent_name, names in by_agent.items():
                parts.append(f"  [{agent_name}] {', '.join(names)}")
        return "\n".join(parts)

    @staticmethod
    def _cost_token(cap: dict) -> str:
        """Compact cost string, or '' for free (free is the default — omit it)."""
        cost_usd = cap.get("cost_usd")
        return f"${cost_usd:.4f}" if cost_usd else ""

    @staticmethod
    def _trim_desc(desc: str, max_chars: int = 90) -> str:
        """Strip boilerplate openers ('Use this to…') and cap length."""
        desc = _DESC_PREFIX_RE.sub("", (desc or "").strip())
        if desc:
            desc = desc[0].upper() + desc[1:]
        return desc[:max_chars]

    def _format_caps_compact(self, caps: list[dict]) -> str:
        """
        Dense single-line-per-capability format grouped by agent.

        Token-reduction strategies (lossless for planning):
          • Agent-level factoring — a cost or path-constraint shared by every
            capability in an agent is hoisted to the agent header instead of
            being repeated on each line.
          • Drop-inferred — `free` cost is omitted entirely, string field types
            are omitted (string is the modal type), and optional fields collapse
            to a `+Nopt` count rather than being listed.
          • Required fields carry a `!` suffix; optional fields listed by name without `!`.
          • Non-string types carry a 1-char code (i=int, b=bool, a=array, o=object).

        Example:
          [filesystem-agent | paths⊂/workspace]
            read_file: Read a file's contents → path!
            write_file: Create or overwrite a file → path! content! encoding
        """
        from collections import defaultdict
        by_agent: dict[str, list[dict]] = defaultdict(list)
        for cap in caps:
            by_agent[cap["agent_name"]].append(cap)

        lines: list[str] = []
        for agent_name, agent_caps in by_agent.items():
            # ── Strategy 2: hoist uniform cost / path-constraint to header ──
            costs = {self._cost_token(c) for c in agent_caps}
            shared_cost = costs.pop() if len(costs) == 1 else None

            path_sets = {tuple(c.get("path_constraint") or ()) for c in agent_caps}
            shared_paths = path_sets.pop() if len(path_sets) == 1 else ()

            header = f"[{agent_name}"
            if shared_cost:
                header += f" | {shared_cost}"
            if shared_paths:
                header += f" | paths⊂{', '.join(str(p) for p in shared_paths)}"
            header += "]"
            lines.append(header)

            for cap in agent_caps:
                cap_name = cap["capability_name"]
                desc = self._trim_desc(cap.get("description") or "")
                properties = cap.get("properties", {})
                required = cap.get("required_fields", [])
                optional = cap.get("optional_fields", [])

                # Required fields carry `!`; optional fields listed by name (no `!`).
                field_parts = [_micro_field(f, properties, True) for f in required]
                field_parts += [_micro_field(f, properties, False) for f in optional]

                line = f"  {cap_name}"
                # Per-cap cost only when not already factored to the header.
                if shared_cost is None:
                    tok = self._cost_token(cap)
                    if tok:
                        line += f" ({tok})"
                if desc:
                    line += f": {desc}"
                if field_parts:
                    line += f" → {', '.join(field_parts)}"
                lines.append(line)

                # Per-cap path constraint only when not factored to the header.
                if not shared_paths:
                    pc = cap.get("path_constraint")
                    if pc:
                        roots_str = ", ".join(str(p) for p in pc)
                        lines.append(f"    [paths⊂{roots_str}]")
        return "\n".join(lines)

    def _format_caps_verbose(self, caps: list[dict]) -> str:
        """Full multi-line format with all fields and descriptions. Used for non-compact mode."""
        lines: list[str] = []
        for cap in caps:
            cap_name = cap["capability_name"]
            agent_name = cap["agent_name"]
            cap_desc = cap.get("description", "")
            required_fields = cap.get("required_fields", [])
            optional_fields = cap.get("optional_fields", [])
            properties = cap.get("properties", {})
            cost_type = cap.get("cost_type", "free")
            cost_usd = cap.get("cost_usd")
            cost_note = cap.get("cost_note", "")

            lines.append(f"  - {cap_name} (agent: {agent_name})")
            if cap_desc:
                lines.append(f"    Description: {cap_desc}")
            if cost_type == "free" or cost_usd is None:
                lines.append("    Cost: free")
            else:
                note = f" ({cost_note})" if cost_note else ""
                lines.append(f"    Cost: ${cost_usd:.4f}/call{note}")
            path_constraint = cap.get("path_constraint")
            if path_constraint:
                roots_str = ", ".join(str(p) for p in path_constraint)
                lines.append(f"    Path constraint: ALL file paths MUST be within: {roots_str}")
            if properties:
                lines.append("    Input fields:")
                for field_name in list(required_fields) + list(optional_fields):
                    field_info = properties.get(field_name, {})
                    marker = " [REQUIRED]" if field_name in required_fields else " [optional]"
                    field_type = field_info.get("type", "any")
                    field_desc = field_info.get("description", "")
                    lines.append(f"      - {field_name} ({field_type}){marker}: {field_desc}")
        return "\n".join(lines)

    def _compact_memory_context(self, memory_context: str, max_chars: int = _MAX_MEMORY_CONTEXT_CHARS) -> str:
        text = (memory_context or "").strip()
        if not text:
            return ""
        if len(text) <= max_chars:
            return text
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        keep: list[str] = []
        used = 0
        for line in lines:
            if used + len(line) + 1 > max_chars:
                break
            keep.append(line)
            used += len(line) + 1
        if not keep:
            return text[: max_chars - 16] + "... (truncated)"
        return "\n".join(keep) + "\n... (truncated)"

    @staticmethod
    def _strip_thinking_blocks(text: str) -> str:
        """Remove <think>…</think> reasoning blocks (Qwen3, DeepSeek-R1, etc.)."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

    def _extract_json(self, text: str) -> dict:
        # Strip Qwen3-style <think>…</think> reasoning blocks.  These often
        # contain partial JSON / braces which break the greedy \{.*\} regex.
        text = self._strip_thinking_blocks(text)

        # 0. Try the whole text as bare JSON first — handles models that return
        #    a clean JSON block with no surrounding prose.  This is also the only
        #    safe path when the response is a nested structure: the reverse-scan
        #    below would otherwise return an inner object (e.g. a step's
        #    input_data dict) instead of the outer plan object.
        stripped = text.strip()
        if stripped.startswith("{"):
            try:
                result = json.loads(stripped)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        # 1. Fenced code block  ```json … ```
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            return json.loads(m.group(1))

        # 2. Forward scan: find the first `{` and walk to its matching `}`.
        #    Most prose-prefix responses look like "Here is the plan:\n{...}"
        #    where the first `{` IS the outer plan object.
        candidates = list(re.finditer(r"\{", text))
        for start_match in candidates:
            start = start_match.start()
            depth = 0
            for i, ch in enumerate(text[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i + 1]
                        try:
                            result = json.loads(candidate)
                            if isinstance(result, dict):
                                return result
                        except json.JSONDecodeError:
                            break

        # 3. Reverse scan: last-resort for responses where the model sandwiched
        #    the JSON between two blocks of prose (start AND end have prose).
        for start_match in reversed(candidates):
            start = start_match.start()
            depth = 0
            for i, ch in enumerate(text[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i + 1]
                        try:
                            result = json.loads(candidate)
                            if isinstance(result, dict) and (
                                "steps" in result or "title" in result
                                or "needs_clarification" in result
                                or "ready" in result or "found" in result
                            ):
                                return result
                        except json.JSONDecodeError:
                            break

        raise ValueError(f"No JSON found in LLM response: {text[:300]!r}")

    @staticmethod
    def is_sensitive_memory_entry(entry: dict) -> bool:
        content = str((entry or {}).get("content", "")).lower()
        if not content:
            return False
        if any(k in content for k in _SENSITIVE_KEYWORDS):
            return True
        # Simple PII patterns
        if re.search(r"\b\d{3}-\d{2}-\d{4}\b", content):   # US SSN
            return True
        if re.search(r"\b(?:\d[ -]*?){13,19}\b", content): # card-like long digit sequence
            return True
        if re.search(r"\b\d{10,}\b", content):             # long numeric identifiers
            return True
        return False

    @staticmethod
    def _parse_iso_datetime(raw: str) -> datetime | None:
        s = (raw or "").strip()
        if not s:
            return None
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def _iso_utc(dt: datetime) -> str:
        return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds")

    def _normalise_schedule_times(self, plan_dict: dict, now_utc: datetime) -> None:
        for step in plan_dict.get("steps", []):
            if step.get("capability") != "schedule_task":
                continue
            input_data = step.get("input_data")
            if not isinstance(input_data, dict):
                continue
            scheduled_raw = input_data.get("scheduled_at")
            if not isinstance(scheduled_raw, str):
                continue
            parsed = self._parse_iso_datetime(scheduled_raw)
            if parsed is None or parsed > now_utc:
                continue
            corrected = parsed
            for candidate_year in (now_utc.year, now_utc.year + 1):
                try:
                    candidate = corrected.replace(year=candidate_year)
                except ValueError:
                    continue
                if candidate > now_utc:
                    corrected = candidate
                    break
            if corrected <= now_utc:
                corrected = now_utc + timedelta(minutes=5)
            input_data["scheduled_at"] = self._iso_utc(corrected)
            logger.info(
                "Adjusted past scheduled_at: %s -> %s",
                scheduled_raw,
                input_data["scheduled_at"],
            )

    async def refine_plan(
        self,
        plan_dict: dict,
        instruction: str,
        privacy_ctx: PrivacyContext | None = None,
    ) -> dict:
        """
        Revise an existing plan according to *instruction* with a single LLM call.
        Returns a normalised plan dict {title, description, goal, steps, total_steps}
        with fresh step_ids. Nothing is persisted or executed.
        """
        now_utc = datetime.now(timezone.utc)
        agents = await self.discover_capabilities()
        goal = str(plan_dict.get("goal", "") or plan_dict.get("title", ""))
        caps_text = self._format_capabilities(
            agents, goal=f"{goal} {instruction}", compact=True
        )

        current_plan_json = json.dumps(
            {
                "title":       plan_dict.get("title", ""),
                "description": plan_dict.get("description", ""),
                "goal":        goal,
                "steps": [
                    {
                        "name":           s.get("name", ""),
                        "goal":           s.get("goal", ""),
                        "description":    s.get("description", ""),
                        "capability":     s.get("capability", ""),
                        "input_data":     s.get("input_data", {}),
                        "depends_on":     s.get("depends_on", []),
                        "execution_mode": s.get("execution_mode", "strict"),
                    }
                    for s in plan_dict.get("steps", [])
                    if isinstance(s, dict)
                ],
            },
            indent=2,
        )

        user_msg = (
            f"CURRENT_UTC: {self._iso_utc(now_utc)}\n\n"
            f"Here is an existing workflow plan:\n```json\n{current_plan_json}\n```\n\n"
            f"Revise it according to this instruction:\n{instruction}\n\n"
            "Return the COMPLETE revised plan in the same JSON format you use for "
            "new plans (title, description, steps). Keep steps that the instruction "
            "does not affect unchanged. Only use capabilities from the available "
            "capabilities list. Do not add memory_entries."
        )

        logger.info(
            "Refining plan: title=%r  instruction=%r",
            plan_dict.get("title", "")[:60], instruction[:80],
        )
        raw_text: str = await self._proxy_complete(
            messages=[],
            system="",
            max_tokens=self._max_tokens,
            segments=[
                {
                    "name": "system_prompt",
                    "type": "system",
                    "content": self._plan_system_prompt,
                    "cacheable": True,
                },
                {
                    "name": "capabilities",
                    "type": "context",
                    "content": f"Available agent capabilities:\n{caps_text}",
                    "cacheable": True,
                },
                {
                    "name": "conversation",
                    "type": "messages",
                    "content": [{"role": "user", "content": user_msg}],
                    "cacheable": False,
                },
            ],
            privacy_ctx=privacy_ctx,
            model=self._model_plan or self._model,
        )

        revised = self._extract_json(raw_text)
        self._normalise_schedule_times(revised, now_utc)
        revised.pop("memory_entries", None)

        steps: list[dict] = []
        for i, step_d in enumerate(revised.get("steps", []), 1):
            if not isinstance(step_d, dict):
                continue
            raw_mode = str(step_d.get("execution_mode", "")).strip().lower()
            steps.append(
                WorkflowStep.create(
                    order=i,
                    name=step_d.get("name", f"Step {i}"),
                    goal=step_d.get("goal", step_d.get("description", "")),
                    description=step_d.get("description", ""),
                    capability=step_d.get("capability", ""),
                    input_data=step_d.get("input_data", {}),
                    depends_on=step_d.get("depends_on") or [],
                    execution_mode=raw_mode if raw_mode in ("strict", "emergent") else "strict",
                ).to_dict()
            )

        return {
            "title":       revised.get("title", plan_dict.get("title", "")),
            "description": revised.get("description", plan_dict.get("description", "")),
            "goal":        revised.get("goal", goal),
            "steps":       steps,
            "total_steps": len(steps),
        }

    async def plan(
        self,
        goal: str,
        requester_id: str,
        channel_id: str = "",
        thread_id: str = "",
        user_id: str = "",
        delivery_channel: str = "",
        persona: str = "",
        summary_format: str = "",
        source: str = "",
        memory_context: str = "",
        clarification_message: str = "",
        clarification_answers: str = "",
        clarification_history: list[dict] | None = None,
        session_history: list[dict] | None = None,
        privacy_ctx: PrivacyContext | None = None,
    ) -> WorkflowPlan:
        """
        Generate a WorkflowPlan for *goal* using a single Anthropic API call.
        Capabilities are fetched from the orchestrator (cached, TTL 60 s).

        channel_id / thread_id identify where the completion response must be
        sent (the same channel and conversation thread as the original request).
        user_id is used to fetch/write Cortex user memory for personalisation.
        memory_context: pass a pre-fetched value to skip an extra Cortex fetch
        (e.g. when the caller already fetched it for the clarification check).
        clarification_message / clarification_answers: single-round Q&A (legacy).
        clarification_history: ordered list of {"role": "assistant"|"user",
            "content": str} turns covering all clarification rounds. When provided
            it supersedes clarification_message/answers and builds a complete
            multi-turn conversation so the LLM sees the full back-and-forth.
        """
        # ── Skill replay: check for a matching past plan before any LLM call ──
        # Only attempted when skill_learning_enabled=true and there are no
        # clarification answers (clarification implies a novel or ambiguous goal).
        if self._skill_learning_enabled and not clarification_answers:
            replayed = await self._find_skill(goal)
            if replayed:
                try:
                    steps = [
                        WorkflowStep.create(
                            order=s.get("order", i),
                            name=s.get("name", ""),
                            goal=s.get("description", ""),
                            description=s.get("description", ""),
                            capability=s["capability"],
                            input_data=s.get("input_data", {}),
                        )
                        for i, s in enumerate(
                            sorted(replayed.get("steps", []), key=lambda x: x.get("order", 0))
                        )
                    ]
                    plan = WorkflowPlan(
                        task_id=str(__import__("uuid").uuid4()),
                        title=replayed.get("title", ""),
                        description=replayed.get("description", ""),
                        goal=goal,
                        steps=steps,
                        requester_id=requester_id,
                        memory_entries=replayed.get("memory_entries", []),
                        planning_mode="skill-replay",
                    )
                    logger.info("Replaying learned skill for goal: %r", goal[:80])
                    return plan
                except Exception as exc:
                    logger.warning(
                        "Skill replay failed (%s) — falling back to LLM planning", exc
                    )

        # ── Use pre-fetched memory context or fetch now ───────────────────────
        if not memory_context:
            memory_context = await self.fetch_memory_context(user_id=user_id)

        now_utc = datetime.now(timezone.utc)

        # ── Capability retrieval: tool-discovery, vector search, or full-context ─
        planning_mode: str

        # Build the common parts of the user message (shared by all planning modes)
        # Channel/thread routing is handled by the orchestration layer after
        # the workflow completes — keep it OUT of the LLM planning context.
        reply_context_lines: list[str] = [f"REQUESTER_ID: {requester_id}"]
        if source:
            reply_context_lines.append(f"SOURCE: {source}")
        if user_id:
            reply_context_lines.append(f"USER_ID: {user_id}")
        if delivery_channel:
            reply_context_lines.append(f"DELIVERY_CHANNEL: {delivery_channel}")
        if persona:
            reply_context_lines.append(f"PERSONA: {persona}")
        if summary_format:
            reply_context_lines.append(f"SUMMARY_FORMAT: {summary_format}")
        reply_context = "\n".join(reply_context_lines)

        compact_memory = self._compact_memory_context(memory_context)
        memory_section = (
            f"\nPersonalisation context from Cortex memory:\n{compact_memory}\n"
            if compact_memory else ""
        )
        session_section = _format_session_history(session_history or [])

        user_msg = (
            f"CURRENT_UTC: {self._iso_utc(now_utc)}\n\n"
            f"Request context:\n{reply_context}\n"
            f"{memory_section}"
            f"{session_section}\n"
            f"Goal: {goal}\n\n"
            "Create a workflow plan to accomplish this goal. "
            "Include a clear goal for each step. "
            "Do NOT add a final messaging/notification step — "
            "the orchestration layer delivers the result to the user automatically. "
            "If the goal or memory context reveals stable user preferences or facts "
            "worth remembering, include them in the optional memory_entries array."
        )

        if self._tool_discovery_enabled:
            # ── Tool-discovery mode: LLM drives capability lookup ─────────────
            # The LLM calls search_capabilities (backed by vector/keyword index)
            # then calls generate_plan with the workflow as a structured tool call.
            # Saves ~40% tokens vs full-context on novel goals; more accurate
            # because the LLM expresses exactly what capabilities it needs.
            logger.info(
                "Planning workflow: goal=%r  mode=tool-discovery  memory=%s",
                goal[:80], "yes" if memory_context else "none",
            )
            if clarification_history:
                # Multi-round: flatten the full conversation into the user message
                turns_text = "\n".join(
                    f"{'Planner' if t['role'] == 'assistant' else 'User'}: {t['content']}"
                    for t in clarification_history
                )
                user_msg += f"\n\nClarification dialogue:\n{turns_text}"
            elif clarification_message and clarification_answers:
                user_msg += (
                    f"\n\nClarification Q: {clarification_message}\n"
                    f"User answered: {clarification_answers}"
                )
            elif clarification_answers:
                user_msg += (
                    f"\n\nThe user provided these answers to clarification questions:\n"
                    f"{clarification_answers}"
                )
            plan_dict, planning_mode = await self._plan_with_tool_discovery(
                goal=goal,
                requester_id=requester_id,
                user_msg=user_msg,
                now_utc=now_utc,
                user_id=user_id,
                privacy_ctx=privacy_ctx,
            )
        else:
            # ── Pre-filter modes: fetch caps up-front, single LLM call ────────
            if self._vector_search_enabled:
                if self._vector_search_multiphase:
                    decomposition = await self.decompose_goal(
                        goal, memory_context=memory_context, privacy_ctx=privacy_ctx
                    )
                    phases = decomposition.get("phases", [])
                    complexity = decomposition.get("complexity", "simple")
                    caps_text = await self._gather_multiphase_caps(goal, phases)
                    planning_mode = f"vector-multiphase({complexity},{len(phases)}ph)"
                else:
                    vec_results = await self.discover_capabilities_semantic(goal)
                    if vec_results:
                        caps_text = self._format_semantic_caps(vec_results, goal=goal)
                        planning_mode = f"vector-single(top{self._vector_search_top_k})"
                    else:
                        logger.warning("Vector search returned no results — falling back to full context")
                        agents = await self.discover_capabilities()
                        caps_text = self._format_capabilities(agents, goal=goal, compact=True)
                        planning_mode = "full-context(vector-fallback)"
            else:
                agents = await self.discover_capabilities()
                caps_text = self._format_capabilities(agents, goal=goal, compact=True)
                planning_mode = "full-context"

            logger.info(
                "Planning workflow: goal=%r  mode=%s  memory=%s",
                goal[:80], planning_mode, "yes" if memory_context else "none",
            )

            # ── Single LLM call with pre-fetched capabilities ─────────────────
            if clarification_history:
                # Multi-round conversational path: prepend the initial goal message
                # and append all clarification turns so the LLM sees the full dialogue.
                conv_messages = [{"role": "user", "content": user_msg}] + clarification_history
                logger.info(
                    "Planning with multi-round clarification history (%d turns)",
                    len(clarification_history),
                )
            elif clarification_message and clarification_answers:
                conv_messages = [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": clarification_message},
                    {"role": "user", "content": clarification_answers},
                ]
                logger.info("Planning with single-round clarification history (3 turns)")
            elif clarification_answers:
                user_msg += (
                    f"\n\nThe user provided these answers to clarification questions:\n"
                    f"{clarification_answers}"
                )
                conv_messages = [{"role": "user", "content": user_msg}]
                logger.info("Planning with clarification answers injected into context")
            else:
                conv_messages = [{"role": "user", "content": user_msg}]

            raw_text: str = await self._proxy_complete(
                messages=[],
                system="",
                max_tokens=self._max_tokens,
                segments=[
                    {
                        "name": "system_prompt",
                        "type": "system",
                        "content": self._plan_system_prompt,
                        "cacheable": True,
                    },
                    {
                        "name": "capabilities",
                        "type": "context",
                        "content": f"Available agent capabilities:\n{caps_text}",
                        "cacheable": True,
                    },
                    {
                        "name": "conversation",
                        "type": "messages",
                        "content": conv_messages,
                        "cacheable": False,
                    },
                ],
                privacy_ctx=privacy_ctx,
                # Plan generation is the most reasoning-heavy task — use the
                # powerful plan model if configured, else fall back to _model.
                model=self._model_plan or self._model,
            )
            logger.debug("LLM response: %s", raw_text[:500])

            plan_dict = self._extract_json(raw_text)
            self._normalise_schedule_times(plan_dict, now_utc)

        # ── Extract and persist memory entries (fire-and-forget) ─────────────
        raw_memory_entries: list[dict] = plan_dict.pop("memory_entries", []) or []
        user_entries = [
            e for e in raw_memory_entries
            if isinstance(e, dict) and e.get("content")
        ][:3]

        # Always record the goal as a planning pattern in planner's own namespace
        goal_summary = goal[:120].replace("\n", " ")
        planner_entries: list[dict] = [
            {
                "category": "Patterns",
                "content": (
                    f"Planned '{plan_dict.get('title', 'workflow')}' "
                    f"({len(plan_dict.get('steps', []))} steps) for goal: {goal_summary}"
                ),
            }
        ]

        if planner_entries:
            import asyncio as _asyncio
            _asyncio.create_task(
                self.write_memory_entries(
                    user_entries=[],
                    user_id=user_id,
                    planner_entries=planner_entries,
                ),
                name="cortex-write-plan",
            )
        if user_entries:
            logger.info(
                "Extracted %d candidate user memory entr%s for consented storage",
                len(user_entries),
                "ies" if len(user_entries) != 1 else "y",
            )

        steps: list[WorkflowStep] = []
        for i, step_d in enumerate(plan_dict.get("steps", []), 1):
            raw_confidence = step_d.get("confidence", 1.0)
            try:
                confidence = float(raw_confidence)
                confidence = max(0.0, min(1.0, confidence))
            except (TypeError, ValueError):
                confidence = 1.0
            raw_mode = str(step_d.get("execution_mode", "")).strip().lower()
            execution_mode = raw_mode if raw_mode in ("strict", "emergent") else "strict"
            steps.append(
                WorkflowStep.create(
                    order=i,
                    name=step_d.get("name", f"Step {i}"),
                    goal=step_d.get("goal", step_d.get("description", "")),
                    description=step_d.get("description", ""),
                    capability=step_d.get("capability", ""),
                    input_data=step_d.get("input_data", {}),
                    confidence=confidence,
                    execution_mode=execution_mode,
                )
            )

        plan = WorkflowPlan.create(
            title=plan_dict.get("title", "Untitled Workflow"),
            description=plan_dict.get("description", ""),
            goal=goal,
            steps=steps,
            requester_id=requester_id,
        )
        plan.memory_entries = user_entries
        plan.planning_mode = planning_mode  # stored for dashboard/logging inspection
        logger.info(
            "Plan created: task_id=%s  title=%r  steps=%d  mode=%s\n%s",
            plan.task_id, plan.title, len(steps), planning_mode,
            json.dumps(plan.to_dict(), indent=2),
        )
        return plan
