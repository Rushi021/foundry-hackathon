# foundry-hackathon

A complete AI consultant — from your co-founder, to your company's engineers, to your personal assistant.

Solo founders and builders get specialist agents that share context and work with real opinions (no sugarcoating).

## Phase 1 — routed agents, memory boundaries, delegation

```text
user → prepare context → Jev decision router ─┬─► Direct Assistant ─(escalate)─► Co-Founder / COO / Research
                                               ├─► Co-Founder ──delegate──► Research · COO · Engineering · Marketing
                                               ├─► COO ─────────delegate──► Research · Engineering · Marketing (· Co-Founder)
                                               ├─► Research / Engineering / Marketing
                                               └─► context resolution ─► re-route once, or ask a clarifying question
```

| Piece | Tech |
|-------|------|
| Orchestration | LangGraph (`prepare_context → route → resolve? → execute → finalize`) |
| Routing | [Jev by TypeSafe AI](https://docs.typesafe.ai/api/system-one) — one call answers destination, complexity, tool, memory and clarification questions with probabilities. LLM router is a flagged fallback only. |
| Agents | Registry-driven (`agents/registry.py`); one runtime loop for all agents (`runtime.py`) |
| Memory | Private (user + Co-Founder only) and Shared, file-backed, keyword retrieval, enforced in code (`memory.py`) |
| Tools | Registry (`tools/registry.py`): `web_search` (Perplexity) connected; Gmail, Calendar, GitHub registered but not connected, approval-gated |
| API | FastAPI |
| LLM | Provider-agnostic (`provider:model`, default OpenAI) |
| Logs | Terminal step stream, per-run metrics, Supabase `workflow_runs` (optional), LangSmith |

Key behaviors:

- **Simple stays simple:** a general question costs one router call + one model call.
- **Memory is conditional:** the router's memory probability decides preloading; agents can
  `recall_memory` later, but only from scopes they're authorized for.
- **Private Memory never leaves the Co-Founder:** the gateway denies every other agent, and
  delegation briefs/results that reproduce private content are blocked.
- **Honest status:** `completed | partial | needs_clarification | awaiting_approval | failed`,
  derived from tool and subtask outcomes, not from the agent's claim.
- **Bounded delegation:** allowlists per agent, no cycles, max depth 3, per-run dedupe of
  identical searches and delegations.

Add an agent: register an `AgentSpec` in `agents/registry.py` (and a persona in
`agents/prompts.py`). Add a tool: register a `ToolSpec` in `tools/registry.py` and list it
on the agents allowed to use it. Neither requires graph changes.

See [SPEC.md](./SPEC.md) for the full contract.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # TYPESAFE_API_KEY (Jev router), optional PERPLEXITY_API_KEY / Supabase / LangSmith
```

Without `TYPESAFE_API_KEY` the workflow still runs, using the LLM router fallback; every run
prints `router.fallback` and records `metrics.router_fallback_reason`.

### LLM billing (ChatGPT plan or API key)

Prefer **Sign in with ChatGPT** so eligible Responses API calls use your ChatGPT
subscription instead of a separate API key ([OpenAI docs](https://developers.openai.com/siwc/token-sharing-open-source)):

```bash
foundry login          # browser OAuth + plan-usage consent
foundry auth           # confirm plan_usage: true
foundry models         # slugs available to your account
# optional: set CATEGORIZE_LLM / FOUNDER_LLM to those slugs in .env
```

Credentials live in `~/.config/foundry/` (mode `0600`). Manage connected apps and
limits in **ChatGPT → Settings → Usage**.

Fallback: set `OPENAI_API_KEY`, or force API billing with `FOUNDRY_LLM_BILLING=api_key`.

## Run

Live per-step status prints to the terminal (CLI and API server) so you can verify
each node before Supabase logging is finalized:

```text
[foundry:abcd1234] ▶ run.start | prompt=...
[foundry:abcd1234] ▶ context.prepared | session_id=... | recent_turns=2 | last_agent=research
[foundry:abcd1234] ▶ router.decision | provider=jev | destination=cofounder | confidence=0.93 | memory_p=0.81 | next=cofounder
[foundry:abcd1234] ▶ agent.start | agent=cofounder | depth=0
[foundry:abcd1234] ▶ memory.read | agent=cofounder | scope=private | hits=1 | via=preload
[foundry:abcd1234] ▶ delegation.start | frm=cofounder | to=research | objective=...
[foundry:abcd1234] ▶ delegation.result | frm=cofounder | to=research | status=completed
[foundry:abcd1234] ▶ run.complete | agent=cofounder | status=completed | model_calls=3 | decision_calls=1 | ...
```

```bash
# API (primary) — watch the uvicorn terminal for step stream
uvicorn foundry.api:app --reload --port 8000

curl -s http://localhost:8000/v1/workflows/run \
  -H 'content-type: application/json' \
  -d '{"prompt":"Should I build an AI co-founder for solo founders?"}'
# → { session_id, status, route, decision, response, task, metrics, ... }
# pass "session_id" on the next request to continue the conversation

# Memory (you own both scopes; agents only reach it through the gateway)
curl -s localhost:8000/v1/memory -H 'content-type: application/json' \
  -d '{"scope":"private","content":"I want to bootstrap, no VC."}'
curl -s localhost:8000/v1/memory/private

# CLI (live steps by default; --quiet to hide them)
foundry run "Should I build an AI co-founder for solo founders?"
foundry run --session <id> "What about its competition?"
foundry chat                                  # interactive, keeps context
foundry memory add --scope private "I want to bootstrap, no VC."
foundry memory list --scope shared
```

## Tests

```bash
pytest
```
