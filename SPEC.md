# Spec: Foundry — Phase 1 Intelligence & Communication Foundation

## Objective

Foundry is an AI company for solo founders: specialized agents with clear responsibilities,
authorized tools, shared knowledge, and the ability to coordinate. Phase 1 establishes the
foundation: understand a request in conversational context, route it with a dedicated
decision model, give the selected agent only the memory and tools it is authorized for,
execute (delegating when needed), and return one honest result.

Simple requests stay simple; sophisticated multi-agent work remains possible.

### Success criteria

- [x] Requests are routed by Jev (TypeSafe AI decision model); the LLM router is a flagged fallback only.
- [x] One Jev call returns destination, complexity, tool hint, memory relevance and clarification need.
- [x] Follow-ups keep conversational context (sessions); no automatic long-term memory retrieval.
- [x] Uncertain routing goes to context resolution once, then falls back to the Direct Assistant — no loops.
- [x] Direct Assistant, Co-Founder, COO, Research, Engineering, Marketing run through one registry-driven runtime.
- [x] Private Memory is readable only by the Co-Founder, enforced in code; leaks via delegation are blocked.
- [x] Delegation returns results to the requesting agent; cycles and depth are bounded.
- [x] Task status is derived from evidence; failures are never reported as success.
- [x] Per-run metrics: router provider/fallback, decision + model calls, tokens, tools, delegations, memory reads/denials, failures, latency.
- [x] New agents/tools are registry entries; the graph does not change.

---

## Tech Stack

| Layer | Choice |
|-------|--------|
| Language | Python 3.11+ |
| Orchestration | LangGraph (`StateGraph`) |
| Decision router | Jev via `POST https://api.typesafe.ai/v1/systemone` (`TYPESAFE_API_KEY`) |
| LLM | Provider-agnostic `provider:model` via `init_chat_model`; ChatGPT plan usage (SIWC) or API key |
| Research | Answer-only provider (default Perplexity Sonar) exposed as the `web_search` tool |
| Memory / sessions | JSON files under `FOUNDRY_DATA_DIR` (default `.foundry/`, gitignored) |
| API | FastAPI; CLI for local use |
| Observability | Terminal step stream, run metrics, LangSmith, Supabase `workflow_runs` (optional) |

---

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # set TYPESAFE_API_KEY for Jev routing

uvicorn foundry.api:app --reload --port 8000   # API
foundry run "Your prompt" [--session ID]       # one request
foundry chat                                   # interactive session
foundry memory add --scope private|shared "…"  # user-owned memory
pytest                                         # tests (no network)
```

---

## Project Structure

```text
src/foundry/
├── api.py / cli.py          # ingestion: POST /v1/workflows/run, /v1/memory; foundry run|chat|memory
├── graph.py                 # prepare_context → route → resolve? → execute → finalize
├── context.py               # sessions + context preparation (no model calls)
├── routing/
│   ├── models.py            # RoutingDecision, DecisionProvider protocol
│   ├── jev.py               # Jev provider (choice + noul questions, one round trip)
│   ├── llm_fallback.py      # LLM router, used only if Jev is unconfigured/fails
│   └── policy.py            # thresholds → destination / resolve / fallback
├── agents/
│   ├── registry.py          # AgentSpec: responsibility, model, tools, memory scopes, delegation
│   ├── prompts.py           # personas + status protocol
│   └── resolver.py          # context resolution / clarifying question
├── runtime.py               # agent loop: tools, approval gate, recall, delegate, escalate, status
├── memory.py                # Private/Shared stores + access gateway + leak detection
├── tools/                   # ToolSpec registry; web_search (Perplexity)
├── metrics.py               # RunMetrics
└── persistence/             # Supabase run log (soft no-op)
```

---

## Workflow

```text
user message
  → prepare_context   recent turns (≤6), earlier requests or existing summary, last agent
  → route             Jev: destination · complexity · tool · needs_memory · needs_clarification
      ├─ confident            → execute(destination)
      ├─ uncertain/missing    → resolve (fast LLM, only on this path)
      │     ├─ rewritten      → route again (once)
      │     ├─ unchanged      → execute(destination if confident else direct)
      │     └─ unresolved     → clarifying question (status needs_clarification)
      └─ uncertain, no history, nothing missing → execute(direct)
  → execute           run_agent(destination) — may escalate (Direct) or delegate
  → finalize          persist turns, return result + metrics
```

Router output is a recommendation. It never grants memory, tools, or actions.

## Agents

| Agent | Role | Tools | Memory | Delegates / escalates to |
|-------|------|-------|--------|---------------------------|
| `direct` | Fast answers, concepts, current info | web_search; gmail, calendar (not connected) | shared | escalates: cofounder, coo, research |
| `cofounder` | Strategy, opportunity evaluation, opinionated partner | web_search | **private**, shared | research, coo, engineering, marketing |
| `coo` | Operational execution, workstreams, consolidation | — | shared | research, engineering, marketing, cofounder (strategic escalation) |
| `research` | Evidence-based research | web_search | shared | — |
| `engineering` | Requirements, plans, readiness (no code execution in Phase 1) | github (not connected) | shared | — |
| `marketing` | Positioning, messaging, launch plans (draft only) | — | shared | — |

Result flows: Direct → user; Specialist → user; Co-Founder → specialist → Co-Founder → user;
COO → specialists → COO → user; Co-Founder → COO → specialists → COO → Co-Founder → user.
Escalation hands the whole request over; the target answers the user.

## Memory (basic)

- **Private:** user + Co-Founder only. `PRIVATE_MEMORY_AGENTS` is a hard-coded boundary that a
  registry misconfiguration cannot widen.
- **Shared:** any agent whose spec lists the scope.
- **Retrieval:** preloaded for the top-level agent only when `needs_memory ≥ ROUTER_MEMORY_THRESHOLD`;
  any agent may call `recall_memory` for its authorized scopes later. Keyword scoring.
- **Leak guard:** delegation briefs to, and results from, private-authorized agents are checked
  against private content surfaced in the run; matches are blocked.
- **Writes:** user-initiated only (CLI/API). Autonomous memory curation is out of scope.

## Task results

`completed | partial | needs_clarification | awaiting_approval | failed`. The runtime takes the
worse of the agent's declared status (`[[status:…]]` marker) and the derived status (tool
errors, unresolved failed subtasks, pending approvals). Consequential tools are recorded as
pending approvals and never executed in Phase 1.

---

## Boundaries

- **Always:** keep secrets in `.env`; structured router output; enforce permissions in code;
  report limitations (missing tools/keys) instead of pretending; fail soft if Supabase/Perplexity unset.
- **Ask first:** changing default research or decision provider; schema changes beyond
  `workflow_runs`; memory-writing policy; connecting consequential tools.
- **Never:** commit `.env`, keys or `.foundry/`; let a non-Co-Founder agent read Private Memory;
  report delegated or tool-issued work as completed without a result; invent research facts.

## Out of scope (Phase 1)

Autonomous software development and deployment, long-running scheduling, autonomous memory
curation, connected Gmail/Calendar/GitHub actions, voice transport (the ingestion path takes
text; a voice front-end would transcribe into the same `run_workflow` call), multi-user auth.
