# Spec: Foundry — Founder Workflow Foundation

## Objective

Build the first vertical slice of **Foundry**: an AI consultancy platform where solo founders can delegate work to specialized agents that share context and work with real opinions (not sycophancy).

**This slice:** a LangGraph-orchestrated workflow with:

1. **Chief of Agents (categorization)** — classifies the user prompt and routes to the right specialist workflow.
2. **Founder agent (co-founder)** — the only specialist implemented now. Acts as a real co-founder: challenges, counters, brainstorms; never sugarcoats. Can run web research (Perplexity-like) when evidence is needed.
3. **Observability + persistence hooks** — LangSmith tracing and Supabase-ready run logging.

**Out of scope for this slice:** lawyer, sales/marketing, GTM, engineer, personal assistant agents; multi-user auth; product UI; full long-term memory/RAG of user company context.

### User stories

- As a solo founder, I submit a prompt (idea, decision, critique request) and get routed to the Founder agent when it’s a strategy/idea/co-founder conversation.
- As a solo founder, I get a direct, opinionated response that can cite research when claims need grounding.
- As a builder, I can run the workflow from a CLI with ChatGPT plan usage (`foundry login`) or env-configured API keys, and see traces in LangSmith.

### Success criteria

- [ ] `POST /v1/workflows/run` with `{ "prompt": "..." }` runs end-to-end (FastAPI).
- [ ] `python -m foundry.cli "Should I build X for Y?"` also works.
- [ ] Categorization returns a structured route with rationale; `founder` is fully handled; other known agents stubbed.
- [ ] Founder agent can call answer-only research (Perplexity by default) and never sugarcoats.
- [ ] LLM factory is provider-agnostic (`CATEGORIZE_LLM` / `FOUNDER_LLM` env as `provider:model`).
- [ ] LangSmith traces when tracing env is set; Supabase logs when configured, else soft no-op.
- [ ] State includes optional `user_context` hook for a future persistent brain (empty/unused in v0).

---

## Tech Stack

| Layer | Choice |
|-------|--------|
| Language | Python 3.11+ |
| Orchestration | LangGraph (`StateGraph`) |
| LLM | Provider-agnostic via `langchain.chat_models.init_chat_model` (`provider:model`). Default OpenAI (`openai:gpt-4o-mini` categorize, `openai:gpt-4o` founder) |
| Research | Answer-only research provider (default: Perplexity Sonar). Abstract interface so users can later connect other apps. Returns the answer — not a DIY search/browse workflow. |
| API | FastAPI (`POST /v1/workflows/run`) — primary interface for web/other clients; CLI also available |
| Observability | LangSmith (`LANGCHAIN_*` env vars) |
| Database | Supabase (Postgres) — `workflow_runs` table for logs/analytics |
| Packaging | `pyproject.toml` + `uv`/`pip` |

---

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # fill keys

# API (primary)
uvicorn foundry.api:app --reload --port 8000

# CLI
python -m foundry.cli "Your prompt here"

# Tests
pytest

# Lint (optional)
ruff check .
```

---

## Project Structure

```text
foundry-hackathon/
├── SPEC.md
├── README.md
├── pyproject.toml
├── .env.example
├── supabase/
│   └── migrations/001_workflow_runs.sql
├── src/foundry/
│   ├── __init__.py
│   ├── api.py                 # FastAPI app
│   ├── cli.py                 # CLI entrypoint
│   ├── config.py              # env settings
│   ├── llm.py                 # provider-agnostic chat model factory (API key or ChatGPT OAuth)
│   ├── chatgpt_auth/          # Sign in with ChatGPT (PKCE + plan usage → Responses API)
│   ├── state.py               # shared LangGraph state (+ future user_context)
│   ├── graph.py               # StateGraph: categorize → route → founder|unsupported
│   ├── agents/
│   │   ├── categorize.py      # Chief of Agents
│   │   ├── founder.py         # Co-founder + research tool
│   │   └── unsupported.py     # stub for future agents
│   ├── tools/
│   │   └── research.py        # ResearchProvider protocol + Perplexity default
│   └── persistence/
│       └── supabase_logger.py # run logging (optional if unset)
└── tests/
    ├── test_categorize.py
    ├── test_research.py
    └── test_graph_routing.py
```

---

## Architecture (this slice)

```text
User prompt
    │
    ▼
┌─────────────────────┐
│  Categorize (Chief) │  → route: founder | lawyer | sales | gtm | engineer | assistant | unsupported
└─────────┬───────────┘
          │
          ├─ founder ──────────► Founder agent (tools: research) ──► response
          │
          └─ anything else ────► Unsupported stub (name future agent, ask to rephrase / wait)
```

**State (shared):** `messages`, `user_prompt`, `route`, `route_rationale`, `run_id`, `final_response`.

---

## Agent contracts

### Categorize (Chief of Agents)

- **Input:** raw user prompt (+ optional future context blob).
- **Output:** structured `{ route, rationale, confidence }`.
- **Routes known today:** `founder` fully handled; others recognized for future wiring but stubbed.
- **Heuristic:** strategy, idea validation, brainstorming, product direction, “be my co-founder” → `founder`.

### Founder (Co-founder)

- **Persona:** Equal partner. Own opinions. Push back when the idea is weak, crowded, or poorly scoped. Brainstorm when the idea has legs. Never empty praise.
- **Tools:** `research(query)` — answer-only research via pluggable `ResearchProvider` (default Perplexity Sonar). Returns the researched answer string; does not expose a DIY browse/search workflow to the agent.
- **When to research:** market size, competitors, claims that need evidence, “is this a thing” questions.
- **Output:** structured co-founder reply: verdict posture (push / refine / pursue), key counters, suggested next moves, research notes if used.

---

## Code Style

```python
# Prefer TypedDict / Pydantic models for graph I/O
class RouteDecision(BaseModel):
    route: Literal["founder", "lawyer", "sales", "gtm", "engineer", "assistant", "unsupported"]
    rationale: str
    confidence: float
```

- Package under `src/foundry`.
- Env via pydantic-settings; no secrets in code.
- Thin nodes: one file per agent; graph wires them.

---

## Testing Strategy

- **Unit:** categorization parsing / routing map with mocked LLM.
- **Graph:** with mocked LLM + research tool, assert `founder` path produces `final_response`; non-founder path hits stub.
- No live API calls in CI; optional manual smoke with real keys.

---

## Boundaries

- **Always:** keep secrets in `.env`; structured route output; founder system prompt forbids sugarcoating; fail soft if Supabase/Perplexity unset (clear warning).
- **Ask first:** changing default research provider; adding new agent nodes; schema changes beyond `workflow_runs`; wiring the persistent brain.
- **Never:** commit `.env` or API keys; pretend unimplemented agents work; invent research facts without calling the tool.

---

## Persistence (minimal)

`workflow_runs` columns: `id`, `created_at`, `user_prompt`, `route`, `route_rationale`, `response`, `latency_ms`, `metadata` (jsonb).

LangSmith remains the trace UI; Supabase is analytics/logs for the product.

---

## Decisions (locked)

1. **LLM:** OpenAI by default via Sign in with ChatGPT plan usage (OAuth + Responses API) when signed in; `OPENAI_API_KEY` remains a fallback. Factory is provider-agnostic (`provider:model`).
2. **Research:** Default is Perplexity-like answer-only research (Perplexity Sonar API). Pluggable provider interface for future user-connected apps. Agent gets the answer, not a search workflow.
3. **Invocation:** FastAPI is the primary interface; CLI remains for local smoke tests.
4. **User context / brain:** Persistent company brain is planned separately. v0 leaves an optional `user_context` field in state but does not implement the brain.

---

## Assumptions (confirmed)

1. Default OpenAI models; default Perplexity for research answers.
2. FastAPI + CLI; web UI later.
3. Supabase optional at runtime (schema shipped; skip if env missing).
4. Only `founder` fully implemented; other routes stubbed for the future.
5. No auth / multi-tenancy yet.
