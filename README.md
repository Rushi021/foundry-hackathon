# foundry-hackathon

A complete AI consultant — from your co-founder, to your company's engineers, to your personal assistant.

Solo founders and builders get specialist agents that share context and work with real opinions (no sugarcoating). This repo starts with the **founder (co-founder) workflow**.

## v0 — Founder workflow POC

```text
prompt → Chief of Agents (categorize) → Founder agent (+ Perplexity research)
                                      ↘ stub for future agents
```

| Piece | Tech |
|-------|------|
| Orchestration | LangGraph |
| Observability | LangSmith |
| API | FastAPI |
| LLM | Provider-agnostic (`provider:model`, default OpenAI) |
| Research | Answer-only provider (default Perplexity Sonar) |
| Logs | Supabase `workflow_runs` (optional) |

See [SPEC.md](./SPEC.md) for the full contract.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # add OPENAI_API_KEY, optional PERPLEXITY_API_KEY / Supabase / LangSmith
```

## Run

Live per-step status prints to the terminal (CLI and API server) so you can verify
each node before Supabase logging is finalized:

```text
[foundry:abcd1234] ▶ run.start | prompt=...
[foundry:abcd1234] ▶ categorize.start | model=openai:gpt-4o-mini
[foundry:abcd1234] ▶ categorize.decision | route=founder | confidence=0.9 | ...
[foundry:abcd1234] ▶ founder.start | ...
[foundry:abcd1234] ▶ research.start | query=...
[foundry:abcd1234] ▶ founder.response | response=...
[foundry:abcd1234] ▶ run.complete | ...
```

```bash
# API (primary) — watch the uvicorn terminal for step stream
uvicorn foundry.api:app --reload --port 8000

curl -s http://localhost:8000/v1/workflows/run \
  -H 'content-type: application/json' \
  -d '{"prompt":"Should I build an AI co-founder for solo founders?"}'

# CLI (live steps by default; --quiet to hide them)
python -m foundry.cli "Should I build an AI co-founder for solo founders?"
```

## Tests

```bash
pytest
```
