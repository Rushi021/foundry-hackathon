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
cp .env.example .env   # optional PERPLEXITY_API_KEY / Supabase / LangSmith
```

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
foundry run "Should I build an AI co-founder for solo founders?"
# same as: python -m foundry "…"
```

## Tests

```bash
pytest
```
