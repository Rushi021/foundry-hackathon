-- Foundry workflow run logs / analytics
create table if not exists public.workflow_runs (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  user_prompt text not null,
  route text,
  route_rationale text,
  response text,
  latency_ms integer,
  metadata jsonb not null default '{}'::jsonb
);

create index if not exists workflow_runs_created_at_idx
  on public.workflow_runs (created_at desc);

create index if not exists workflow_runs_route_idx
  on public.workflow_runs (route);
