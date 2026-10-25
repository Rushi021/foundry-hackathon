"""CLI — ChatGPT login + founder workflow smoke tests."""

from __future__ import annotations

import argparse
import json
import os
import sys

from dotenv import load_dotenv

_AUTH_COMMANDS = frozenset({"login", "logout", "auth", "models", "run"})


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    raw = list(argv if argv is not None else sys.argv[1:])

    # Legacy: `foundry "prompt"` → `foundry run "prompt"`
    if raw and raw[0] not in _AUTH_COMMANDS and not raw[0].startswith("-"):
        raw = ["run", *raw]

    parser = argparse.ArgumentParser(
        prog="foundry",
        description="Foundry — ChatGPT plan usage + founder workflow",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    login_p = sub.add_parser(
        "login",
        help="Sign in with ChatGPT (OAuth + plan usage consent)",
    )
    login_p.add_argument(
        "--no-browser",
        action="store_true",
        help="Print the authorize URL only (do not open a browser)",
    )
    login_p.add_argument(
        "--new-account",
        action="store_true",
        help="Register a new ChatGPT account (dynamic client registration)",
    )
    login_p.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Seconds to wait for the OAuth callback (default: 300)",
    )

    sub.add_parser("logout", help="Sign out and revoke the ChatGPT session")
    sub.add_parser("auth", help="Show ChatGPT sign-in / plan-usage status")
    sub.add_parser(
        "models",
        help="List models available to the signed-in ChatGPT account",
    )

    run_p = sub.add_parser("run", help="Run the founder workflow")
    run_p.add_argument("prompt", help="User prompt")
    run_p.add_argument(
        "--context",
        default=None,
        help="Optional user_context (future persistent brain hook)",
    )
    run_p.add_argument(
        "--json",
        action="store_true",
        help="Print final result as JSON after the live step stream",
    )
    run_p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress live per-step terminal output",
    )

    args = parser.parse_args(raw)

    if args.command == "login":
        from foundry.chatgpt_auth import login

        login(
            open_browser=not args.no_browser,
            timeout=args.timeout,
            new_account=args.new_account,
        )
        return 0

    if args.command == "logout":
        from foundry.chatgpt_auth import logout

        logout()
        return 0

    if args.command == "auth":
        from foundry.chatgpt_auth import status

        print(json.dumps(status(), indent=2))
        return 0

    if args.command == "models":
        from foundry.chatgpt_auth import list_models

        for m in list_models():
            print(f"{m['slug']}\t{m['display_name']}")
        return 0

    return _run_workflow(
        args.prompt,
        context=args.context,
        as_json=args.json,
        quiet=args.quiet,
    )


def _run_workflow(
    prompt: str,
    *,
    context: str | None,
    as_json: bool,
    quiet: bool,
) -> int:
    from foundry.graph import run_workflow
    from foundry.llm import chatgpt_plan_ready

    if not chatgpt_plan_ready() and not os.environ.get("OPENAI_API_KEY"):
        print(
            "No LLM credentials. Run `foundry login` (ChatGPT plan) "
            "or set OPENAI_API_KEY.",
            file=sys.stderr,
        )
        return 1

    result = run_workflow(
        prompt,
        user_context=context,
        verbose=not quiet,
    )

    if as_json:
        print(json.dumps(result, indent=2))
    elif quiet:
        print(f"route: {result['route']} ({result.get('route_confidence')})")
        print(f"why:   {result.get('route_rationale')}")
        print(f"run:   {result['run_id']} ({result['latency_ms']} ms)")
        print("---")
        print(result["response"])
    else:
        print("---")
        print(
            f"done run={result['run_id']} route={result['route']} "
            f"latency_ms={result['latency_ms']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
