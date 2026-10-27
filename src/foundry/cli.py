"""CLI — ChatGPT login, workflow runs, interactive chat, and memory management."""

from __future__ import annotations

import argparse
import json
import os
import sys

from dotenv import load_dotenv

_AUTH_COMMANDS = frozenset({"login", "logout", "auth", "models", "run", "chat", "memory"})


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

    run_p = sub.add_parser("run", help="Run one request through the workflow")
    run_p.add_argument("prompt", help="User prompt")
    run_p.add_argument(
        "--session",
        default=None,
        help="Conversation id to continue (printed after each run)",
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

    chat_p = sub.add_parser("chat", help="Interactive conversation (keeps session context)")
    chat_p.add_argument("--session", default=None, help="Conversation id to resume")
    chat_p.add_argument("--quiet", action="store_true", help="Hide live step output")

    mem_p = sub.add_parser("memory", help="Manage your long-term memory")
    mem_sub = mem_p.add_subparsers(dest="memory_command", required=True)
    mem_add = mem_sub.add_parser("add", help="Store a memory")
    mem_add.add_argument("content")
    mem_add.add_argument(
        "--scope",
        choices=["private", "shared"],
        default="shared",
        help="private: only you and the Co-Founder; shared: all authorized agents",
    )
    mem_list = mem_sub.add_parser("list", help="List memories")
    mem_list.add_argument("--scope", choices=["private", "shared"], default="shared")

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

    if args.command == "memory":
        return _memory(args)

    if not _llm_ready():
        return 1

    if args.command == "chat":
        return _chat(session_id=args.session, quiet=args.quiet)

    return _run_workflow(
        args.prompt,
        session_id=args.session,
        as_json=args.json,
        quiet=args.quiet,
    )


def _llm_ready() -> bool:
    from foundry.llm import chatgpt_plan_ready

    if not chatgpt_plan_ready() and not os.environ.get("OPENAI_API_KEY"):
        print(
            "No LLM credentials. Run `foundry login` (ChatGPT plan) "
            "or set OPENAI_API_KEY.",
            file=sys.stderr,
        )
        return False
    return True


def _memory(args: argparse.Namespace) -> int:
    from foundry.config import get_settings
    from foundry.memory import MemoryStore

    store = MemoryStore(get_settings().foundry_data_dir)
    if args.memory_command == "add":
        item = store.add(args.scope, args.content)
        print(f"stored {item.scope} memory {item.id}")
    else:
        for item in store.list(args.scope):
            print(f"{item.id}\t{item.created_at}\t{item.content}")
    return 0


def _print_result(result: dict) -> None:
    print("---")
    print(result["response"])
    print("---")
    print(
        f"agent={result['route']} status={result['status']} session={result['session_id']} "
        f"latency_ms={result['latency_ms']} model_calls={result['metrics']['model_calls']}"
    )


def _run_workflow(
    prompt: str,
    *,
    session_id: str | None,
    as_json: bool,
    quiet: bool,
) -> int:
    from foundry.graph import run_workflow

    result = run_workflow(prompt, session_id=session_id, verbose=not quiet)

    if as_json:
        print(json.dumps(result, indent=2))
    else:
        _print_result(result)
    return 0 if result["status"] != "failed" else 2


def _chat(*, session_id: str | None, quiet: bool) -> int:
    from foundry.graph import run_workflow

    print("Foundry chat — empty line or Ctrl-D to exit.")
    while True:
        try:
            prompt = input("\nyou> ").strip()
        except EOFError:
            break
        if not prompt:
            break
        result = run_workflow(prompt, session_id=session_id, verbose=not quiet)
        session_id = result["session_id"]
        _print_result(result)
    if session_id:
        print(f"\nresume with: foundry chat --session {session_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
