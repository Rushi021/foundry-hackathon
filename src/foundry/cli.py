"""CLI smoke-test entrypoint."""

from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the Foundry founder workflow")
    parser.add_argument("prompt", nargs="?", help="User prompt")
    parser.add_argument(
        "--context",
        default=None,
        help="Optional user_context (future persistent brain hook)",
    )
    parser.add_argument("--json", action="store_true", help="Print raw JSON result")
    args = parser.parse_args(argv)

    if not args.prompt:
        parser.error("prompt is required")

    from foundry.graph import run_workflow

    result = run_workflow(args.prompt, user_context=args.context)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"route: {result['route']} ({result.get('route_confidence')})")
        print(f"why:   {result.get('route_rationale')}")
        print(f"run:   {result['run_id']} ({result['latency_ms']} ms)")
        print("---")
        print(result["response"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
