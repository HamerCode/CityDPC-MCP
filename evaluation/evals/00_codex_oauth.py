from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    from evals.codex_oauth import (
        DEFAULT_CALLBACK_PORTS,
        DEFAULT_CLIENT_ID,
        DEFAULT_ISSUER,
        DEFAULT_SCOPE,
        ensure_fresh_tokens,
        login_with_pkce,
        oauth_status,
        save_project_tokens,
    )
except ModuleNotFoundError:
    from codex_oauth import (
        DEFAULT_CALLBACK_PORTS,
        DEFAULT_CLIENT_ID,
        DEFAULT_ISSUER,
        DEFAULT_SCOPE,
        ensure_fresh_tokens,
        login_with_pkce,
        oauth_status,
        save_project_tokens,
    )


def default_project_token_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "codex_oauth.json"


def main() -> int:
    args = parse_args()
    if args.command == "status":
        print(oauth_status(args.output, include_global=not args.no_global))
        return 0
    if args.command == "refresh":
        tokens = ensure_fresh_tokens(
            args.output,
            include_global=not args.no_global,
            issuer=args.issuer,
            client_id=args.client_id,
        )
        save_project_tokens(args.output, tokens)
        print(f"Codex OAuth refreshed: accountId={tokens.account_id or 'unknown'}")
        print(f"Stored project token file: {args.output}")
        return 0
    if args.command == "login":
        tokens = login_with_pkce(
            output_path=args.output,
            issuer=args.issuer,
            client_id=args.client_id,
            scope=args.scope,
            callback_ports=tuple(args.port),
            timeout_seconds=args.timeout,
            open_browser=not args.no_browser,
            manual=args.manual,
        )
        print(f"Codex OAuth login complete: accountId={tokens.account_id or 'unknown'}")
        print(f"Stored project token file: {args.output}")
        return 0
    raise AssertionError(args.command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Codex OAuth PKCE helper for nationwide websearch enrichment.")
    parser.add_argument(
        "command",
        choices=["login", "status", "refresh"],
        help="login starts PKCE OAuth; status prints token metadata; refresh refreshes/exports tokens.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_project_token_path(),
        help="Project-local token file. Stores {access, refresh, expires, accountId}.",
    )
    parser.add_argument(
        "--issuer",
        default=os.environ.get("CODEX_OAUTH_ISSUER", DEFAULT_ISSUER),
        help="OAuth issuer base URL.",
    )
    parser.add_argument(
        "--client-id",
        default=os.environ.get("CODEX_OAUTH_CLIENT_ID", DEFAULT_CLIENT_ID),
        help="OAuth client id.",
    )
    parser.add_argument(
        "--scope",
        default=os.environ.get("CODEX_OAUTH_SCOPE", DEFAULT_SCOPE),
        help="OAuth scope string.",
    )
    parser.add_argument(
        "--port",
        type=int,
        action="append",
        default=list(DEFAULT_CALLBACK_PORTS),
        help="Callback port to try. Can be passed multiple times.",
    )
    parser.add_argument("--timeout", type=int, default=600, help="Callback wait timeout in seconds.")
    parser.add_argument("--manual", action="store_true", help="Skip local callback server and paste URL/code.")
    parser.add_argument("--no-browser", action="store_true", help="Print URL without opening the browser.")
    parser.add_argument(
        "--no-global",
        action="store_true",
        help="For status/refresh, do not fall back to ~/.codex/auth.json.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
