from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any


DEFAULT_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
DEFAULT_ISSUER = "https://auth.openai.com"
DEFAULT_SCOPE = "openid profile email offline_access api.connectors.read api.connectors.invoke"
DEFAULT_CALLBACK_HOST = "localhost"
DEFAULT_CALLBACK_PATH = "/auth/callback"
DEFAULT_CALLBACK_PORTS = (1455, 1457)
MIN_VALID_SECONDS = 120


@dataclass
class CodexOAuthTokens:
    access: str
    refresh: str
    expires: int
    account_id: str
    source: str = ""

    def is_fresh(self, min_valid_seconds: int = MIN_VALID_SECONDS) -> bool:
        return bool(self.access) and self.expires > int(time.time()) + min_valid_seconds

    def to_project_json(self) -> dict[str, object]:
        return {
            "access": self.access,
            "refresh": self.refresh,
            "expires": self.expires,
            "accountId": self.account_id,
        }


def default_project_token_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "codex_oauth.json"


def default_global_auth_path() -> Path:
    return Path.home() / ".codex" / "auth.json"


def oauth_status(project_path: Path | None = None, *, include_global: bool = True) -> str:
    token = load_tokens(project_path or default_project_token_path(), include_global=include_global)
    if not token:
        return "Codex OAuth: no token file found"
    expires_text = datetime.fromtimestamp(token.expires, tz=timezone.utc).isoformat()
    freshness = "fresh" if token.is_fresh() else "expired/refresh-needed"
    scopes = " ".join(token_scopes(token.access)) or "unknown"
    return (
        f"Codex OAuth: {freshness}, accountId={token.account_id or 'unknown'}, "
        f"expires={expires_text}, scopes={scopes}, source={token.source}"
    )


def load_tokens(project_path: Path | None = None, *, include_global: bool = True) -> CodexOAuthTokens | None:
    if project_path and project_path.exists():
        try:
            return load_project_tokens(project_path)
        except Exception:
            pass
    if include_global:
        global_path = default_global_auth_path()
        if global_path.exists():
            try:
                return load_global_codex_tokens(global_path)
            except Exception:
                pass
    return None


def load_project_tokens(path: Path) -> CodexOAuthTokens:
    data = json.loads(path.read_text(encoding="utf-8"))
    access = str(data.get("access", ""))
    return CodexOAuthTokens(
        access=access,
        refresh=str(data.get("refresh", "")),
        expires=int(data.get("expires") or jwt_exp(access) or 0),
        account_id=str(data.get("accountId", "") or extract_account_id(access)),
        source=str(path),
    )


def load_global_codex_tokens(path: Path) -> CodexOAuthTokens:
    data = json.loads(path.read_text(encoding="utf-8"))
    tokens = data.get("tokens") or {}
    access = str(tokens.get("access_token", ""))
    return CodexOAuthTokens(
        access=access,
        refresh=str(tokens.get("refresh_token", "")),
        expires=int(jwt_exp(access) or 0),
        account_id=str(tokens.get("account_id", "") or extract_account_id(access)),
        source=str(path),
    )


def save_project_tokens(path: Path, tokens: CodexOAuthTokens) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(tokens.to_project_json(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def ensure_fresh_tokens(
    project_path: Path | None = None,
    *,
    include_global: bool = True,
    issuer: str | None = None,
    client_id: str | None = None,
) -> CodexOAuthTokens:
    path = project_path or default_project_token_path()
    tokens = load_tokens(path, include_global=include_global)
    if not tokens:
        raise RuntimeError("No Codex OAuth tokens found. Run 00_codex_oauth.py login first.")
    if tokens.is_fresh():
        return tokens
    if not tokens.refresh:
        raise RuntimeError("Codex OAuth access token is expired and no refresh token is available.")
    refreshed = refresh_tokens(
        tokens.refresh,
        issuer=issuer or os.environ.get("CODEX_OAUTH_ISSUER", DEFAULT_ISSUER),
        client_id=client_id or os.environ.get("CODEX_OAUTH_CLIENT_ID", DEFAULT_CLIENT_ID),
    )
    save_project_tokens(path, refreshed)
    return refreshed


def refresh_tokens(
    refresh_token: str,
    *,
    issuer: str = DEFAULT_ISSUER,
    client_id: str = DEFAULT_CLIENT_ID,
) -> CodexOAuthTokens:
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
    }
    data = token_request(issuer, payload)
    access = str(data.get("access_token", ""))
    refresh = str(data.get("refresh_token", "") or refresh_token)
    expires = expires_from_response(data, access)
    account_id = str(data.get("account_id", "") or extract_account_id(access))
    if not access:
        raise RuntimeError("Token refresh did not return an access token.")
    return CodexOAuthTokens(access=access, refresh=refresh, expires=expires, account_id=account_id)


def login_with_pkce(
    *,
    output_path: Path | None = None,
    issuer: str = DEFAULT_ISSUER,
    client_id: str = DEFAULT_CLIENT_ID,
    scope: str = DEFAULT_SCOPE,
    callback_host: str = DEFAULT_CALLBACK_HOST,
    callback_ports: tuple[int, ...] = DEFAULT_CALLBACK_PORTS,
    callback_path: str = DEFAULT_CALLBACK_PATH,
    timeout_seconds: int = 600,
    open_browser: bool = True,
    manual: bool = False,
) -> CodexOAuthTokens:
    state = token_urlsafe(32)
    verifier = token_urlsafe(64)
    challenge = pkce_challenge(verifier)

    callback: CallbackCapture | None = None
    redirect_uri = f"http://{callback_host}:{callback_ports[0]}{callback_path}"
    if not manual:
        callback = CallbackCapture(callback_host, callback_ports, callback_path, state, timeout_seconds)
        try:
            callback.start()
            redirect_uri = callback.redirect_uri
        except OSError:
            callback = None

    authorize_url = build_authorize_url(
        issuer=issuer,
        client_id=client_id,
        redirect_uri=redirect_uri,
        scope=scope,
        code_challenge=challenge,
        state=state,
    )

    print("Open this URL to sign in:")
    print(authorize_url)
    if open_browser:
        webbrowser.open(authorize_url)

    if callback is not None:
        try:
            code = callback.wait_for_code()
        except TimeoutError:
            print("Callback timed out; paste the final redirect URL or code manually.")
            code = prompt_for_code(expected_state=state)
        finally:
            callback.close()
    else:
        print("Could not bind callback listener; paste the final redirect URL or code manually.")
        code = prompt_for_code(expected_state=state)

    data = exchange_code(
        code=code,
        code_verifier=verifier,
        client_id=client_id,
        issuer=issuer,
        redirect_uri=redirect_uri,
    )
    access = str(data.get("access_token", ""))
    refresh = str(data.get("refresh_token", ""))
    expires = expires_from_response(data, access)
    account_id = str(data.get("account_id", "") or extract_account_id(access))
    if not access or not refresh:
        raise RuntimeError("OAuth exchange did not return both access and refresh tokens.")
    tokens = CodexOAuthTokens(access=access, refresh=refresh, expires=expires, account_id=account_id)
    save_project_tokens(output_path or default_project_token_path(), tokens)
    return tokens


def build_authorize_url(
    *,
    issuer: str,
    client_id: str,
    redirect_uri: str,
    scope: str,
    code_challenge: str,
    state: str,
) -> str:
    query = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": scope,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
        "codex_cli_simplified_flow": "true",
        "id_token_add_organizations": "true",
    }
    return f"{issuer.rstrip('/')}/oauth/authorize?{urllib.parse.urlencode(query)}"


def exchange_code(
    *,
    code: str,
    code_verifier: str,
    client_id: str,
    issuer: str,
    redirect_uri: str,
) -> dict[str, Any]:
    payload = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "code_verifier": code_verifier,
    }
    return token_request(issuer, payload)


def token_request(issuer: str, payload: dict[str, str]) -> dict[str, Any]:
    body = urllib.parse.urlencode(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{issuer.rstrip('/')}/oauth/token",
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OAuth token request failed with HTTP {exc.code}: {error_body[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OAuth token request failed: {exc.reason}") from exc


class CallbackCapture:
    def __init__(
        self,
        host: str,
        ports: tuple[int, ...],
        path: str,
        state: str,
        timeout_seconds: int,
    ) -> None:
        self.host = host
        self.ports = ports
        self.path = path
        self.state = state
        self.timeout_seconds = timeout_seconds
        self.server: HTTPServer | None = None
        self.redirect_uri = ""
        self._code: str | None = None
        self._error: BaseException | None = None
        self._event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        capture = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                try:
                    parsed = urllib.parse.urlparse(self.path)
                    params = urllib.parse.parse_qs(parsed.query)
                    if parsed.path != capture.path:
                        self.respond(404, "Not Found")
                        return
                    returned_state = first(params.get("state"))
                    if returned_state != capture.state:
                        self.respond(400, "State mismatch. Return to the terminal and paste the redirect URL.")
                        capture._error = RuntimeError("OAuth callback state mismatch")
                        capture._event.set()
                        return
                    error = first(params.get("error"))
                    if error:
                        description = first(params.get("error_description"))
                        self.respond(400, "Sign-in failed. Return to the terminal.")
                        capture._error = RuntimeError(f"OAuth error: {error} {description}".strip())
                        capture._event.set()
                        return
                    code = first(params.get("code"))
                    if not code:
                        self.respond(400, "Missing code. Return to the terminal.")
                        capture._error = RuntimeError("OAuth callback did not contain a code")
                        capture._event.set()
                        return
                    capture._code = code
                    self.respond(200, "Sign-in complete. You can return to Codex.")
                    capture._event.set()
                except BaseException as exc:
                    capture._error = exc
                    capture._event.set()

            def respond(self, status: int, message: str) -> None:
                body = f"<html><body><h1>{message}</h1></body></html>".encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format: str, *args: object) -> None:
                return

        last_error: OSError | None = None
        for port in self.ports:
            try:
                self.server = HTTPServer((self.host, port), Handler)
            except OSError as exc:
                last_error = exc
                continue
            self.redirect_uri = f"http://{self.host}:{port}{self.path}"
            break
        if self.server is None:
            raise last_error or OSError("Could not bind OAuth callback listener")
        self.server.timeout = 1
        self._thread = threading.Thread(target=self._serve_until_done, daemon=True)
        self._thread.start()

    def _serve_until_done(self) -> None:
        assert self.server is not None
        deadline = time.time() + self.timeout_seconds
        while not self._event.is_set() and time.time() < deadline:
            self.server.handle_request()
        if not self._event.is_set():
            self._error = TimeoutError("OAuth callback timed out")
            self._event.set()

    def wait_for_code(self) -> str:
        self._event.wait(self.timeout_seconds + 2)
        if self._error:
            raise self._error
        if not self._code:
            raise TimeoutError("OAuth callback timed out")
        return self._code

    def close(self) -> None:
        if self.server:
            self.server.server_close()


def prompt_for_code(*, expected_state: str) -> str:
    raw = input("Paste redirect URL or authorization code: ").strip()
    if raw.startswith("http://") or raw.startswith("https://"):
        parsed = urllib.parse.urlparse(raw)
        params = urllib.parse.parse_qs(parsed.query)
        state = first(params.get("state"))
        if state and state != expected_state:
            raise RuntimeError("Pasted redirect URL has the wrong OAuth state.")
        code = first(params.get("code"))
        if not code:
            raise RuntimeError("Pasted redirect URL did not contain a code parameter.")
        return code
    return raw


def pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def token_urlsafe(num_bytes: int) -> str:
    return secrets.token_urlsafe(num_bytes).rstrip("=")


def jwt_exp(token: str) -> int | None:
    claims = decode_jwt_payload(token)
    value = claims.get("exp")
    return int(value) if isinstance(value, int | float) else None


def extract_account_id(token: str) -> str:
    claims = decode_jwt_payload(token)
    candidates = [
        claims.get("accountId"),
        claims.get("account_id"),
        claims.get("chatgpt_account_id"),
    ]
    auth_claim = claims.get("https://api.openai.com/auth")
    if isinstance(auth_claim, dict):
        candidates.extend(
            [
                auth_claim.get("chatgpt_account_id"),
                auth_claim.get("account_id"),
                auth_claim.get("accountId"),
            ]
        )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate
    return ""


def token_scopes(token: str) -> list[str]:
    scopes = decode_jwt_payload(token).get("scp")
    if isinstance(scopes, list):
        return [str(scope) for scope in scopes]
    if isinstance(scopes, str):
        return scopes.split()
    return []


def token_has_scope(token: str, required_scope: str) -> bool:
    return required_scope in token_scopes(token)


def decode_jwt_payload(token: str) -> dict[str, Any]:
    if not token or token.count(".") < 2:
        return {}
    payload = token.split(".", 2)[1]
    payload += "=" * ((4 - len(payload) % 4) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload.encode("ascii"))
        claims = json.loads(decoded.decode("utf-8"))
    except Exception:
        return {}
    return claims if isinstance(claims, dict) else {}


def expires_from_response(data: dict[str, Any], access_token: str) -> int:
    jwt_value = jwt_exp(access_token)
    if jwt_value:
        return jwt_value
    expires_in = data.get("expires_in")
    if isinstance(expires_in, int | float):
        return int(time.time() + expires_in)
    return int(time.time())


def first(values: list[str] | None) -> str:
    return values[0] if values else ""
