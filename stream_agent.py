"""Test client for the RAG agent: streaming answers plus chat management,
with no auth friction.

Authentication is automatic: the client logs in as a test user (default
"stream_agent", auto-created on first use; the session cookie is cached in
the OS temp directory). Set RAG_TOKEN to use an existing session token
instead.

Usage:
    python stream_agent.py                 # interactive conversation
    python stream_agent.py "a question"    # one-shot (exit code 1 on failure)

Interactive commands:
    /chats / /ls   list all chats (the current one is marked)
    /use <id>      switch to a chat (a unique prefix is enough)
    /new           start a fresh chat on the next question
    /del <id>      delete one chat (conversation + ingested data)
    /delall        delete all chats
    /help          show the commands
    /exit, /quit   leave

RAG_BASE_URL points the client at another server; RAG_CHAT starts in a
given chat; RAG_TEST_USER / RAG_TEST_PASSWORD override the test account.
"""

from http import cookiejar
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

BASE_URL = os.environ.get("RAG_BASE_URL", "http://127.0.0.1:8000")
HOST = urllib.parse.urlparse(BASE_URL).hostname or "127.0.0.1"
TOKEN = os.environ.get("RAG_TOKEN", "")
TEST_USER = os.environ.get("RAG_TEST_USER", "stream_agent")
TEST_PASSWORD = os.environ.get("RAG_TEST_PASSWORD", "stream_agent_test_password")
# The session cookie the server sets at login (must match app/api/auth.py).
COOKIE_NAME = "auth_token"
# In the OS temp dir so the project directory stays clean.
SESSION_CACHE = Path(tempfile.gettempdir()) / "stream_agent_session"
RESULT_PREVIEW_CHARS = 400

# --- HTTP helpers (cookie-carrying) ---

COOKIE_JAR = cookiejar.CookieJar()
OPENER = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(COOKIE_JAR))


def http(method, path, body=None):
    """A small JSON HTTP helper; returns (status, parsed body). The session
    cookie travels with every request via the shared cookie jar."""
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        BASE_URL + path, data=data, headers=headers, method=method
    )
    try:
        with OPENER.open(request, timeout=300) as response:
            text = response.read().decode()
            content_type = response.headers.get("Content-Type", "")
            return response.status, (
                json.loads(text) if text and "json" in content_type else text
            )
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw[:200]
        return exc.code, parsed
    except urllib.error.URLError as exc:
        detail = getattr(exc, "reason", None) or exc
        print(f"[cannot reach {BASE_URL}: {detail}]")
        sys.exit(1)


# --- authentication (test account, session cookie cached) ---


def _session_cookie(value: str) -> cookiejar.Cookie:
    """A cookie for the session token, addressed at this server's host."""
    return cookiejar.Cookie(
        version=0,
        name=COOKIE_NAME,
        value=value,
        port=None,
        port_specified=False,
        domain=HOST,
        domain_specified=True,
        domain_initial_dot=False,
        path="/",
        path_specified=True,
        secure=False,
        expires=None,
        discard=False,
        comment=None,
        comment_url=None,
        rest={},
        rfc2109=False,
    )


def ensure_session() -> None:
    """A valid session for the test account: RAG_TOKEN if set, else the
    cached one if still valid, else login (registering the user on first
    use). The session lives in the auth_token cookie."""
    if TOKEN:
        COOKIE_JAR.set_cookie(_session_cookie(TOKEN))
        status, _ = http("GET", "/auth/me")
        if status != 200:
            print(f"[RAG_TOKEN was rejected by {BASE_URL}]")
            sys.exit(1)
        return
    if SESSION_CACHE.exists():
        COOKIE_JAR.set_cookie(_session_cookie(SESSION_CACHE.read_text().strip()))
        status, _ = http("GET", "/auth/me")
        if status == 200:
            return
        COOKIE_JAR.clear()
    http("POST", "/auth/register", {"username": TEST_USER, "password": TEST_PASSWORD})
    status, body = http(
        "POST", "/auth/login", {"username": TEST_USER, "password": TEST_PASSWORD}
    )
    if status != 200:
        print(f"[could not log in as '{TEST_USER}': {body}]")
        sys.exit(1)
    cookies = [cookie for cookie in COOKIE_JAR if cookie.name == COOKIE_NAME]
    value = cookies[0].value if cookies else None
    if value is None:
        print("[the server did not set a session cookie]")
        sys.exit(1)
    SESSION_CACHE.write_text(value)


# --- chat management ---


def list_chats() -> list[dict]:
    status, body = http("GET", "/chats")
    if status != 200 or not isinstance(body, list):
        print(f"[could not list chats: {status} {body}]")
        return []
    return body


def resolve_chat(chats: list[dict], ident: str) -> dict | None:
    """An exact id or a unique prefix -> the chat dict, else None."""
    exact = [c for c in chats if c["chat_id"] == ident]
    if exact:
        return exact[0]
    prefixed = [c for c in chats if c["chat_id"].startswith(ident)]
    if len(prefixed) == 1:
        return prefixed[0]
    if not prefixed:
        print(f"[no chat matching '{ident}']")
    else:
        print(f"['{ident}' is ambiguous:]")
        for chat in prefixed:
            print(f"   {chat['chat_id'][:12]}")
    return None


def confirm(prompt):
    answer = input(f"{prompt} [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def delete_chats(chat_ids: list[str]) -> None:
    """Delete chats through the server (DELETE /chats/{id}), which removes
    each chat's conversation history, ingested documents/chunks/vectors, and
    the chat row. Ownership is enforced server-side."""
    for chat_id in chat_ids:
        status, body = http("DELETE", f"/chats/{chat_id}")
        if status == 204:
            print(f"deleted {chat_id[:12]}")
        elif status == 404:
            print(f"[chat {chat_id[:12]} not found]")
        else:
            print(f"[could not delete {chat_id[:12]}: {status} {body}]")


def fmt_time(stamp: float) -> str:
    return time.strftime("%m-%d %H:%M", time.localtime(stamp))


def print_chats(chats: list[dict], current: str | None) -> None:
    if not chats:
        print("  (no chats)")
        return
    for chat in chats:
        marker = "*" if chat["chat_id"] == current else " "
        print(
            f" {marker} {chat['chat_id'][:12]}  "
            f"{fmt_time(chat['created_at'])}  "
            f"{len(chat['sources'])} source(s)"
        )


HELP_TEXT = """commands:
  /chats (or /ls)  list all chats (the current one is marked)
  /use <id>        switch to a chat (a unique prefix is enough)
  /new             start a fresh chat on the next question
  /del <id>        delete one chat (conversation + ingested data)
  /delall          delete all chats
  /help            show this help
  /exit, /quit     leave"""


def handle_command(line: str, chat_id: str | None) -> tuple[str | None, bool]:
    """A REPL command; returns (new chat_id, should_exit)."""
    parts = line.split()
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else None

    if command in ("/chats", "/ls", "/list"):
        print_chats(list_chats(), chat_id)
    elif command in ("/use", "/switch") and arg:
        target = resolve_chat(list_chats(), arg)
        if target is not None:
            print(f"switched to {target['chat_id'][:12]}")
            chat_id = target["chat_id"]
    elif command == "/new":
        chat_id = None
        print("(a new chat will be created with the next question)")
    elif command == "/del" and arg:
        target = resolve_chat(list_chats(), arg)
        if target is not None and confirm(
            f"delete chat {target['chat_id'][:12]} "
            "(conversation + ingested documents/chunks/vectors)?"
        ):
            delete_chats([target["chat_id"]])
            if chat_id == target["chat_id"]:
                chat_id = None
    elif command == "/delall":
        chats = list_chats()
        if chats and confirm(f"delete all {len(chats)} chat(s)?"):
            delete_chats([c["chat_id"] for c in chats])
            chat_id = None
    elif command == "/help":
        print(HELP_TEXT)
    else:
        print(HELP_TEXT)
    return chat_id, False


# --- the answer stream ---


def sse_frames(response):
    """Yield (event, data) pairs from a server-sent event byte stream."""
    event = None
    data = []
    for raw_line in response:
        line = raw_line.decode("utf-8").rstrip("\n")
        if line.startswith("event: "):
            event = line.removeprefix("event: ")
        elif line.startswith("data: "):
            data.append(line.removeprefix("data: "))
        elif line == "" and event is not None:
            yield event, json.loads("".join(data))
            event, data = None, []


def preview(text: str) -> str:
    if len(text) <= RESULT_PREVIEW_CHARS:
        return text
    return (
        f"{text[:RESULT_PREVIEW_CHARS]} "
        f"... [{len(text) - RESULT_PREVIEW_CHARS} more chars]"
    )


def show_chunk_refs(data: dict) -> None:
    refs = data["chunk_refs"]
    print(f"cited chunks: {len(refs)}")
    for ref in refs:
        print(f"  - {ref}")


def show(event: str, data: dict) -> None:
    if event == "tool_call":
        print(f"> {data['name']} {json.dumps(data['arguments'])}")
    elif event == "tool_result":
        print(f"< {data['name']}:")
        for line in preview(data["content"]).splitlines():
            print(f"   {line}")
    elif event == "answer":
        print(f"\n{data['answer']}\n")
        show_chunk_refs(data)
    else:
        print(f"? {event}: {data}")


def turn(question: str, chat_id: str | None) -> tuple[str | None, str | None]:
    """Run one turn against the server.

    Streams the run to the terminal as it happens (tool steps and the
    answer's tokens) and returns (answer, chat_id): the final answer, or
    None when the request failed, and the chat the turn ran on (created by
    the server when none was given).
    """
    payload = {"query": question, "chat_id": chat_id}
    request = urllib.request.Request(
        f"{BASE_URL}/rag/answer/stream",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with OPENER.open(request, timeout=300) as response:
            answer = None
            chat = chat_id
            streamed = ""
            for event, data in sse_frames(response):
                if event == "chat":
                    chat = data["chat_id"]
                elif event == "answer_delta":
                    # The answer, live: print each token as it arrives.
                    print(data["content"], end="", flush=True)
                    streamed += data["content"]
                elif event == "tool_call" and streamed:
                    # Text streamed, then a tool call: it was commentary,
                    # not the answer. Mark it and drop it.
                    print("\n  (commentary - the model called a tool)\n")
                    streamed = ""
                elif event == "answer":
                    if streamed:
                        # The answer already typed itself out; finish the
                        # line and print the cited chunks under it.
                        print()
                        print()
                        show_chunk_refs(data)
                    else:
                        show("answer", data)
                    answer = data["answer"]
                else:
                    show(event, data)
            if answer is None:
                print("\n[the agent produced no answer]")
            return answer, chat
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            print(
                "\n[unauthorized: the session is no longer valid; delete "
                f"{SESSION_CACHE} and retry]"
            )
        else:
            print(f"\n[request failed: HTTP {exc.code}]")
        return None, chat_id
    except urllib.error.URLError as exc:
        detail = getattr(exc, "reason", None) or exc
        print(f"\n[request failed: {detail}]")
        return None, chat_id


# --- entry points ---


def interactive() -> None:
    status, me = http("GET", "/auth/me")
    who = me.get("username") if status == 200 and isinstance(me, dict) else TEST_USER
    chat_id = os.environ.get("RAG_CHAT") or None
    print(f"Testing the agent at {BASE_URL} as '{who}' (auth handled for you).")
    if chat_id:
        print(f"Starting in chat {chat_id[:12]}.")
    print("Ask a question and press Enter; /help for the commands;")
    print("/exit or Ctrl-D to quit.\n")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in {"/exit", "/quit"}:
            break
        try:
            if line.startswith("/"):
                chat_id, _ = handle_command(line, chat_id)
                continue
            answer, chat_id = turn(line, chat_id)
        except KeyboardInterrupt:
            print("\n[interrupted]")
            continue
        if answer is not None and chat_id and not os.environ.get("RAG_CHAT"):
            print(f"(chat: {chat_id[:12]})")
        print()


def main() -> None:
    ensure_session()
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        answer, _ = turn(question, os.environ.get("RAG_CHAT") or None)
        sys.exit(0 if answer is not None else 1)
    interactive()


if __name__ == "__main__":
    main()
