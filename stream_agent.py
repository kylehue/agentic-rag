"""Converse with the RAG agent and watch it think, token by token."""

import json
import os
import sys
import urllib.error
import urllib.request

# Overridable so the client can point at a non-default server:
#   RAG_BASE_URL=http://127.0.0.1:9000 python stream_agent.py
BASE_URL = os.environ.get("RAG_BASE_URL", "http://127.0.0.1:8000")
RESULT_PREVIEW_CHARS = 400


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


def turn(question: str, history: list[dict]) -> str | None:
    """Run one conversational turn against the server.

    Streams the run to the terminal as it happens — tool steps and the
    answer's tokens — and returns the final answer, or None when the
    request failed or the agent produced no answer. A failed turn appends
    nothing to the conversation.
    """
    payload = {"query": question, "history": history}
    request = urllib.request.Request(
        f"{BASE_URL}/rag/answer/stream",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            answer = None
            streamed = ""
            for event, data in sse_frames(response):
                if event == "answer_delta":
                    # The answer, live: print each token as it arrives.
                    print(data["content"], end="", flush=True)
                    streamed += data["content"]
                elif event == "tool_call" and streamed:
                    # Text streamed, then a tool call: it was commentary,
                    # not the answer. Mark it and drop it.
                    print("\n  (commentary — the model called a tool)\n")
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
            return answer
    except urllib.error.URLError as exc:
        detail = getattr(exc, "reason", None) or exc
        print(f"\n[request failed: {detail}]")
        return None


def interactive() -> None:
    history: list[dict] = []
    print(f"Conversing with the agent at {BASE_URL} (memory: in-process).")
    print("Ask a question and press Enter; 'exit' or Ctrl-D to quit.\n")
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        try:
            answer = turn(question, history)
        except KeyboardInterrupt:
            print("\n[interrupted]")
            continue
        if answer is None:
            continue
        # In-memory memory: the turn joins the conversation, oldest first.
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        print()


def main() -> None:
    if len(sys.argv) > 1:
        answer = turn(" ".join(sys.argv[1:]), [])
        sys.exit(0 if answer is not None else 1)
    interactive()


if __name__ == "__main__":
    main()
