import json


def sse_frame(event: str, data) -> str:
    """One server-sent event frame: `event: <name>\\ndata: <json>\\n\\n`."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
