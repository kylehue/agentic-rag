from dataclasses import dataclass, field
from typing import Any

# Well-known ingest progress event names. The set is open: plugins may emit
# their own fine-grained `plugin_state` events without adding new names.
INGEST_CHAT = "chat"
INGEST_QUEUED = "queued"
INGEST_STARTED = "started"
INGEST_STAGE = "stage"
INGEST_PLUGIN_STATE = "plugin_state"
INGEST_FILE_DONE = "file_done"
INGEST_DONE = "done"
INGEST_ERROR = "error"

# Events that end a job's event stream.
INGEST_TERMINAL = frozenset({INGEST_DONE, INGEST_ERROR})


@dataclass(frozen=True)
class IngestEvent:
    """One neutral item in an ingestion's progress stream: a wire event name
    plus a payload dict.

    The pipeline and plugins publish these (through a ProgressEmitter); the
    SSE endpoint serializes them without knowing about ingestion. `payload`
    keys are per-event: `file` (the file the event is about), `stage` /
    `state` / `plugin` for progress, `error` for failures, counts for
    completions.
    """

    name: str
    payload: dict[str, Any] = field(default_factory=dict)
