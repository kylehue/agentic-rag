# The ingest progress event names, on top of the neutral `Event` type
# (`app/utils/events.py`). The set is open: plugins may emit their own
# fine-grained `plugin_state` events without adding new names.
INGEST_CHAT = "chat"
INGEST_QUEUED = "queued"
INGEST_STARTED = "started"
INGEST_STAGE = "stage"
INGEST_PLUGIN_STATE = "plugin_state"
INGEST_FILE_DONE = "file_done"
INGEST_DONE = "done"
INGEST_ERROR = "error"

# Event names that end an ingest job's event stream.
INGEST_TERMINAL = frozenset({INGEST_DONE, INGEST_ERROR})
