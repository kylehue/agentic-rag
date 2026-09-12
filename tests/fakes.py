from collections.abc import AsyncIterator
from types import SimpleNamespace

from unstructured.documents.elements import (
    ElementMetadata,
    Footer,
    Image,
    NarrativeText,
    PageBreak,
    Table,
)

from app.agent_tools import AgentTool
from app.embedders.base import Embedder
from app.llm.base import (
    ChatMessage,
    LLMProvider,
    RawDelta,
    RawResult,
    ToolCall,
    ToolSpec,
)
from app.plugin.context import IngestionContext, IngestionFile
from app.plugin.registry import PluginRegistry
from app.plugin.runtime import IngestionRuntime
from app.services.rag import RagService
from app.store_file.base import FileStorage
from app.store_sql.base import SqlStorage
from app.store_vector.base import VectorStorage


class FakeLLM(LLMProvider):
    """LLM stub that replays scripted raw completions.

    One entry (the common case) is returned for every call; several entries
    are consumed in order and the last one repeats once the script runs out.
    Entries may be a str (plain text content), a RawResult, or an Exception
    to raise. The provider's strategy layer (tool-convention parsing,
    corrective retries, structured validation) runs for real on top of these
    raw completions.

    With `chunk_size`, the replayed text is split into that many characters
    per `RawDelta`, to exercise token-level streaming for real.
    """

    def __init__(
        self,
        *responses: str | RawResult | BaseException,
        chunk_size: int | None = None,
    ) -> None:
        self._responses = list(responses) or [""]
        self._index = 0
        self._chunk_size = chunk_size
        self.calls: list[list[ChatMessage]] = []
        # Recorded message content; a str, or a list of text/image parts.
        self.prompts: list = []
        self.tools: list[list[ToolSpec]] = []
        self.json_schemas: list[dict | None] = []

    def _next(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None,
        json_schema: dict | None,
    ) -> RawResult:
        self.calls.append(messages)
        self.prompts.append(messages[-1].content if messages else "")
        self.tools.append(tools or [])
        self.json_schemas.append(json_schema)

        if len(self._responses) == 1:
            item = self._responses[0]
        else:
            item = self._responses[self._index]
            self._index = min(self._index + 1, len(self._responses) - 1)

        if isinstance(item, BaseException):
            raise item
        if isinstance(item, RawResult):
            return item
        return RawResult(content=item)

    async def stream_complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        json_schema: dict | None = None,
    ) -> AsyncIterator[RawDelta]:
        """Replay the scripted result as a stream: the text in `chunk_size`
        pieces (or one piece), then each tool call whole."""
        result = self._next(messages, tools, json_schema)
        if isinstance(result.content, str):
            text = result.content
        else:
            text = "".join(part for part in result.content if isinstance(part, str))  # type: ignore
        if self._chunk_size:
            for start in range(0, len(text), self._chunk_size):
                yield RawDelta(text=text[start : start + self._chunk_size])
        elif text:
            yield RawDelta(text=text)
        for call in result.tool_calls:
            yield RawDelta(tool_call=call)


class FakeEmbedder(Embedder):
    async def embed_documents(self, texts) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeVectorStorage(VectorStorage):
    def __init__(self) -> None:
        self.added: list[tuple[list[str], list[list[float]]]] = []

    async def close(self) -> None:
        pass

    async def add(self, ids, embeddings) -> None:
        if len(ids) != len(embeddings):
            raise ValueError("ids and embeddings must have the same length")
        self.added.append((list(ids), [list(e) for e in embeddings]))

    async def search(self, query_embedding, top_k=5):
        return []

    async def delete(self, ids) -> None:
        pass


class FakeSqlStorage(SqlStorage):
    def __init__(self) -> None:
        self.tables: dict[str, list[dict]] = {}

    def get_sql_dialect(self) -> str:
        return "SQLite"

    async def close(self) -> None:
        pass

    async def get_table(self, table_name):
        raise NotImplementedError

    async def ensure_table(self, table_name, columns) -> None:
        self.tables.setdefault(table_name, [])

    async def upsert(self, table_name, rows, conflict_columns=()) -> None:
        self.tables.setdefault(table_name, []).extend(rows)

    async def get(self, table_name, condition):
        return None

    async def get_all(self, table_name, condition=None, limit=None):
        return []

    async def delete(self, table_name, condition) -> bool:
        return False

    async def query(self, sql_query, limit):
        return []

    async def search(self, table_name, search_query, limit):
        return []

    @staticmethod
    def create_sql_columns_from_schema(schema):
        return []


class FakeFileStorage(FileStorage):
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    async def upload(
        self,
        *,
        file,
        file_filename,
        file_content_type,
        file_dir="",
    ) -> str:
        path = f"{file_dir}{file_filename}"
        self.files[path] = file.read()
        return path

    async def delete(self, full_path: str) -> bool:
        return self.files.pop(full_path, None) is not None

    async def read_bytes(self, full_path: str) -> bytes:
        return self.files[full_path]


def make_ingestion_file(
    *,
    filename: str = "report.txt",
    content_type: str = "text/plain",
    file_bytes: bytes = b"hello",
    description: str | None = None,
    source_id: str | None = None,
) -> IngestionFile:
    """Build an IngestionFile, optionally pinning its auto-generated source_id."""
    kwargs = dict(
        filename=filename,
        content_type=content_type,
        file_bytes=file_bytes,
        description=description,
    )
    if source_id is not None:
        kwargs["source_id"] = source_id
    return IngestionFile(**kwargs)  # type: ignore


def make_context(
    *,
    filename: str = "report.txt",
    content_type: str = "text/plain",
    source_bytes: bytes = b"hello",
    source_id: str = "source-1",
    parent_source_id: str | None = None,
    origin_source_id: str | None = None,
    source_description: str | None = None,
    parent_file: IngestionFile | None = None,
    origin_file: IngestionFile | None = None,
) -> IngestionContext:
    file = make_ingestion_file(
        filename=filename,
        content_type=content_type,
        file_bytes=source_bytes,
        description=source_description,
        source_id=source_id,
    )

    if parent_file is None and parent_source_id is not None:
        parent_file = make_ingestion_file(source_id=parent_source_id)

    # Default the origin: an explicit origin wins, otherwise the parent (in a
    # two-level chain the parent is the top-most file), otherwise the file
    # itself.
    if origin_file is None:
        if origin_source_id is not None:
            origin_file = make_ingestion_file(source_id=origin_source_id)
        elif parent_file is not None:
            origin_file = parent_file
        else:
            origin_file = file

    return IngestionContext(
        file=file,
        origin_file=origin_file,
        parent_file=parent_file,
    )


def build_runtime(
    context: IngestionContext,
    *,
    registry: PluginRegistry | None = None,
    llm: LLMProvider | None = None,
    embedder: Embedder | None = None,
    vector_storage: VectorStorage | None = None,
    sql_storage: SqlStorage | None = None,
    file_storage: FileStorage | None = None,
) -> SimpleNamespace:
    """Build an IngestionRuntime with inspectable fakes.

    Returns a namespace with `runtime` plus each fake for assertions.
    """
    parts = SimpleNamespace(
        registry=registry if registry is not None else PluginRegistry(),
        llm=llm if llm is not None else FakeLLM(),
        embedder=embedder if embedder is not None else FakeEmbedder(),
        vector_storage=(
            vector_storage if vector_storage is not None else FakeVectorStorage()
        ),
        sql_storage=sql_storage if sql_storage is not None else FakeSqlStorage(),
        file_storage=file_storage if file_storage is not None else FakeFileStorage(),
    )

    parts.runtime = IngestionRuntime(
        context=context,
        registry=parts.registry,
        llm=parts.llm,
        embedder=parts.embedder,
        vector_storage=parts.vector_storage,
        sql_storage=parts.sql_storage,
        file_storage=parts.file_storage,
    )

    return parts


def make_text_element(
    text: str,
    page_number: int | None = None,
):
    element = NarrativeText(text=text)
    if page_number is not None:
        element.metadata = ElementMetadata(page_number=page_number)
    return element


def make_table_element(
    html: str,
    page_number: int | None = None,
):
    element = Table(text="placeholder")
    metadata = ElementMetadata(text_as_html=html)
    if page_number is not None:
        metadata = ElementMetadata(
            text_as_html=html,
            page_number=page_number,
        )
    element.metadata = metadata
    return element


def make_image_element(
    image_bytes: bytes,
    page_number: int | None = None,
    caption: str = "",
    mime_type: str = "image/jpeg",
):
    import base64

    element = Image(text=caption)
    metadata = ElementMetadata(
        image_base64=base64.b64encode(image_bytes).decode("utf-8"),
        image_mime_type=mime_type,
    )
    if page_number is not None:
        metadata = ElementMetadata(
            image_base64=metadata.image_base64,
            image_mime_type=mime_type,
            page_number=page_number,
        )
    element.metadata = metadata
    return element


def make_page_break_element(page_number: int | None = None):
    element = PageBreak(text="PageBreak")
    if page_number is not None:
        element.metadata = ElementMetadata(page_number=page_number)
    return element


def make_footer_element(text: str, page_number: int | None = None):
    element = Footer(text=text)
    if page_number is not None:
        element.metadata = ElementMetadata(page_number=page_number)
    return element


def make_tool(name, output="tool result", calls=None):
    """A stub agent tool: records its invocations and returns fixed text."""

    class FakeTool(AgentTool):
        @property
        def name(self) -> str:
            return name

        @property
        def description(self) -> str:
            return f"{name} tool."

        @property
        def parameters(self) -> dict:
            return {"type": "object", "properties": {}}

        def create_executor(self, rag_service):
            async def execute(arguments):
                if calls is not None:
                    calls.append((name, arguments))
                return output

            return execute

    return FakeTool()


def build_rag_service(
    llm, retriever, *, sql_storage=None, file_storage=None, vector_storage=None
):
    """A real `RagService` over fake collaborators, for agent tests.

    Building the real service (instead of a stand-in) keeps the agent's
    dependency typed as `RagService`; pass real storages to exercise the
    tools against them.
    """
    return RagService(
        llm=llm,
        embedder=FakeEmbedder(),
        retriever=retriever,
        vector_storage=(
            vector_storage if vector_storage is not None else FakeVectorStorage()
        ),
        sql_storage=sql_storage if sql_storage is not None else FakeSqlStorage(),
        file_storage=file_storage if file_storage is not None else FakeFileStorage(),
    )


def tool_call_response(name: str, arguments: dict | None = None) -> RawResult:
    """A scripted RawResult in which the model requests one tool call."""
    return RawResult(
        content="",
        tool_calls=(ToolCall(id="c1", name=name, arguments=arguments or {}),),
    )


async def drain(stream) -> list:
    """Collect an async iterator into a list (for assertions)."""
    return [item async for item in stream]
