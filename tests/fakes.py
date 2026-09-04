from types import SimpleNamespace

from unstructured.documents.elements import ElementMetadata, NarrativeText, Table

from app.embedders.base import Embedder
from app.llm.base import LLMProvider
from app.plugin.context import IngestionContext
from app.plugin.hooks import HookBus
from app.plugin.runtime import IngestionRuntime
from app.store_file.base import FileStorage
from app.store_sql.base import SqlStorage
from app.store_vector.base import VectorStorage


class FakeLLM(LLMProvider):
    """Text-only LLM stub returning a fixed response."""

    def __init__(self, response: str = "") -> None:
        self.response = response
        self.prompts: list[str] = []

    async def answer(self, query: str) -> str:
        self.prompts.append(query)
        return self.response


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


def make_context(
    *,
    filename: str = "report.txt",
    content_type: str = "text/plain",
    source_bytes: bytes = b"hello",
    elements=None,
    source_id: str = "source-1",
    parent_source_id: str | None = None,
) -> IngestionContext:
    return IngestionContext(
        source_id=source_id,
        source_filename=filename,
        source_content_type=content_type,
        source_bytes=source_bytes,
        elements=elements or [],
        parent_source_id=parent_source_id,
    )


def build_runtime(
    context: IngestionContext,
    *,
    hooks: HookBus | None = None,
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
        hooks=hooks if hooks is not None else HookBus(),
        llm=llm if llm is not None else FakeLLM(),
        embedder=embedder if embedder is not None else FakeEmbedder(),
        vector_storage=vector_storage
        if vector_storage is not None
        else FakeVectorStorage(),
        sql_storage=sql_storage
        if sql_storage is not None
        else FakeSqlStorage(),
        file_storage=file_storage
        if file_storage is not None
        else FakeFileStorage(),
    )

    parts.runtime = IngestionRuntime(
        context=context,
        hooks=parts.hooks,
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
