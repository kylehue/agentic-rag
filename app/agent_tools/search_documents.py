from app.agent_tools.base import AgentTool
from app.agent_tools.common import object_schema
from app.models.chunk import RetrievedTextChunk


class SearchDocumentTool(AgentTool):
    """Search the ingested corpus for evidence (hybrid dense + lexical)."""

    @property
    def name(self) -> str:
        return "search_documents"

    @property
    def description(self) -> str:
        return (
            "Search the ingested document corpus for evidence relevant to a "
            "query. Returns ranked chunks; each result lists its source_id, "
            "chunk_id, origin_source_id, plugin, and the chunk text (image "
            "chunks have no text). For a table chunk, the text is the table's "
            "description and a small sample, not its full data."
        )

    @property
    def parameters(self) -> dict:
        return object_schema(
            {
                "query": {
                    "type": "string",
                    "description": (
                        "A detailed, comprehensive search query optimized for RAG retrieval. "
                        "Do NOT use vague keywords or a simple restatement of the user's question. "
                        "Instead, synthesize a descriptive sentence or paragraph containing the exact technical "
                        "terms, potential synonyms, acronyms, and core semantic concepts likely to be physically "
                        "written inside the matching document chunks."
                    ),
                }
            },
            ["query"],
        )

    def create_executor(self, rag_service, context):
        async def execute(arguments: dict) -> str:
            query = str(arguments.get("query", ""))
            chunks = list(await rag_service.retrieve(query, context.chat_id))
            if not chunks:
                return "No evidence found."

            lines = []
            for chunk in chunks:
                # Register the chunk so the answer can cite it by number. The
                # ids are still shown: the model passes source_id to the other
                # table tools (validate / perform_sql_to_document).
                index = context.evidence.register(
                    chunk.origin_source_id, chunk.chunk_id
                )
                header = (
                    f"[{index}] source_id={chunk.source_id} "
                    f"chunk_id={chunk.chunk_id} "
                    f"origin_source_id={chunk.origin_source_id} "
                    f"plugin={chunk.plugin}"
                )
                if isinstance(chunk, RetrievedTextChunk):
                    lines.append(f"{header}\n{chunk.text}")
                else:
                    # Image chunks carry no text.
                    lines.append(header)
            return "\n\n".join(lines)

        return execute
