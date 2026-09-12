from app.agent_tools.base import AgentTool
from app.agent_tools.common import object_schema


class SearchDocumentTool(AgentTool):
    """Search the ingested corpus for evidence (hybrid dense + lexical)."""

    @property
    def name(self) -> str:
        return "search_documents"

    @property
    def description(self) -> str:
        return (
            "Search the ingested document corpus for evidence relevant to a "
            "query. Returns ranked chunks, each with its source id, chunk id, "
            "and text (for tables, the text is the table's description and a "
            "sample, not the full data)."
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

    def create_executor(self, rag_service):
        async def execute(arguments: dict) -> str:
            query = str(arguments.get("query", ""))
            chunks = list(await rag_service.retrieve(query))
            if not chunks:
                return "No evidence found."

            lines = []
            for rank, chunk in enumerate(chunks, start=1):
                lines.append(
                    f"[{rank}] source={chunk.source_id} chunk={chunk.chunk_id} "
                    f"{chunk.text}"
                )
            return "\n\n".join(lines)

        return execute
