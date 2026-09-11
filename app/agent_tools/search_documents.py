from app.agent.tools import AgentTool
from app.agent_tools.common import Retrieve, object_schema


def make_search_documents_tool(retrieve: Retrieve) -> AgentTool:
    """Search the ingested corpus for evidence (hybrid dense + lexical).

    Returns the ranked chunks' text and descriptions only — the agent works
    from that, not from whole files.
    """

    async def execute(arguments: dict) -> str:
        query = str(arguments.get("query", ""))
        chunks = list(await retrieve(query))
        if not chunks:
            return "No evidence found."

        lines = []
        for rank, chunk in enumerate(chunks, start=1):
            lines.append(
                f"[{rank}] source={chunk.source_id} chunk={chunk.chunk_id} "
                f"plugin={chunk.plugin}\n{chunk.text}"
            )
        return "\n\n".join(lines)

    return AgentTool(
        name="search_documents",
        description=(
            "Search the ingested document corpus for evidence relevant to a "
            "query. Returns ranked chunks, each with its source id, chunk id, "
            "and text (for tables, the text is the table's description and a "
            "sample, not the full data)."
        ),
        parameters=object_schema(
            {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                }
            },
            ["query"],
        ),
        execute=execute,
    )
