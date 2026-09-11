from app.agent.tools import AgentTool
from app.agent_tools.common import Retrieve
from app.agent_tools.inspect_table import make_inspect_table_tool
from app.agent_tools.list_tables import make_list_tables_tool
from app.agent_tools.query_chunks import make_query_chunks_tool
from app.agent_tools.query_table import make_query_table_tool
from app.agent_tools.search_documents import make_search_documents_tool
from app.store_file.base import FileStorage
from app.store_sql.base import SqlStorage

__all__ = [
    "Retrieve",
    "build_rag_tools",
    "make_inspect_table_tool",
    "make_list_tables_tool",
    "make_query_chunks_tool",
    "make_query_table_tool",
    "make_search_documents_tool",
]


def build_rag_tools(
    retrieve: Retrieve,
    sql_storage: SqlStorage,
    file_storage: FileStorage,
) -> list[AgentTool]:
    """The built-in RAG tools. New tools are added here (or a custom set is
    passed to the agent).

    `retrieve` must be the same retrieval path the RAG service exposes
    (plugins' finalization included), so the agent sees exactly the chunks
    the service reports.
    """
    return [
        make_search_documents_tool(retrieve),
        make_list_tables_tool(sql_storage),
        make_inspect_table_tool(sql_storage, file_storage),
        make_query_table_tool(sql_storage, file_storage),
        make_query_chunks_tool(sql_storage),
    ]
