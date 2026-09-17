from app.agent_tools.base import AgentToolset
from app.agent_tools.inspect_table import InspectTableTool
from app.agent_tools.inspect_table_relationships import InspectTableRelationshipsTool
from app.agent_tools.search_documents import SearchDocumentTool
from app.agent_tools.sql_query_documents import SqlQueryDocumentsTool
from app.agent_tools.sql_query_table import SqlQueryTableTool

# The RAG tools, declared once. They are stateless (their executors are built
# per answer), so sharing these instances is safe.
SEARCH_DOCUMENT_TOOL = SearchDocumentTool()
INSPECT_TABLE_TOOL = InspectTableTool()
INSPECT_TABLE_RELATIONSHIPS_TOOL = InspectTableRelationshipsTool()
SQL_QUERY_TABLE_TOOL = SqlQueryTableTool()
SQL_QUERY_DOCUMENTS_TOOL = SqlQueryDocumentsTool()

# The cross-tool orchestration for the RAG tools
RAG_TOOLSET_INSTRUCTIONS = f"""How to work with these tools:
- Before anything else, consider starting with `{SEARCH_DOCUMENT_TOOL.name}` tool first. This is the retrieval tool of the RAG pipeline. Start with it if you need to find the relevant documents and to discover which tables exist and their source ids. Don't guess a source id; take it from the results.
- For a document table's data (e.g. spreadsheet or csv documents), call `{INSPECT_TABLE_TOOL.name}` to see its schema.
- It is possible that a document table has relationships with other tables. Think of an excel workbook with multiple sheets. When you think the question spans several tables, call `{INSPECT_TABLE_RELATIONSHIPS_TOOL.name}` to find the related tables, then join them in one `{SQL_QUERY_TABLE_TOOL.name}` call.
- After inspecting a document's table, you can use `{SQL_QUERY_TABLE_TOOL.name}` to actually perform SQL queries for answers that need more specifics.
- Only use `{SQL_QUERY_DOCUMENTS_TOOL.name}` as a last resort - when you need something that can't be searched with the normal RAG retrieval tool. The correct tool for searching documents is `{SEARCH_DOCUMENT_TOOL.name}`."""

RAG_TOOLSET = AgentToolset(
    name="RAG",
    tools=[
        SEARCH_DOCUMENT_TOOL,
        INSPECT_TABLE_TOOL,
        INSPECT_TABLE_RELATIONSHIPS_TOOL,
        SQL_QUERY_TABLE_TOOL,
        SQL_QUERY_DOCUMENTS_TOOL,
    ],
    instructions=RAG_TOOLSET_INSTRUCTIONS,
)
