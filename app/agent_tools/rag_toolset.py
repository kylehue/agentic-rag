from app.agent_tools.base import AgentToolset
from app.agent_tools.perform_sql_to_document import PerformSqlToDocumentTool
from app.agent_tools.perform_sql_to_document_records import (
    PerformSqlToDocumentRecordsTool,
)
from app.agent_tools.search_documents import SearchDocumentTool
from app.agent_tools.validate_chunk_as_structured_data import (
    ValidateChunkAsStructuredDataTool,
)
from app.agent_tools.view_images import ViewImagesTool

# The RAG tools, declared once. They are stateless (their executors are built
# per answer), so sharing these instances is safe.
SEARCH_DOCUMENT_TOOL = SearchDocumentTool()
VALIDATE_CHUNK_AS_STRUCTURED_DATA_TOOL = ValidateChunkAsStructuredDataTool()
PERFORM_SQL_TO_DOCUMENT_TOOL = PerformSqlToDocumentTool()
PERFORM_SQL_TO_DOCUMENT_RECORDS_TOOL = PerformSqlToDocumentRecordsTool()
VIEW_IMAGES_TOOL = ViewImagesTool()

# The cross-tool orchestration for the RAG tools. Guidance, not a required
# sequence: the model picks what it needs and skips tools it already has. The
# identifier note lives here (not in the tools) because source_id is shared
# across them and is where the model otherwise confuses the file/table name.
RAG_TOOLSET_INSTRUCTIONS = f"""How to work with these tools (guidance, not a required sequence):
- Identifiers: the `source_id` you pass to these tools is the opaque id shown as `source_id=` in the search results (a UUID-like string). It is not the file name and not a table name. Table names (used in SQL) are separate and come from the source's table schema(s).
- To find relevant evidence, use `{SEARCH_DOCUMENT_TOOL.name}`. For a question about document text, that is usually enough to answer.
- When the question needs computation over tabular data (totals, filtering, joins): use `{SEARCH_DOCUMENT_TOOL.name}` to find the relevant source and note its `source_id`, then `{VALIDATE_CHUNK_AS_STRUCTURED_DATA_TOOL.name}` to see that source's table name(s) and columns, then `{PERFORM_SQL_TO_DOCUMENT_TOOL.name}` to run the SQL. A multi-sheet workbook's sheets are separate tables you can JOIN in one `{PERFORM_SQL_TO_DOCUMENT_TOOL.name}` call.
- `{PERFORM_SQL_TO_DOCUMENT_RECORDS_TOOL.name}` runs read-only SQL over the stored chunk and document records; use it to inspect what is stored, not to answer content questions. Only use this as a last resort. Ideally, you only need `{SEARCH_DOCUMENT_TOOL.name}` tool to search for document records.
- When a `{SEARCH_DOCUMENT_TOOL.name}` result is an image, use `{VIEW_IMAGES_TOOL.name}` with its `source_id` to see what the image shows before answering about it.
- Don't re-run a tool when you already have what you need from an earlier turn or the conversation history."""

RAG_TOOLSET = AgentToolset(
    name="RAG",
    tools=[
        SEARCH_DOCUMENT_TOOL,
        VALIDATE_CHUNK_AS_STRUCTURED_DATA_TOOL,
        PERFORM_SQL_TO_DOCUMENT_TOOL,
        PERFORM_SQL_TO_DOCUMENT_RECORDS_TOOL,
        VIEW_IMAGES_TOOL,
    ],
    instructions=RAG_TOOLSET_INSTRUCTIONS,
)
