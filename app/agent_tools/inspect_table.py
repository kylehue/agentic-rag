from app.agent_tools.base import AgentTool
from app.agent_tools.common import (
    CHUNK_ID_DESCRIPTION,
    SCHEMA_INFER_ROWS,
    SOURCE_ID_DESCRIPTION,
    dataframe_schema,
    format_schema,
    object_schema,
    read_sheet_sample,
    resolve_table,
    resolve_table_file,
)


class InspectTableTool(AgentTool):
    """Show a table's schema and shape."""

    @property
    def name(self) -> str:
        return "inspect_table"

    @property
    def description(self) -> str:
        return (
            "Read one stored table's structure without loading its data: its "
            "shape and column names with types (schema). Address the table "
            "by its source id. Use it to understand a table before writing "
            "a SQL statement. The table's description is not shown here."
        )

    @property
    def parameters(self) -> dict:
        return object_schema(
            {
                "source_id": {"type": "string", "description": SOURCE_ID_DESCRIPTION},
                "chunk_id": {"type": "string", "description": CHUNK_ID_DESCRIPTION},
            },
            ["source_id"],
        )

    def create_executor(self, rag_service, chat_id=None):
        sql_storage = rag_service.sql_storage
        file_storage = rag_service.file_storage

        async def execute(arguments: dict) -> str:
            source_id = str(arguments.get("source_id", ""))
            raw_chunk_id = arguments.get("chunk_id")
            chunk_id = str(raw_chunk_id) if raw_chunk_id else None

            row = await resolve_table(sql_storage, source_id, chunk_id, chat_id)
            metadata = row.get("metadata") or {}
            name = metadata.get("table_name") or source_id
            schema = metadata.get("schema")
            row_count = metadata.get("row_count")
            column_count = metadata.get("column_count")

            if not schema:
                # Last resort: infer it from a bounded peek at the sheet
                # rather than loading the whole file.
                file_path, file_bytes, sheet = await resolve_table_file(
                    sql_storage, file_storage, source_id, chunk_id, chat_id
                )
                sample = read_sheet_sample(
                    sheet, file_path, file_bytes, SCHEMA_INFER_ROWS
                )
                schema = dataframe_schema(sample)
                if row_count is None:
                    # Exact only when the peek read every row.
                    row_count = len(sample) if len(sample) < SCHEMA_INFER_ROWS else None
            if column_count is None:
                column_count = len(schema)

            shape_parts = []
            if row_count is not None:
                shape_parts.append(f"{row_count} rows")
            shape_parts.append(f"{column_count} columns")
            parts = [f"Table '{name}' ({' x '.join(shape_parts)})"]
            parts.append("Columns:")
            parts.append(format_schema(schema))
            return "\n".join(parts)

        return execute
