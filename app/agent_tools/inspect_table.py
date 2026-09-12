from app.agent_tools.base import AgentTool
from app.agent_tools.common import (
    SCHEMA_INFER_ROWS,
    dataframe_schema,
    format_schema,
    object_schema,
    read_sheet_sample,
    resolve_table_file,
    resolve_table_row,
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
            "shape and column names with types (schema). Use it to understand "
            "a table before writing a SQL statement. The table's description "
            "is not shown here."
        )

    @property
    def parameters(self) -> dict:
        return object_schema(
            {
                "table": {
                    "type": "string",
                    "description": "The table name, as shown by list_tables.",
                },
                "source_id": {
                    "type": "string",
                    "description": (
                        "Disambiguate when several sources store a table with "
                        "this name; as shown by list_tables."
                    ),
                },
            },
            ["table"],
        )

    def create_executor(self, rag_service):
        sql_storage = rag_service.sql_storage
        file_storage = rag_service.file_storage

        async def execute(arguments: dict) -> str:
            table_name = str(arguments.get("table", ""))
            raw_source_id = arguments.get("source_id")
            source_id = str(raw_source_id) if raw_source_id else None

            row = await resolve_table_row(sql_storage, table_name, source_id)
            metadata = row.get("metadata") or {}
            schema = metadata.get("schema")
            row_count = metadata.get("row_count")
            column_count = metadata.get("column_count")

            if not schema:
                # Last resort: infer it from a bounded peek at the sheet
                # rather than loading the whole file.
                file_path, file_bytes = await resolve_table_file(
                    sql_storage, file_storage, table_name, source_id
                )
                sample = read_sheet_sample(
                    table_name, file_path, file_bytes, SCHEMA_INFER_ROWS
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
            parts = [f"Table '{table_name}' ({' x '.join(shape_parts)})"]
            parts.append("Columns:")
            parts.append(format_schema(schema))
            return "\n".join(parts)

        return execute
