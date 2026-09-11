from app.agent.tools import AgentTool
from app.agent_tools.common import (
    SCHEMA_INFER_ROWS,
    dataframe_schema,
    format_schema,
    object_schema,
    read_sheet_sample,
    resolve_table_file,
    resolve_table_row,
)
from app.store_file.base import FileStorage
from app.store_sql.base import SqlStorage


def make_inspect_table_tool(
    sql_storage: SqlStorage, file_storage: FileStorage
) -> AgentTool:
    """Show a table's schema and shape.

    Reads the schema from the chunk's stored metadata — no file read. Only
    falls back to a bounded file peek when the metadata predates schema
    storage. The table's description is not shown here; it is in the chunk's
    text, which `search_documents` returns.
    """

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
            # Last resort: older tables have no stored schema; infer it from a
            # bounded peek at the sheet rather than loading the whole file.
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

    return AgentTool(
        name="inspect_table",
        description=(
            "Read one stored table's structure without loading its data: its "
            "shape and column names with types (schema). Use it to understand "
            "a table before writing a query_table SQL statement. The table's "
            "description is not shown here; find it via search_documents. If "
            "several sources store a table with the same name, pass the "
            "source_id (from list_tables)."
        ),
        parameters=object_schema(
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
        ),
        execute=execute,
    )
