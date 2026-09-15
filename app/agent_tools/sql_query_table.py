from app.agent_tools.base import AgentTool
from app.agent_tools.common import (
    MAIN_TABLE_NAME,
    QUERY_TABLE_MAX_ROWS,
    SOURCE_ID_DESCRIPTION,
    load_csv_table,
    object_schema,
    resolve_csv_table,
    run_table_sql,
    validate_table_aliases,
)


class SqlQueryTableTool(AgentTool):
    """Run a read-only sqlite query against one csv table (and related ones)."""

    @property
    def name(self) -> str:
        return "sql_query_table"

    @property
    def description(self) -> str:
        return (
            "Run a read-only SQL query (sqlite dialect) against one stored "
            "csv table. The table is loaded as `data` in a throw-away "
            "in-memory database; related csv tables can be loaded for JOINs "
            "via table_relationships (for example "
            "SELECT * FROM data JOIN related ON data.id = related.data_id "
            "with table_relationships {\"related\": \"<related source_id>\"}). "
            "Write targeted SQL (specific columns, WHERE filters, LIMIT, and "
            "aggregates); at most the first "
            f"{QUERY_TABLE_MAX_ROWS} result rows are returned."
        )

    @property
    def parameters(self) -> dict:
        return object_schema(
            {
                "source_id": {"type": "string", "description": SOURCE_ID_DESCRIPTION},
                "sql": {
                    "type": "string",
                    "description": (
                        "A read-only SQL SELECT query (sqlite dialect) over "
                        f"`{MAIN_TABLE_NAME}` and the related-table aliases."
                    ),
                },
                "table_relationships": {
                    "type": "object",
                    "description": (
                        "Optional map of table alias -> related table's "
                        "source_id. Each related csv table is loaded under "
                        "its alias so the query can JOIN it."
                    ),
                },
            },
            ["source_id", "sql"],
        )

    def create_executor(self, rag_service, chat_id=None):
        sql_storage = rag_service.sql_storage
        file_storage = rag_service.file_storage

        async def execute(arguments: dict) -> str:
            source_id = str(arguments.get("source_id", ""))
            sql = str(arguments.get("sql", ""))
            relationships = arguments.get("table_relationships") or {}
            if not isinstance(relationships, dict) or not all(
                isinstance(alias, str) and isinstance(related, str)
                for alias, related in relationships.items()
            ):
                raise ValueError(
                    "table_relationships must map table aliases to source ids."
                )
            validate_table_aliases(list(relationships))

            main_row, main_document = await resolve_csv_table(
                sql_storage, source_id, chat_id
            )
            dataframes = {
                MAIN_TABLE_NAME: await load_csv_table(file_storage, main_document)
            }
            for alias, related_source_id in relationships.items():
                _related_row, related_document = await resolve_csv_table(
                    sql_storage, related_source_id, chat_id
                )
                dataframes[alias] = await load_csv_table(file_storage, related_document)

            result = run_table_sql(dataframes, sql)

            total_rows = len(result)
            truncated = total_rows > QUERY_TABLE_MAX_ROWS
            csv = result.head(QUERY_TABLE_MAX_ROWS).to_csv(index=False).rstrip()
            note = (
                f"\n(showing the first {QUERY_TABLE_MAX_ROWS} of {total_rows} rows)"
                if truncated
                else ""
            )
            related_note = (
                f"; related tables loaded as: {', '.join(relationships)}"
                if relationships
                else ""
            )
            name = (main_row.get("metadata") or {}).get("table_name") or source_id
            return (
                f"Result of the SQL query against table '{name}' (loaded as "
                f"`{MAIN_TABLE_NAME}`{related_note}):\n{csv}{note}"
            )

        return execute
