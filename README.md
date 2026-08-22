# Multimodal RAG

### Architecture

![rag architecture](./assets/architecture.png)

### Document Chunk

#### Processors (Chunkers)

- When `file_bytes`, `file_content_type`, and `file_filename` is provided, it will store the chunk file in the file database. This will attach the `chunk_file_path` to chunk metadata.
- `text` is the one used for dense/sparse search and LLM final answer context.
- `metadata` contains the chunk-specific configurations.
- `category` is used to route the document elements into processor pipeline.

#### Metadata

- Keys that are prefixed with **"chunk\_"** will be saved to the SQL database.
- `sql_schema`, `sql_table_name`, and `sql_rows` is used to save spreadsheet chunks' table data into the SQL database.
- The file in `chunk_file_path` is given to the LLM for the final answer if `chunk_attach_file_to_llm` is set to `True`.
