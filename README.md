# Plugin RAG Server

A Retrieval-Augmented Generation (RAG) backend built around a **plugin architecture** for document types. It ingests text documents and spreadsheets, creates searchable representations, retrieves relevant evidence using both semantic and lexical search, and uses a text-only LLM to generate grounded answers.

The application is designed around separate ingestion, retrieval, and answer-generation stages. Document-type behavior lives in **plugins**, which the system discovers and orchestrates through a registry, a hook bus, and per-run contexts and runtimes.

## Table of Contents

- [Features](#features)
- [Running the Application](#running-the-application)
- [Architecture](#architecture)
  - [Plugin Architecture](#plugin-architecture)
  - [Ingestion Pipeline](#ingestion-pipeline)
  - [Retrieval Pipeline](#retrieval-pipeline)
  - [Answer Generation](#answer-generation)
- [Hooks](#hooks)
- [Context and Runtime](#context-and-runtime)
- [File Emission and Subprocess](#file-emission-and-subprocess)
- [Project Structure](#project-structure)
- [Design Principles](#design-principles)
- [Document Chunk Conventions](#document-chunk-conventions)
- [Tests](#tests)
- [Extending the Application](#extending-the-application)

## Features

### Plugin Architecture

Each supported document type is a **plugin**. During ingestion the system offers every file to all registered plugins, and each plugin accepts or rejects the file for itself via `accepts(context)`. A plugin bundles both sides of the document types it claims:

- **Ingestion** (`process`): turns the source document into chunks and persists them.
- **Retrieval** (`finalize`): enriches retrieved chunks before they are used as evidence.

Built-in plugins:

- `TextPlugin` — text documents (pdf, docx, txt, md, html, ...). Chunks parsed text and emits embedded tables as CSV files.
- `TablePlugin` — spreadsheets (csv, xlsx, xls) and emitted table files. Creates one chunk per table with LLM-generated semantic context and stores the table's CSV file.

Adding a new document type means adding one plugin and registering it — no changes to the ingestion or retrieval services.

### Semantic and Lexical Retrieval

The retrieval pipeline combines two complementary strategies:

- **Dense retrieval** using vector embeddings and ChromaDB
- **Sparse retrieval** using SQL-based full-text search (FTS5)

The results are merged using **Reciprocal Rank Fusion (RRF)** to produce a combined ranking.

This allows the system to handle both semantic queries and queries that depend on exact terms, names, or values.

### Spreadsheet Semantics

Spreadsheets are read with pandas and analyzed once with the LLM:

- Workbook meaning
- Table descriptions
- Column semantics
- Relationships between tables

Each table becomes one searchable chunk. The actual table data is **stored as a file** (CSV per table), not as SQL rows. Table content is expected to be queried through agentic tools in the future; the chunk file and its metadata (`chunk_table_name`, `chunk_schema`, `chunk_file_path`) are the source of truth for that.

### Text-Only LLM

The `LLMProvider` interface is text-only: no binary attachments. The system processes text documents and spreadsheets; image handling is intentionally out of scope.

## Running the Application

### Start the Application

```bash
python -m run
```

### Requirements

The application requires:

- Python 3.11+
- FastAPI
- Unstructured.io
- SQLAlchemy
- SQLite
- ChromaDB
- Google Gemini API access
- A configured embedding model
- An LLM model supported by the configured Gemini provider

Some Unstructured document formats may also require additional system dependencies.

### Environment Configuration

Create a `.env` file in the root directory containing the configuration required by the application.

At minimum, the Gemini provider requires a Google API key.

The application configuration in `app/core/config.py` is the source of truth for the available settings.

Example:

```env
GOOGLE_API_KEY=your-api-key
```

Additional settings may control:

- Database locations
- Vector database locations
- Storage directories
- Collection names
- Retrieval limits
- Other application options

## Architecture

![rag architecture](./assets/architecture.png)

The architecture is divided into two independent pipelines: **ingestion** and **retrieval**. Both pipelines meet at the document chunks stored across the application's databases, and both are driven by **plugins**.

### Plugin Architecture

The plugin system lives in `app/plugin`:

| Piece | Role |
| --- | --- |
| `Plugin` (`app/plugin/base.py`) | The contract: `name`, `uses_elements`, `hooks`, `accepts`, `process`, `finalize`. |
| `IngestionContext` (`app/plugin/context.py`) | Read-only details of the document being ingested: `source_id`, `source_filename`, `source_content_type`, `source_bytes`, parsed `elements`, `parent_source_id`. |
| `IngestionRuntime` (`app/plugin/runtime.py`) | Per-ingestion actions and shared services a plugin can use: `llm`, `embed`, `add_vectors`, `save_chunk_file`, `save_chunk_metadata`, `save_chunks`, `save_chunk`, `save_source`, `emit_file`. |
| `FinalizeRuntime` (`app/plugin/runtime.py`) | Per-retrieval actions available to a plugin's finalizer: `llm`, `hooks`. |
| `HookBus` (`app/plugin/hooks.py`) | Named event bus the system emits on; plugins tap in via `Plugin.hooks`. |
| `PluginRegistry` (`app/plugin/registry.py`) | Holds the registered plugins, offers each document to all of them via `accepts`, wires plugin hooks into the bus, and routes retrieved chunks to the finalizer of the plugin that produced them (matched by `chunk.plugin`). |

The plugin framework lives in `app/plugin/`; the built-in plugins live in `app/plugins/`. The container (`app/container.py`) is the composition root that builds the hook bus, registry, plugins, and services.

A plugin is responsible for persisting the chunks it produces — the ingestion service never saves chunks itself.

### Ingestion Pipeline

The `IngestionService` is a thin orchestrator:

1. Build the `IngestionContext` for the file (no type resolution — the system does not judge what a file is).
2. Offer the file to **every registered plugin** via `plugin.accepts(context)`. Each plugin decides for itself whether it wants to manage that file (the built-ins claim their file extensions; a plugin may claim as many or as few document shapes as it likes). If no plugin accepts the file, ingestion fails with `InvalidDocumentError`.
3. If any accepting plugin declares `uses_elements`, parse the document with Unstructured.io once and populate `context.elements` (spreadsheets skip parsing entirely).
4. Build a per-run `IngestionRuntime` and store the source file (`save_source`).
5. Run `process(context, runtime)` on **every accepting plugin**, in registration order. Each plugin produces chunks and persists them through the runtime:
   - chunk text is embedded and stored in the **Vector DB**,
   - `chunk_*` metadata is stored in the **SQL DB** (`__chunks__`),
   - optional chunk files are stored in the **File DB** (`chunk_files/`).
6. Drain files the plugins emitted (`runtime.take_emitted_files()`). Each emitted file is re-ingested through the same pipeline (`ingest_bytes`) with a fresh `source_id` and a `parent_source_id` pointing back at the document that emitted it. The returned chunks are combined with the parent's.
7. Emit lifecycle hooks throughout (`ingestion.started`, `ingestion.plugins_selected`, `chunk.saved`, `file.emitted`, `file.subprocessed`, `ingestion.completed`).

For text documents, `TextPlugin` chunks the parsed text and emits every embedded table as a CSV file. Those CSVs are subprocessed by `TablePlugin`, which creates semantic table chunks and stores the CSV file for each table.

### Retrieval Pipeline

The retrieval pipeline starts with a **user query** and performs two retrieval strategies in parallel through the **Hybrid Retriever**.

The **Vector Retriever** embeds the query, searches the Vector DB, and receives ranked chunk IDs. Those IDs are then used to retrieve the corresponding chunk records from the SQL DB.

At the same time, the **Sparse Retriever** performs lexical search against the SQL/FTS database and produces its own ranked list of chunks.

The two ranked result sets are then combined using **Reciprocal Rank Fusion (RRF)**. RRF combines the rankings rather than the raw scores, allowing dense and sparse retrieval to contribute to a common ranking even though their scoring systems are different.

The resulting ranked chunks are routed through the `PluginRegistry` to the **owning plugin's finalizer** (`plugin.finalize(query, chunk, finalize_runtime)`). The default finalizer is a no-op, so only plugins that need query-aware enrichment do any work.

### Answer Generation

After finalization, the **RAG Service** assembles the finalized chunks into evidence and sends the user query plus evidence to the (text-only) LLM to generate a grounded answer.

## Hooks

The system emits named hooks at lifecycle points. Plugins tap hooks by returning a mapping of hook name to async handler from `Plugin.hooks`; the registry wires them into the shared `HookBus` when the plugin is registered.

| Hook | Payload |
| --- | --- |
| `ingestion.started` | `context` |
| `ingestion.plugins_selected` | `context`, `plugins` |
| `chunk.saved` | `context`, `chunk` |
| `file.emitted` | `context`, `emitted_file` |
| `file.subprocessed` | `emitted_file`, `chunks` |
| `ingestion.completed` | `context`, `chunks` |
| `chunk.finalized` | `query`, `chunk` |
| `retrieval.completed` | `query`, `chunks` |

Example — a plugin that logs every saved chunk:

```python
class AuditPlugin(Plugin):
    @property
    def hooks(self):
        async def on_chunk_saved(context, chunk, **_):
            print(f"[audit] saved {chunk.plugin} chunk for {context.source_filename}")

        return {"chunk.saved": on_chunk_saved}
```

Hook handlers receive the payload as keyword arguments and run concurrently.

## Context and Runtime

Plugins never talk to storage or providers directly, and their constructors take **options only — no services**. They receive:

- **`IngestionContext`** — what is being ingested: `source_id`, `source_filename`, `source_content_type`, `source_bytes`, the parsed Unstructured `elements`, and `parent_source_id` (set for subprocessed files).
- **`IngestionRuntime`** — what the plugin may do:

| Action | Purpose |
| --- | --- |
| `embed(texts)` | Embed texts into vectors. |
| `add_vectors(chunk_ids, embeddings)` | Store chunk vectors in the vector DB. |
| `save_chunk_file(chunk)` | Store the chunk's file in the File DB and record `chunk_file_path` / `chunk_file_filename` / `chunk_file_content_type` in chunk metadata. |
| `save_chunk_metadata(chunk)` | Persist the chunk's `chunk_*` metadata to the SQL DB. |
| `save_chunk(chunk)` / `save_chunks(chunks)` | Full chunk persistence: embed, vectorize, store file, store metadata, emit `chunk.saved`. |
| `save_source(...)` | Store the source file in the File DB and record a `__documents__` row (used by the service, not plugins). |
| `emit_file(filename, content_type, file_bytes)` | Emit a file for subprocess by its owning plugin. |

The runtime also exposes shared services directly: `runtime.llm` (text-only) and `runtime.hooks`.

`save_chunks` also stamps `chunk_source_id` (and `chunk_parent_source_id` for subprocessed files) into chunk metadata, so plugins do not have to repeat source bookkeeping.

On the retrieval side, finalizers receive a **`FinalizeRuntime`** exposing `llm` and `hooks`.

## File Emission and Subprocess

Plugins can emit files so embedded content is processed as its own document:

1. A plugin calls `runtime.emit_file(filename, content_type, file_bytes)` during `process`.
2. After `process` returns, the `IngestionService` takes all emitted files and runs each one back through `ingest_bytes` — full pipeline, fresh `source_id`, with `parent_source_id` set to the emitting document.
3. The emitted file is offered to all plugins again, and the one(s) that accept it process it (`TextPlugin` emits `.csv` files, which `TablePlugin` accepts).

This is how a text document with embedded tables is processed: `TextPlugin` converts each parsed table to CSV and emits it; `TablePlugin` ingests each CSV, creates semantic chunks, and stores the table file. Subprocess depth is bounded (`MAX_SUBPROCESS_DEPTH`) to prevent runaway emission loops.

## Project Structure

### `app/plugin`

The plugin framework: `Plugin` contract, `IngestionContext` / `EmittedFile`, `IngestionRuntime` / `FinalizeRuntime`, `HookBus` and hook names, and `PluginRegistry`.

### `app/plugins`

Built-in document-type plugins. Each module contains one `Plugin` subclass.

#### `app/plugins/text.py`

`TextPlugin` — accepts text document extensions.

Splits parsed elements into text and tables, emits each table as a CSV file, chunks the remaining text with Unstructured (`chunk_by_title` when titles are present, otherwise `chunk_elements`), and persists the chunks through the runtime.

#### `app/plugins/table.py`

`TablePlugin` — accepts spreadsheet extensions.

Reads CSV/XLS/XLSX with pandas (one table per sheet), normalizes table and column names, builds a row-sampled catalog, asks the LLM for one workbook-level semantic analysis (descriptions, roles, relationships), and produces one chunk per table. Each chunk carries its own CSV file and `chunk_table_name` / `chunk_schema` metadata. Table rows are not loaded into SQL.

### `app/api`

Contains the FastAPI HTTP routes.

#### `app/api/rag.py`

Contains the RAG endpoints:

- `/rag/ingest`: uploads and ingests a document.
- `/rag/retrieve`: retrieves relevant chunks without generating a final answer.
- `/rag/answer`: performs retrieval and generates the final LLM answer.

### `app/api_schemas`

Contains Pydantic models used specifically at the API boundary. These are separate from the application's internal dataclasses so internal objects can contain fields that should not be exposed through JSON responses.

### `app/core`

Application-wide configuration and core infrastructure.

- `app/core/config.py` — settings (API keys, model names, storage paths, collection names).
- `app/core/exceptions.py` — exception handlers registered on the FastAPI app.

### `app/errors`

Domain-specific errors (`InvalidDocumentError` for unsupported/invalid uploads).

### `app/llm`

- `app/llm/base.py` — the text-only `LLMProvider` interface.
- `app/llm/gemini.py` — the Gemini implementation.

### `app/embedders`

- `app/embedders/base.py` — the `Embedder` interface.
- `app/embedders/gemini.py` — the Gemini embedding implementation.

### `app/retrievers`

- `app/retrievers/base.py` — the `Retriever` interface.
- `app/retrievers/vector.py` — dense retrieval via Chroma, re-loading chunk records from SQL.
- `app/retrievers/sparse.py` — lexical retrieval via FTS5.
- `app/retrievers/hybrid.py` — runs retrievers concurrently and merges rankings with RRF.

### `app/models`

Internal application models.

- `app/models/chunk.py` — `IngestedChunk` and `RetrievedChunk`. Both carry `plugin` (the name of the plugin that produced the chunk), which is how retrieval routes chunks back to their finalizer.
- `app/models/rag.py` — `RagAnswer`.
- `app/models/vector.py` — `VectorSearchResult`.

### `app/services`

- `app/services/ingestion.py` — the plugin orchestrator: offers each file to all plugins, context/runtime construction, source saving, emission subprocess, lifecycle hooks.
- `app/services/retrieval.py` — runs the hybrid retriever and routes chunks to their plugin's finalizer.
- `app/services/rag.py` — top-level facade for the API: ingest, retrieve, answer.

### `app/store_file`, `app/store_sql`, `app/store_vector`

Storage abstractions and local implementations:

- `FileStorage` — upload/delete/read files. Local implementation stores under `.storage/file` (`documents/`, `chunk_files/`).
- `SqlStorage` — tables, upserts, read-only queries, FTS5 search. Local implementation is one SQLite database (`.storage/sqlite/spreadsheets.sqlite3/database.db`) holding the `__chunks__` and `__documents__` system tables.
- `VectorStorage` — stores chunk IDs + embeddings, cosine search. Local implementation is a persistent Chroma collection.

### `app/utils`

- `app/utils/ranking.py` — RRF ranking.
- `app/utils/string.py` — brace-safe prompt template rendering.

### `app/container.py`

The composition root. It builds the hook bus, plugin registry, built-in plugins, providers, storages, retrievers, and services, and exposes the FastAPI lifespan (table initialization, storage shutdown).

## Design Principles

- **Plugins own document types.** A plugin decides which files it wants to manage (`accepts`) and bundles both ingestion (`process`) and retrieval (`finalize`) for them. The system does not special-case document types and offers every file to all plugins.
- **Services orchestrate; plugins act.** The `IngestionService` resolves, emits hooks, and subprocesses. It never persists chunks — plugins do, through the runtime.
- **Context and runtime, not globals.** Plugins receive everything they need per run: read-only context (document details) plus a runtime of actions and shared services (like the LLM). Plugin constructors take options only, never services.
- **Hooks for cross-cutting behavior.** Plugins tap system lifecycle points instead of the system calling plugin hooks.
- **Decoupled stores and providers.** Storage and AI providers sit behind small interfaces (`FileStorage`, `SqlStorage`, `VectorStorage`, `LLMProvider`, `Embedder`).
- **Text-only LLM.** No binary attachments anywhere in the pipeline.

## Document Chunk Conventions

- `IngestedChunk` fields: `plugin` (which plugin produced it, used for finalizer routing), `text` (search + answer evidence), `metadata` (chunk configuration), and optional `file_filename` / `file_content_type` / `file_bytes` (stored as a chunk file by the runtime).
- Only metadata keys prefixed with **`chunk_`** are persisted to the `__chunks__` table.
- Common persisted keys:
  - `chunk_source_id` / `chunk_parent_source_id` — source lineage (stamped by the runtime).
  - `chunk_source_page_number` — originating page, when known.
  - `chunk_file_path` / `chunk_file_filename` / `chunk_file_content_type` — the chunk's stored file (set by `save_chunk_file`).
  - `chunk_table_name` / `chunk_schema` — table identity and schema (set by `TablePlugin`), intended for future agentic table querying.

## Tests

The test suite lives in `tests/` and uses pytest.

```bash
python -m pytest tests
```

The suite covers the hook bus, plugin registry routing and acceptance, both built-in plugins (with fakes for LLM/embedding/storage), and end-to-end ingestion pipelines using real local SQLite and file storage with a stubbed Unstructured partition.

## Extending the Application

### Adding Another Plugin

1. Create a module in `app/plugins/` with a `Plugin` subclass. The constructor may take plugin options (tunables, thresholds) but never services — the LLM is available on `runtime.llm` during ingestion and `FinalizeRuntime.llm` during retrieval.
2. Implement `accepts(context)` — decide which files this plugin wants to manage. A plugin may accept as many document shapes as it likes; multiple plugins may accept the same file and will all run.
3. Declare `uses_elements` when you require parsed elements, and any `hooks` you want to tap. Stamps `plugin=self.name` on every chunk it produces so retrieval can route the chunk back to this plugin's `finalize`.
4. Implement `process` — produce `IngestedChunk`s and persist them with `runtime.save_chunks` (emit files with `runtime.emit_file` for embedded content of other types).
5. Implement `finalize` only if the type needs query-aware enrichment.
6. Register it in `app/container.py`.

### Adding Another LLM Provider

Implement `LLMProvider` (text-only) and provide the new implementation to the container.

### Adding Another Embedding Provider

Implement `Embedder` without changing retrieval or ingestion.

### Adding Another Retrieval Strategy

Implement `Retriever` and include it in the `HybridRetriever`. The RRF implementation combines its ranked results with the others.

### Changing Storage Technology

The storage interfaces allow the current local implementations to be replaced with other backends.

For example:

```text
LocalSqlStorage    -> PostgreSQL implementation
LocalFileStorage   -> S3 bucket
LocalVectorStorage -> Alternative vector database
```

The services and plugins remain unchanged as long as the replacement implements the corresponding interface.
