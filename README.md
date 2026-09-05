# Plugin RAG Server

A Retrieval-Augmented Generation (RAG) backend built around a **plugin architecture** for document types. It ingests text documents and spreadsheets, creates searchable representations, retrieves relevant evidence using both semantic and lexical search, and uses a text-only LLM to generate grounded answers.

The application is designed around separate ingestion, retrieval, and answer-generation stages. Document-type behavior lives in **plugins**, which the system discovers and orchestrates through a registry, a hook bus, and per-run contexts and runtimes.

## Table of Contents

- [Plugin RAG Server](#plugin-rag-server)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
    - [Plugin Architecture](#plugin-architecture)
    - [Semantic and Lexical Retrieval](#semantic-and-lexical-retrieval)
    - [Spreadsheet Semantics](#spreadsheet-semantics)
  - [Running the Application](#running-the-application)
    - [Start the Application](#start-the-application)
    - [Requirements](#requirements)
    - [Environment Configuration](#environment-configuration)
  - [Architecture](#architecture)
    - [Plugin Architecture](#plugin-architecture-1)
    - [Ingestion Pipeline](#ingestion-pipeline)
    - [Retrieval Pipeline](#retrieval-pipeline)
    - [Answer Generation](#answer-generation)
  - [Hooks](#hooks)
  - [Context and Runtime](#context-and-runtime)
  - [File Emission and Subprocess](#file-emission-and-subprocess)
  - [Project Structure](#project-structure)
    - [`app/plugin`](#appplugin)
    - [`app/plugins`](#appplugins)
    - [`app/api`](#appapi)
    - [`app/api_schemas`](#appapi_schemas)
    - [`app/core`](#appcore)
    - [`app/errors`](#apperrors)
    - [`app/llm`](#appllm)
    - [`app/embedders`](#appembedders)
    - [`app/retrievers`](#appretrievers)
    - [`app/models`](#appmodels)
    - [`app/services`](#appservices)
    - [`app/store_file`, `app/store_sql`, `app/store_vector`](#appstore_file-appstore_sql-appstore_vector)
    - [`app/utils`](#apputils)
    - [`app/container.py`](#appcontainerpy)
  - [Design Principles](#design-principles)
  - [Document Chunk Conventions](#document-chunk-conventions)
  - [Tests](#tests)
  - [Extending the Application](#extending-the-application)
    - [Adding Another Plugin](#adding-another-plugin)
    - [Adding Another LLM Provider](#adding-another-llm-provider)
    - [Adding Another Embedding Provider](#adding-another-embedding-provider)
    - [Adding Another Retrieval Strategy](#adding-another-retrieval-strategy)
    - [Changing Storage Technology](#changing-storage-technology)

## Features

### Plugin Architecture

Each supported document type is a **plugin** — identity plus a set of hook handlers. The pipeline is purely hook-driven: the services emit hooks, and plugins react. A plugin:

- decides which files it manages (`accepts(context)`),
- **generates chunks** by handling the `ingestion_process` hook (returning `IngestedChunk`s; it never persists them),
- **emits files** through the runtime for subprocess by other plugins,
- **finalizes its own chunks** by handling the `retrieval_finalize` hook during retrieval/answering.

All database and file persistence is owned by the services — plugins only produce data and react to events.

Built-in plugins:

- `TextPlugin` — text documents (pdf, docx, txt, md, html, ...).
- `TablePlugin` — spreadsheets (csv, xlsx, xls) and emitted table files. Generates one chunk per table with LLM-generated semantic context and the table's CSV file.

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

Each table becomes one searchable chunk. The actual table data is **stored as a file** (CSV per table), not as SQL rows. Table content is expected to be queried through agentic tools in the future; the chunk file and its metadata (`table_name`, `schema`, `file_path`) are the source of truth for that.

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
- LLM API access (either Gemini or OpenAI)
- A configured embedding model

Some Unstructured document formats may also require additional system dependencies.

### Environment Configuration

Create a `.env` file in the root directory containing the configuration required by the application.

At minimum, the Gemini provider requires a Google API key.

The application configuration in `app/core/config.py` is the source of truth for the available settings.

Example:

```env
GOOGLE_API_KEY=your-api-key
OPENAI_API_KEY=your-api-key
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

| Piece                                        | Role                                                                                                                                                                                                                                                                                                                                                                            |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Plugin` (`app/plugin/base.py`)              | The contract: `name`, `accepts`. No pipeline methods — all behavior is hook handlers, declared with the `@hook(name)` decorator.                                                                                                                                                                                                                                     |
| `IngestionContext` (`app/plugin/context.py`) | Read-only details of the document being ingested: `source_id`, `source_filename`, `source_content_type`, `source_bytes`, `parent_source_id`.                                                                                                                                                                                                                        |
| `IngestionRuntime` (`app/plugin/runtime.py`) | Per-ingestion-run state, created by the ingestion service: `context`, `llm`, `hooks`, `emit_file`. No storage — the service persists everything.                                                                                                                                                                                                                                |
| `RetrievalRuntime` (`app/plugin/runtime.py`) | Per-retrieval-run state, created by the retrieval service: `query`, `llm`, `hooks`.                                                                                                                                                                                                                                                                                             |
| `HookBus` + `@hook` (`app/plugin/hooks.py`)  | The event bus that drives the pipeline, typed per hook. Hook names are plain wire strings; the built-in names are typed with `Literal` overloads on `@hook` and `emit` (misspelling one is a type error), and each hook has a `TypedDict` payload and a handler type. Some hooks are observational, others are response hooks whose handler return values the services consume. |
| `PluginRegistry` (`app/plugin/registry.py`)  | Holds the registered plugins, discovers their `@hook`-decorated methods and wires them into the bus, and offers each document to all plugins via `accepts`.                                                                                                                                                                                                                     |

The plugin framework lives in `app/plugin/`; the built-in plugins live in `app/plugins/`. The container (`app/container.py`) is the composition root that builds the hook bus, registry, plugins, and services.

The **ingestion service owns all persistence** — plugins only generate chunks and emit files.

### Ingestion Pipeline

The `IngestionService` drives the pipeline through hooks:

1. Build the `IngestionContext` for the file (no type resolution — the system does not judge what a file is).
2. Offer the file to **every registered plugin** via `plugin.accepts(context)`. Each plugin decides for itself whether it wants to manage that file (the built-ins claim their file extensions; a plugin may claim as many or as few document shapes as it likes). If no plugin accepts the file, ingestion fails with `InvalidDocumentError`.
3. Build a per-run `IngestionRuntime` and store the source file (File DB + `__documents__` row).
4. Emit the **`ingestion_process`** hook (`context`, `runtime`). Every plugin's handler that accepts the file generates chunks — and may call `runtime.emit_file` along the way — and returns them. Any parsing a plugin needs is done inside the handler (e.g. `TextPlugin` parses with Unstructured.io itself — the service never parses). The service collects all returned chunks and persists them:
    - chunk text is embedded and stored in the **Vector DB**,
    - the chunk record (including any chunk file in the **File DB**) is stored in the **SQL DB** (`__chunks__`), with the stored file path and lineage stamped into the chunk's metadata,
    - chunk metadata is saved **as-is** — plugins define their own keys.
5. Drain the files the plugins emitted (`runtime.take_emitted_files()`). Each emitted file is re-ingested through the same pipeline (`ingest_bytes`) with a fresh `source_id` and a `parent_source_id` pointing back at the document that emitted it. The returned chunks are combined with the parent's.
6. Emit lifecycle hooks throughout (`ingestion_started`, `file_emitted`, `file_subprocessed`, `ingestion_completed`) — every payload carries the per-run `IngestionRuntime`.

For text documents, `TextPlugin` parses the source with Unstructured.io and chunks the text. Emitting embedded tables as CSV files (to be subprocessed by `TablePlugin`) is disabled for now pending better table-aware chunking — see the TODO in `TextPlugin._process`. The emission mechanism itself remains fully available on the runtime for any plugin.

### Retrieval Pipeline

The retrieval pipeline starts with a **user query** and performs two retrieval strategies in parallel through the **Hybrid Retriever**.

The **Vector Retriever** embeds the query, searches the Vector DB, and receives ranked chunk IDs. Those IDs are then used to retrieve the corresponding chunk records from the SQL DB.

At the same time, the **Sparse Retriever** performs lexical search against the SQL/FTS database and produces its own ranked list of chunks.

The two ranked result sets are then combined using **Reciprocal Rank Fusion (RRF)**. RRF combines the rankings rather than the raw scores, allowing dense and sparse retrieval to contribute to a common ranking even though their scoring systems are different.

The service then builds a per-run **`RetrievalContext`** (holding `user_query`) and **`RetrievalRuntime`** (`context`, `llm`, `hooks`) and offers each resulting chunk to the **`retrieval_finalize`** hook (`context`, `chunk`, `runtime`). Plugins that want query-aware enrichment handle this hook, recognize their own chunks via `chunk.plugin`, and return an enriched replacement (or `None` to leave the chunk unchanged). Plugins without a finalize handler do no work.

### Answer Generation

After finalization, the **RAG Service** assembles the finalized chunks into evidence and sends the user query plus evidence to the (text-only) LLM to generate a grounded answer.

## Hooks

The pipeline is driven entirely by named hooks. Hook names are plain wire strings (`"ingestion_process"`, `"retrieval_finalize"`, ...). The built-in names are typed with `Literal` overloads, so misspelling one is a type error; any other string works for custom hooks. Plugins tap hooks by decorating handler methods with `@hook("hook_name")`; the registry discovers the decorated methods and wires them into the shared `HookBus` when the plugin is registered. Handlers receive their hook's **typed payload** as a single argument and run concurrently.

The decorator also enforces the hook's signature: annotate the payload parameter with the hook's payload type (e.g. `payload: IngestionProcessPayload`) and the return type with what the hook expects — a mismatch is a type error. Annotate it to get a fully typed handler body:

```python
@hook("ingestion_process")
async def _process(
    self,
    payload: IngestionProcessPayload,
) -> list[IngestedChunk]:
    context = payload["context"]
    runtime = payload["runtime"]
    ...
```

Every hook payload carries the runtime of its phase — ingestion hooks carry `IngestionRuntime`, retrieval hooks carry `RetrievalRuntime`.

Two hooks are **response hooks** — the service consumes what handlers return:

- `ingestion_process` — handlers return the `IngestedChunk`s they generated (or nothing).
- `retrieval_finalize` — handlers return a replacement `RetrievedChunk` for chunks they produced, or `None` to leave the chunk unchanged.

All other hooks are observational.

| Hook                  | Payload                              | Returns                           |
| --------------------- | ------------------------------------ | --------------------------------- |
| `ingestion_started`   | `context`, `runtime`                 | —                                 |
| `ingestion_process`   | `context`, `runtime`                 | `Sequence[IngestedChunk] \| None` |
| `file_emitted`        | `context`, `emitted_file`, `runtime` | —                                 |
| `file_subprocessed`   | `emitted_file`, `chunks`, `runtime`  | —                                 |
| `ingestion_completed` | `context`, `chunks`, `runtime`       | —                                 |
| `retrieval_finalize`  | `chunk`, `runtime`                   | `RetrievedChunk \| None`          |
| `retrieval_completed` | `chunks`, `runtime`                  | —                                 |

Example — a plugin that generates chunks and audits ingestion:

```python
class AuditPlugin(Plugin):
    @hook("ingestion_process")
    async def on_process(self, payload: IngestionProcessPayload) -> list[IngestedChunk]:
        if not self.accepts(payload["context"]):
            return []
        return [self.make_chunk(payload["context"])]  # the service persists these

    @hook("ingestion_completed")
    async def on_completed(self, payload: IngestionCompletedPayload) -> None:
        print(f"[audit] {payload['context'].source_filename} -> {len(payload['chunks'])} chunks")
```

Custom hooks work the same way: decorate a handler with any name and emit it through the bus (`runtime.hooks.emit("my.event", {...})`).

## Context and Runtime

Plugins never talk to storage or providers directly, and their constructors take **options only — no services**. Hook handlers receive:

- **`IngestionContext`** — what is being ingested: `source_id`, `source_filename`, `source_content_type`, `source_bytes`, and `parent_source_id` (set for subprocessed files).
- **`RetrievalContext`** — what is being retrieved: `user_query`. It is the retrieval-phase counterpart of `IngestionContext` and is passed in every retrieval hook payload.
- **`IngestionRuntime`** — the only per-run state and action a plugin gets:

| Member                                          | Purpose                                                   |
| ----------------------------------------------- | --------------------------------------------------------- |
| `context`                                       | The ingestion context for this run.                       |
| `llm`                                           | The text-only LLM provider.                               |
| `hooks`                                         | The shared hook bus (plugins may emit their own events).  |
| `emit_file(filename, content_type, file_bytes)` | Emit a file for subprocess by the plugins that accept it. |

There are deliberately no save/embed/vector actions — the ingestion service persists everything the plugins return. When storing chunk records, it stamps the storage path (`file_path`) and lineage (`parent_source_id`) into the chunk's metadata, which is otherwise saved **as-is** with no key conventions.

On the retrieval side, the `RetrievalService` builds a per-run **`RetrievalContext`** (holding `user_query`) and a **`RetrievalRuntime`** exposing `context`, `llm`, and `hooks`. Both are passed in every retrieval hook payload.

## File Emission and Subprocess

Plugins can emit files so embedded content is processed as its own document:

1. A plugin calls `runtime.emit_file(filename, content_type, file_bytes)` while handling the `ingestion_process` hook.
2. After the hook completes, the `IngestionService` takes all emitted files and runs each one back through `ingest_bytes` — full pipeline, fresh `source_id`, with `parent_source_id` set to the emitting document.
3. The emitted file is offered to all plugins again, and the one(s) that accept it process it.

The intended use is a text document with embedded tables: `TextPlugin` would convert each parsed table to CSV and emit it, and `TablePlugin` would generate semantic chunks for each CSV (the service stores the table file).

## Project Structure

### `app/plugin`

The plugin framework: `Plugin` contract, `IngestionContext` / `EmittedFile`, `IngestionRuntime` / `RetrievalRuntime`, the `HookBus` + `@hook` decorator with per-hook typed payloads, and `PluginRegistry`.

### `app/plugins`

Built-in document-type plugins. Each module contains one `Plugin` subclass.

#### `app/plugins/text.py`

`TextPlugin` — accepts text document extensions.

Parses the source with Unstructured.io (the only place in the app that uses it) and chunks the text (`chunk_by_title` when titles are present, otherwise `chunk_elements`), returning the chunks from its `ingestion_process` handler. (Table emission is disabled for now — see the TODO in `_process`.)

#### `app/plugins/table.py`

`TablePlugin` — accepts spreadsheet extensions.

Reads CSV/XLS/XLSX with pandas (one table per sheet), normalizes table and column names, builds a row-sampled catalog, asks the LLM for one workbook-level semantic analysis (descriptions, roles, relationships), and returns one chunk per table from its `ingestion_process` handler. Each chunk carries its own CSV file and `table_name` / `schema` metadata. Table rows are not loaded into SQL.

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

- `app/services/ingestion.py` — the pipeline driver: offers each file to all plugins, builds context/runtime, fires the `ingestion_process` hook, persists everything (source, chunk files, chunk records, vectors), and re-ingests emitted files.
- `app/services/retrieval.py` — runs the hybrid retriever and offers each chunk to the `retrieval_finalize` hook.
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

- **The pipeline is the hook bus.** Services emit hooks; plugins respond with data (chunks, replacement chunks) or reactions. There are no plugin `process`/`finalize` methods and no per-type dispatch in the system.
- **Plugins own document types.** A plugin decides which files it wants to manage (`accepts`) and handles the ingestion/retrieval hooks for them. The system does not special-case document types and offers every file to all plugins.
- **Services own persistence.** Plugins generate and return data; only the services touch the databases and file storage.
- **Context and runtime, not globals.** Hook handlers receive everything they need per run: a read-only context (document details) plus a minimal runtime (shared services like the LLM, and `emit_file`). Plugin constructors take options only, never services.
- **Decoupled stores and providers.** Storage and AI providers sit behind small interfaces (`FileStorage`, `SqlStorage`, `VectorStorage`, `LLMProvider`, `Embedder`).
- **Text-only LLM.** No binary attachments anywhere in the pipeline.

## Document Chunk Conventions

- `IngestedChunk` fields: `plugin` (which plugin produced it, used for finalize routing), `text` (search + answer evidence), `metadata` (free-form, plugin-defined), and optional `file_filename` / `file_content_type` / `file_bytes` (stored as a chunk file by the service).
- **There are no metadata key conventions.** The ingestion service persists a chunk's `metadata` as-is. Keys are defined by the plugin that produced the chunk (e.g. `TablePlugin` uses `table_name` and `schema`, intended for future agentic table querying).
- The service stamps a couple of storage/lineage keys onto the metadata when persisting: `file_path` (where the chunk file was stored) and `parent_source_id` (for subprocessed documents).

## Tests

The test suite lives in `tests/` and uses pytest.

```bash
python -m pytest tests
```

The suite covers the hook bus, plugin registry acceptance, both built-in plugins (with fakes for LLM/embedding/storage), the hook-driven retrieval finalization, and end-to-end ingestion pipelines using real local SQLite and file storage with a stubbed Unstructured partition.

## Extending the Application

### Adding Another Plugin

1. Create a module in `app/plugins/` with a `Plugin` subclass. The constructor may take plugin options (tunables, thresholds) but never services — the LLM is available on `runtime.llm` in both phases.
2. Give it a unique `name` and implement `accepts(context)` — decide which files this plugin wants to manage. A plugin may accept as many document shapes as it likes; multiple plugins may accept the same file and will all run.
3. Decorate a method with `@hook("ingestion_process")` to generate `IngestedChunk`s (set `plugin=self.name` on each so retrieval can route them back; emit embedded content with `runtime.emit_file`). Do any parsing you need inside the handler — the service never parses for you. Return the chunks — the service persists them.
4. Optionally decorate a method with `@hook("retrieval_finalize")` for query-aware enrichment: return a replacement chunk for chunks where `chunk.plugin == self.name`, or `None` otherwise.
5. Register it in `app/container.py`.

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
