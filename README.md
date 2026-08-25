# Multimodal RAG

A multimodal Retrieval-Augmented Generation (RAG) backend that ingests documents, creates searchable representations, retrieves relevant evidence using both semantic and lexical search, optionally performs structured SQL queries over spreadsheet data, and uses an LLM to generate grounded answers.

The application is designed around separate ingestion, retrieval, finalization, and answer-generation stages so that each part of the RAG pipeline can be replaced or extended independently.

## Table of Contents

- [Multimodal RAG](#multimodal-rag)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
    - [Multimodal Document Ingestion](#multimodal-document-ingestion)
    - [Semantic and Lexical Retrieval](#semantic-and-lexical-retrieval)
    - [Query-Aware Document Finalization](#query-aware-document-finalization)
    - [Spreadsheet Reasoning](#spreadsheet-reasoning)
    - [Image Understanding](#image-understanding)
  - [Running the Application](#running-the-application)
    - [Start the Application](#start-the-application)
    - [Requirements](#requirements)
    - [Environment Configuration](#environment-configuration)
  - [Architecture](#architecture)
    - [Ingestion Pipeline](#ingestion-pipeline)
    - [Retrieval Pipeline](#retrieval-pipeline)
    - [Finalization](#finalization)
    - [Answer Generation](#answer-generation)
  - [Spreadsheet Data Model](#spreadsheet-data-model)
    - [Search Representation](#search-representation)
    - [Query Representation](#query-representation)
  - [File Attachments](#file-attachments)
  - [Project Structure](#project-structure)
    - [`app/api`](#appapi)
    - [`app/api_schemas`](#appapi_schemas)
    - [`app/core`](#appcore)
    - [`app/errors`](#apperrors)
    - [`app/llm`](#appllm)
    - [`app/embedders`](#appembedders)
    - [`app/processors`](#appprocessors)
    - [`app/retrievers`](#appretrievers)
    - [`app/finalizers`](#appfinalizers)
    - [`app/models`](#appmodels)
    - [`app/services`](#appservices)
    - [`app/store_file`](#appstore_file)
    - [`app/store_sql`](#appstore_sql)
    - [`app/store_vector`](#appstore_vector)
    - [`app/utils`](#apputils)
    - [`app/container.py`](#appcontainerpy)
  - [Design Principles](#design-principles)
  - [Document Chunk Conventions](#document-chunk-conventions)
    - [Processors (Chunkers)](#processors-chunkers)
    - [Metadata](#metadata)
  - [Extending the Application](#extending-the-application)
    - [Adding Another LLM Provider](#adding-another-llm-provider)
    - [Adding Another Embedding Provider](#adding-another-embedding-provider)
    - [Adding Another Document Processor](#adding-another-document-processor)
    - [Adding Another Retrieval Strategy](#adding-another-retrieval-strategy)
    - [Adding Another Finalizer](#adding-another-finalizer)
    - [Changing Storage Technology](#changing-storage-technology)

## Features

### Multimodal Document Ingestion

The application supports multiple document categories:

- PDF and other text-based documents
- CSV and Excel spreadsheets
- Standalone images
- Images embedded inside documents

Documents are partitioned using Unstructured.io and routed to modality-specific processors.

### Semantic and Lexical Retrieval

The retrieval pipeline combines two complementary strategies:

- **Dense retrieval** using vector embeddings and ChromaDB
- **Sparse retrieval** using SQL-based full-text search

The results are merged using **Reciprocal Rank Fusion (RRF)** to produce a combined ranking.

This allows the system to handle both semantic queries and queries that depend on exact terms, names, or values.

### Query-Aware Document Finalization

Retrieved chunks can be further processed before they are sent to the final LLM.

For example, spreadsheet chunks can trigger SQL execution when the user's question requires calculations or access to the underlying rows.

Finalization is therefore performed only when the retrieved evidence requires additional processing.

### Spreadsheet Reasoning

Spreadsheets are stored in SQL tables in addition to being indexed as searchable chunks.

The application uses the LLM to analyze:

- Workbook meaning
- Table descriptions
- Column semantics
- Column relationships
- Relationships between tables

When a question requires structured computation, the Spreadsheet Finalizer can generate a read-only SQL query using the retrieved schema and relationship information.

This allows questions involving aggregation, filtering, grouping, calculations, and table joins to be answered using the actual spreadsheet rows rather than only sampled data.

### Image Understanding

Images are converted into searchable textual descriptions using the LLM.

The original image can also be preserved as a file and attached to the final LLM request when the retrieved chunk requests it.

This allows the final model to reason from both the generated description and the original visual content.

## Running the Application

### Start the Application

```bash
python -m run
```

### Requirements

The application requires:

- Python 3.11
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

Create a `.env` file in root directory containing the configuration required by the application.

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

The architecture is divided into two independent pipelines: **ingestion** and **retrieval**. Both pipelines meet at the document chunks stored across the application's databases.

### Ingestion Pipeline

The ingestion pipeline converts an uploaded document into searchable and reusable representations.

The process begins by partitioning the document with **Unstructured.io**, which produces a collection of document elements. These elements are passed to the **Processor Pipeline**, which routes them to the appropriate processor based on the document type.

For normal documents, the **Text Processor** chunks the extracted content. If embedded images are encountered, they are extracted and passed to the **Image Processor** so they can be independently described and retrieved.

For spreadsheets, the **Spreadsheet Processor** extracts the actual table data and uses the LLM to generate semantic descriptions, schemas, and relationships between tables.

For images, the **Image Processor** sends the image to the LLM to generate a searchable description.

All processors ultimately produce **document chunks**. These chunks are then persisted in multiple forms:

- Their text is embedded and stored in the **Vector DB** for dense semantic retrieval.
- Their metadata is stored in the **SQL DB**.
- Spreadsheet rows are stored as SQL tables so they can later be queried directly by the LLM.
- Optional chunk files are stored in the **File DB**, while the original uploaded document and its metadata are also preserved.

This gives each source multiple representations optimized for different stages of the RAG pipeline.

### Retrieval Pipeline

The retrieval pipeline starts with a **user query** and performs two retrieval strategies in parallel through the **Hybrid Retriever**.

The **Vector Retriever** embeds the query, searches the Vector DB, and receives ranked chunk IDs. Those IDs are then used to retrieve the corresponding chunk records from the SQL DB.

At the same time, the **Sparse Retriever** performs lexical search against the SQL/FTS database and produces its own ranked list of chunks.

The two ranked result sets are then combined using **Reciprocal Rank Fusion (RRF)**. RRF combines the rankings rather than the raw scores, allowing dense and sparse retrieval to contribute to a common ranking even though their scoring systems are different.

The resulting ranked chunks are passed to the **Finalizer Pipeline**.

### Finalization

Finalization occurs after retrieval and allows the system to enrich each retrieved chunk based on the user's query.

The **Finalizer Pipeline** routes retrieved chunks to the appropriate finalizer. Text and image chunks can pass through unchanged, while specialized chunk types can trigger additional processing.

For spreadsheet chunks, the **Spreadsheet Finalizer** determines whether the user's question requires querying the underlying spreadsheet data. When necessary, it asks the LLM to generate a single read-only SQL query, executes the query against the corresponding SQL tables, and appends the generated SQL and its results to the chunk's `text`.

This allows retrieval to provide the relevant spreadsheet context first, while finalization performs more precise row-level operations only when the query requires them.

### Answer Generation

After finalization, the resulting chunks are assembled into the final LLM context.

Chunks can also request that their associated files be attached to the LLM. When `chunk_attach_file_to_llm` is enabled, the system resolves the chunk's file and includes it as a binary attachment. This allows the LLM to inspect the original file content in addition to the retrieved text.

The **RAG Service** then combines the user's query, finalized chunk context, and any required file attachments into the final LLM request. The LLM generates the grounded answer using this combined evidence.

## Spreadsheet Data Model

Spreadsheet sources have two representations.

### Search Representation

The generated chunk contains semantic information such as:

- Table description
- Column descriptions
- Schema
- Relationships
- Sampled data

This representation is embedded and indexed for retrieval.

### Query Representation

The actual table rows are stored in SQL.

This allows the application to distinguish between:

> "What kind of data is in this spreadsheet?"

and:

> "What is the average value for each category?"

The first can often be answered from retrieved chunk context alone, while the second can trigger a SQL query against the actual rows.

## File Attachments

Chunks can request that their associated file be supplied to the final LLM.

This is controlled using:

```text
chunk_attach_file_to_llm
```

When enabled, the application resolves the file associated with the chunk and supplies it to the LLM together with the textual evidence.

This is particularly useful for image chunks, where the generated textual description may not contain every detail visible in the original image.

If a chunk-specific file is unavailable, the application can fall back to the original source file.

## Project Structure

The application is organized into layers with clear responsibilities. API routes handle HTTP concerns, services orchestrate application workflows, processors transform documents into chunks, retrievers find relevant chunks, finalizers enrich retrieved chunks, and storage implementations handle persistence.

### `app/api`

Contains the FastAPI HTTP routes. This layer is responsible for receiving requests, validating API inputs, calling application services, and converting internal models into API response schemas.

#### `app/api/rag.py`

Contains the RAG endpoints:

- `/rag/ingest`: uploads and ingests a document.
- `/rag/retrieve`: retrieves relevant chunks without generating a final answer.
- `/rag/answer`: performs retrieval and generates the final LLM answer.

The routes delegate the actual work to `RagService` in `api/services/rag_service.py`.

### `app/api_schemas`

Contains Pydantic models used specifically at the API boundary. These are separate from the application's internal dataclasses so internal objects can contain fields that should not be exposed through JSON responses.

### `app/core`

Contains application-wide configuration and core infrastructure concerns.

#### `app/core/config.py`

Loads and exposes application configuration such as API keys, model names, storage paths, collection names, and other environment-dependent settings.

#### `app/core/exceptions.py`

Contains application-level exception definitions or exception handling infrastructure shared across the application.

#### `app/core/file_types.py`

Contains all of the supported file types and file categorization used to determine which processing pipeline a file should use.

### `app/errors`

Contains domain-specific errors.

### `app/llm`

Contains the LLM abstraction and provider implementations.

#### `app/llm/base.py`

Defines the `LLMProvider` interface used by processors, finalizers, and the RAG service.

#### `app/llm/gemini.py`

Implements the `LLMProvider` interface using Google's Gemini API.

### `app/embedders`

Contains abstractions and implementations for generating vector embeddings.

#### `app/embedders/base.py`

Defines the `Embedder` interface. Services depend on this abstraction rather than directly depending on a specific embedding provider.

#### `app/embedders/gemini.py`

Implements the `Embedder` interface using Google's Gemini embedding API.

### `app/processors`

Processors transform extracted source elements into application-specific `IngestedChunk` objects.

#### `app/processors/base.py`

Defines the `Processor` interface.

A processor receives a `ProcessorPayload` and returns a list of `IngestedChunk` objects.

#### `app/processors/hybrid.py`

Routes a `ProcessorPayload` to the appropriate processor based on its `ChunkCategory`.

For example:

```text
DOCUMENT     -> TextProcessor
SPREADSHEET  -> TableProcessor
IMAGE        -> ImageProcessor
```

This keeps the ingestion service independent of individual processor implementations.

#### `app/processors/image.py`

Processes images and creates image chunks.

It extracts image bytes, generates an LLM description using the image as an attachment, and stores the description in the chunk text so the image can participate in semantic retrieval.

The image can also be preserved as a chunk file for later attachment to the final-answer LLM.

#### `app/processors/table.py`

Processes spreadsheets and structured tables.

It uses pandas to read CSV/XLS/XLSX data, normalizes table and column names into SQL-friendly identifiers, infers SQL types from pandas dtypes, samples representative rows, and uses the LLM to generate semantic descriptions of tables, columns, and relationships.

Spreadsheet chunks contain SQL-related metadata used by ingestion to create and populate SQL tables.

#### `app/processors/text.py`

Processes general document text.

It uses Unstructured's chunking functionality to split extracted elements into document chunks while preserving their original elements.

It can also detect embedded images and send them to `ImageProcessor` to create separate searchable image chunks.

### `app/retrievers`

Contains retrieval strategies used to find relevant chunks for a user query.

#### `app/retrievers/base.py`

Defines the `Retriever` interface.

#### `app/retrievers/vector.py`

Performs dense semantic retrieval.

The user query is embedded, the vector store returns the nearest chunk IDs and similarity scores, and the corresponding chunk records are loaded from SQL storage.

#### `app/retrievers/sparse.py`

Performs sparse/lexical retrieval using the SQL storage's search implementation.

The SQLite implementation uses FTS5. Sparse retrieval primarily provides ranking (its absolute score is not required by RRF).

#### `app/retrievers/hybrid.py`

Runs multiple retrievers concurrently and combines their ranked results using Reciprocal Rank Fusion (RRF).

This allows dense semantic search and sparse lexical search to complement each other.

### `app/finalizers`

Finalizers run after retrieval and enrich retrieved chunks before they are provided to the final-answer LLM.

#### `app/finalizers/base.py`

Defines the `Finalizer` interface. A finalizer receives a user query and a retrieved chunk and returns the finalized chunk. A finalizer can attach additional content to a chunk's `text` property based on the user's query.

#### `app/finalizers/hybrid.py`

Coordinates multiple finalizers and applies them to retrieved chunks according to the application's finalization pipeline.

#### `app/finalizers/spreadsheet.py`

Handles spreadsheet-specific finalization.

It can determine whether the user query actually requires querying the underlying spreadsheet data. When necessary, it asks the LLM to generate a read-only SQL query, executes that query against the spreadsheet's SQL table, and appends the SQL query and results to the retrieved chunk's text.

### `app/models`

Contains internal application models and dataclasses.

#### `app/models/chunk.py`

Contains the application's chunk models.

`IngestedChunk` represents a chunk produced during ingestion, including its text, metadata, and source information.

`RetrievedChunk` represents a chunk after retrieval, including its retrieval score.

`ChunkCategory` identifies the type of source/chunk:

- `DOCUMENT`
- `SPREADSHEET`
- `IMAGE`

#### `app/models/ingestion.py`

Contains models used internally during ingestion and processing.

`ProcessorPayload` contains the source file and its extracted Unstructured elements that are passed through the processor pipeline.

### `app/services`

Contains application-level orchestration logic.

#### `app/services/ingestion.py`

Coordinates the entire ingestion pipeline.

#### `app/services/retrieval.py`

Coordinates the retrieval pipeline and invokes the configured retriever/finalizer components to produce the evidence used by the RAG answer stage.

#### `app/services/rag.py`

Provides the application's top-level RAG operations.

It coordinates ingestion, retrieval, file attachment handling, prompt construction, and final LLM answer generation.

### `app/store_file`

Defines the file-storage abstraction and its implementations.

#### `app/store_file/base.py`

Defines the `FileStorage` interface for uploading, retrieving, and deleting stored files.

#### `app/store_file/local.py`

Stores files on the local filesystem.

### `app/store_sql`

Defines the relational database abstraction and implementations.

#### `app/store_sql/base.py`

Defines the `SqlStorage` interface for:

- creating tables
- loading tables
- inserting/upserting rows
- retrieving rows
- deleting rows
- executing read-only SQL
- performing sparse search
- closing the database connection

#### `app/store_sql/local.py`

Implements SQL storage using SQLite and SQLAlchemy.

It provides normal relational tables for application and spreadsheet data and lazily creates SQLite FTS5 search structures when sparse search is requested.

### `app/store_vector`

Defines the vector database abstraction and implementations.

#### `app/store_vector/base.py`

Defines the `VectorStorage` interface used by the vector retriever and ingestion service.

#### `app/store_vector/local.py`

Implements vector storage using a local ChromaDB persistent collection.

It stores chunk IDs and embeddings and performs cosine-similarity vector searches.

### `app/utils`

Contains small reusable utilities.

### `app/container.py`

Acts as the application's dependency-composition root.

It creates and wires together the configured:

- LLM provider
- embedder
- file storage
- vector storage
- SQL storage
- processors
- retrievers
- finalizers
- application services

## Design Principles

The project follows a few important separation-of-concerns rules:

- **Processors** transform source content into chunks.
- **Embedders** generate embeddings.
- **Retrievers** find relevant chunks.
- **Finalizers** enrich retrieved chunks before the final answer.
- **Services** orchestrate application workflows.
- **Stores** handle persistence.
- **Providers** wrap external APIs such as Gemini.
- **API schemas** define the HTTP representation separately from internal models.
- **Models** represent internal application data rather than HTTP concerns.

## Document Chunk Conventions

### Processors (Chunkers)

- When `file_bytes`, `file_content_type`, and `file_filename` are provided, it will store the chunk file in the file database. This will attach the `chunk_file_path` to chunk metadata.
- `text` is the one used for dense/sparse search and LLM final answer context. It is also passed to the LLM by the finalizers along with the user's query to enrich its context.
- `category` is used to route the document elements into processor pipeline.
- `metadata` contains the chunk-specific configurations.

### Metadata

- Keys that are prefixed with **"chunk\_"** will be saved to the SQL database.
- `sql_schema`, `sql_table_name`, and `sql_rows` are used to save spreadsheet chunks' table data into the SQL database.
- If `chunk_attach_file_to_llm` is set to `True`, the file in `chunk_file_path` will be sent to LLM. If `chunk_file_path` is not found, it will fallback to the `file_path` of the chunk's source file.

## Extending the Application

The intended extension points are the interfaces defined throughout the application.

### Adding Another LLM Provider

Implement `LLMProvider` and provide the new implementation to the application's dependency container.

### Adding Another Embedding Provider

Implement `Embedder` without changing the retrieval or ingestion services.

### Adding Another Document Processor

Implement `Processor` and declare the supported `ChunkCategory` values.

The `HybridProcessor` can then route the appropriate source type to it.

### Adding Another Retrieval Strategy

Implement `Retriever` and include it in the `HybridRetriever`.

The existing RRF implementation can combine its ranked results with other retrieval methods.

### Adding Another Finalizer

Implement `Finalizer` and register it in the finalizer pipeline.

This allows specialized post-retrieval behavior without changing the retrievers themselves.

### Changing Storage Technology

The storage interfaces allow the current local implementations to be replaced with other backends.

For example:

```text
LocalSqlStorage    -> PostgreSQL implementation
LocalFileStorage   -> S3 bucket
LocalVectorStorage -> Alternative vector database
```

The higher-level services can remain unchanged as long as the replacement implements the corresponding interface.
