// Hand-maintained mirrors of the rag-server API schemas. Server-side
// schema changes need a manual sync here.

export interface UserResponse {
    username: string;
}

export interface ChatInfo {
    chat_id: string;
    created_at: number;
    // Origin source ids ingested into this chat.
    sources: string[];
}

export interface ChatCreated {
    chat_id: string;
}

// History items are heterogeneous; each carries only the fields for its
// type (null fields are omitted on the wire).
export interface ChatMessageUser {
    type: "user";
    content: string;
}

export interface ChatMessageToolCall {
    type: "tool_call";
    name: string;
    arguments: Record<string, unknown>;
}

export interface ChatMessageToolResult {
    type: "tool_result";
    name: string;
    content: string;
}

export interface ChatMessageAnswer {
    type: "answer";
    answer: string;
    chunk_refs: ChunkRefs;
}

export type ChatMessage =
    | ChatMessageUser
    | ChatMessageToolCall
    | ChatMessageToolResult
    | ChatMessageAnswer;

// Answer citations, keyed by the `#[n]` marker. The server assigns the
// integer per answer and resolves it to the real chunk ids, so the map is
// authoritative: markers it drops were never surfaced to the model.
export type ChunkRefs = Record<
    string,
    { origin_source_id: string; chunk_id: string }
>;

export type IngestStatus = "queued" | "running" | "done" | "error";

export interface IngestJob {
    job_id: string;
    status: IngestStatus;
    file: string;
}

export interface IngestJobsResponse {
    chat_id: string;
    jobs: IngestJob[];
}

export interface IngestJobEvent {
    name: string;
    payload: Record<string, unknown>;
}

export interface IngestJobInfo {
    job_id: string;
    status: IngestStatus;
    file: string;
    created_at: number;
    events: IngestJobEvent[];
}

export interface ListIngestJobsResponse {
    chat_id: string;
    jobs: IngestJobInfo[];
}

export interface StoredFile {
    source_id: string;
    filename: string;
    content_type: string;
    chat_id: string | null;
}

export interface ListFilesResponse {
    chat_id: string;
    files: StoredFile[];
}

export interface FileMetadata {
    source_id: string;
    filename: string;
    content_type: string;
    is_origin: boolean;
    chat_id: string | null;
    // Relative to the API base.
    link: string;
}

export interface StoredChunk {
    chunk_id: string;
    source_id: string;
    parent_source_id: string | null;
    origin_source_id: string;
    plugin: string;
    text: string;
    metadata: Record<string, unknown>;
    chat_id: string | null;
}

export interface ListFileChunksResponse {
    chat_id: string;
    origin_source_id: string;
    chunks: StoredChunk[];
}

export interface DeleteFileResponse {
    chat_id: string;
    origin_source_id: string;
    deleted_files: number;
}

// Terminal answer frame.
export interface AnswerStreamPayload {
    chat_id: string;
    query: string;
    answer: string;
    chunk_refs: ChunkRefs;
}

export interface AnswerDeltaPayload {
    content: string;
}

export interface ToolCallPayload {
    name: string;
    arguments: Record<string, unknown>;
}

export interface ToolResultPayload {
    name: string;
    content: string;
}

export interface StopResponse {
    chat_id: string;
    stopped: boolean;
}

// Ingest stream frames.
export interface IngestChatPayload {
    chat_id: string;
}

export interface IngestFilePayload {
    file: string;
}

export interface IngestStagePayload {
    file: string;
    stage: string;
    chunk_count?: number;
}

export interface IngestPluginStatePayload {
    file: string;
    plugin: string | null;
    state: string;
}

export interface IngestFileDonePayload {
    file: string;
    chunk_count: number;
}

export interface IngestDonePayload {
    file: string;
    total_chunks: number;
}

export interface IngestErrorPayload {
    file?: string | null;
    error: string;
}
