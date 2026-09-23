import type {
    ChatCreated,
    ChatInfo,
    ChatMessage,
    DeleteFileResponse,
    FileMetadata,
    IngestJobsResponse,
    ListFileChunksResponse,
    ListFilesResponse,
    ListIngestJobsResponse,
    StopResponse,
    UserResponse,
} from "./types";
import { NETWORK_STATUS, SseDecoder, parseSseData, type SseFrame } from "./sse";

// Base URL: env override, defaulting to the local dev server. There is no
// dev proxy; every request is cross-origin and carries credentials.
export const API_BASE = (
    import.meta.env.VITE_RAG_BASE_URL ?? "http://localhost:8000"
).replace(/\/+$/, "");

// Sentinel status for network-level failures (distinct from HTTP errors).
export { NETWORK_STATUS };

export class ApiError extends Error {
    readonly status: number;

    constructor(status: number, detail: string) {
        super(detail);
        this.name = "ApiError";
        this.status = status;
    }
}

// Shared by the fetch client and the stream client so the two stay in sync.
export async function extractErrorDetail(response: Response): Promise<string> {
    try {
        const body: unknown = await response.clone().json();
        if (body && typeof body === "object" && "detail" in body) {
            const detail = (body as { detail: unknown }).detail;
            if (typeof detail === "string" && detail) return detail;
        }
    } catch {
        // non-JSON error body
    }
    return `Request failed (${response.status})`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
    let response: Response;
    try {
        response = await fetch(`${API_BASE}${path}`, {
            credentials: "include",
            ...init,
        });
    } catch {
        throw new ApiError(NETWORK_STATUS, "could not reach the server");
    }
    if (response.status === 204) return undefined as T;
    if (!response.ok) {
        throw new ApiError(response.status, await extractErrorDetail(response));
    }
    return (await response.json()) as T;
}

function jsonInit(method: string, body?: unknown): RequestInit {
    return {
        method,
        headers:
            body !== undefined
                ? { "Content-Type": "application/json" }
                : undefined,
        body: body !== undefined ? JSON.stringify(body) : undefined,
    };
}

function query(chatId: string | null | undefined): string {
    return chatId ? `?chat_id=${encodeURIComponent(chatId)}` : "";
}

export const api = {
    // --- auth ---
    register(username: string, password: string): Promise<UserResponse> {
        return request(
            "/auth/register",
            jsonInit("POST", { username, password }),
        );
    },
    login(username: string, password: string): Promise<UserResponse> {
        return request("/auth/login", jsonInit("POST", { username, password }));
    },
    logout(): Promise<{ detail: string }> {
        return request("/auth/logout", { method: "POST" });
    },
    me(): Promise<UserResponse> {
        return request("/auth/me");
    },

    // --- chats ---
    listChats(): Promise<ChatInfo[]> {
        return request("/chats");
    },
    createChat(): Promise<ChatCreated> {
        return request("/chats", { method: "POST" });
    },
    chatMessages(chatId: string): Promise<ChatMessage[]> {
        return request(`/chats/${encodeURIComponent(chatId)}/messages`);
    },
    deleteChat(chatId: string): Promise<void> {
        return request(`/chats/${encodeURIComponent(chatId)}`, {
            method: "DELETE",
        });
    },

    // --- ingest ---
    async uploadFiles(
        files: File[],
        chatId?: string | null,
    ): Promise<IngestJobsResponse> {
        const form = new FormData();
        // One part per file, named `files` (the server accepts repeated parts).
        for (const file of files) form.append("files", file);
        // Multipart must not set a content type: the browser owns the boundary.
        return request(`/rag/ingest${query(chatId)}`, {
            method: "POST",
            body: form,
        });
    },
    ingestJobs(chatId?: string | null): Promise<ListIngestJobsResponse> {
        return request(`/rag/ingest/jobs${query(chatId)}`);
    },
    ingestStream(
        jobId: string,
        signal?: AbortSignal,
    ): AsyncGenerator<SseFrame> {
        return streamSse(
            `/rag/ingest/stream?job_id=${encodeURIComponent(jobId)}`,
            {
                method: "GET",
                signal,
            },
        );
    },

    // --- files ---
    listFiles(chatId?: string | null): Promise<ListFilesResponse> {
        return request(`/rag/files${query(chatId)}`);
    },
    getFile(sourceId: string): Promise<FileMetadata> {
        return request(`/rag/files/${encodeURIComponent(sourceId)}`);
    },
    listFileChunks(
        originSourceId: string,
        chatId?: string | null,
    ): Promise<ListFileChunksResponse> {
        return request(
            `/rag/files/${encodeURIComponent(originSourceId)}/chunks${query(chatId)}`,
        );
    },
    deleteFile(
        originSourceId: string,
        chatId?: string | null,
    ): Promise<DeleteFileResponse> {
        return request(
            `/rag/files/${encodeURIComponent(originSourceId)}${query(chatId)}`,
            {
                method: "DELETE",
            },
        );
    },
    // The download link from metadata is relative to the API base and is
    // fetched with credentials.
    fileLink(metadata: FileMetadata): string {
        return `${API_BASE}${metadata.link}`;
    },
    async fileBlob(sourceId: string): Promise<Blob> {
        const metadata = await api.getFile(sourceId);
        let response: Response;
        try {
            response = await fetch(api.fileLink(metadata), {
                credentials: "include",
            });
        } catch {
            throw new ApiError(NETWORK_STATUS, "could not reach the server");
        }
        if (!response.ok) {
            throw new ApiError(
                response.status,
                await extractErrorDetail(response),
            );
        }
        return response.blob();
    },

    // --- answers ---
    answerStream(
        queryText: string,
        chatId: string | null | undefined,
        signal?: AbortSignal,
    ): AsyncGenerator<SseFrame> {
        return streamSse("/rag/answer/stream", {
            ...jsonInit("POST", { query: queryText, chat_id: chatId ?? null }),
            signal,
        });
    },
    stopAnswer(chatId: string): Promise<StopResponse> {
        return request("/rag/stop", jsonInit("POST", { chat_id: chatId }));
    },
};

async function* streamSse(
    path: string,
    init: RequestInit,
): AsyncGenerator<SseFrame> {
    const signal = init.signal;
    // An already-aborted request fails silently.
    if (signal?.aborted) return;
    let response: Response;
    try {
        response = await fetch(`${API_BASE}${path}`, {
            credentials: "include",
            ...init,
        });
    } catch (error) {
        if (isAbort(error) || signal?.aborted) return;
        throw new ApiError(NETWORK_STATUS, "could not reach the server");
    }
    if (signal?.aborted) return;
    if (!response.ok) {
        throw new ApiError(response.status, await extractErrorDetail(response));
    }
    if (!response.body) return;

    const bodyReader = response.body.getReader();
    const textDecoder = new TextDecoder();
    const sse = new SseDecoder();
    try {
        for (;;) {
            const { done, value } = await bodyReader.read();
            if (done) break;
            for (const frame of sse.push(
                textDecoder.decode(value, { stream: true }),
            )) {
                yield { event: frame.event, data: parseSseData(frame.data) };
            }
        }
        // Decode whatever remains in the buffer once the stream closes.
        for (const frame of sse.flush()) {
            yield { event: frame.event, data: parseSseData(frame.data) };
        }
    } catch (error) {
        if (isAbort(error) || signal?.aborted) return;
        throw new ApiError(NETWORK_STATUS, "could not reach the server");
    } finally {
        bodyReader.releaseLock();
    }
}

function isAbort(error: unknown): boolean {
    return (
        (error instanceof DOMException && error.name === "AbortError") ||
        (error instanceof Error && error.name === "AbortError")
    );
}
