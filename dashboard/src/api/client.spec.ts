import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "./client";
import { fakeSseResponse } from "../tests/fake-streams";

function jsonResponse(body: unknown, status = 200): Response {
    return new Response(JSON.stringify(body), {
        status,
        headers: { "Content-Type": "application/json" },
    });
}

describe("api client", () => {
    beforeEach(() => {
        vi.stubGlobal("fetch", vi.fn());
    });

    afterEach(() => {
        vi.unstubAllGlobals();
    });

    it("sends JSON requests with credentials and the content type", async () => {
        vi.stubGlobal(
            "fetch",
            vi.fn().mockResolvedValue(jsonResponse({ username: "u" })),
        );
        const result = await api.me();
        expect(result).toEqual({ username: "u" });
        const call = vi.mocked(fetch).mock.calls[0];
        expect(call[0]).toContain("/auth/me");
        expect(call[1]).toMatchObject({ credentials: "include" });
    });

    it("extracts the detail from JSON error bodies", async () => {
        vi.stubGlobal(
            "fetch",
            vi
                .fn()
                .mockResolvedValue(
                    jsonResponse(
                        { detail: "Missing or invalid session." },
                        401,
                    ),
                ),
        );
        const error = await api.me().catch((e) => e);
        expect(error).toBeInstanceOf(ApiError);
        expect((error as ApiError).status).toBe(401);
        expect((error as ApiError).message).toBe("Missing or invalid session.");
    });

    it("returns undefined for 204 responses", async () => {
        vi.stubGlobal(
            "fetch",
            vi.fn().mockResolvedValue(new Response(null, { status: 204 })),
        );
        const result = await api.deleteChat("abc");
        expect(result).toBeUndefined();
    });

    it("distinguishes network failures with the sentinel status", async () => {
        vi.stubGlobal(
            "fetch",
            vi.fn().mockRejectedValue(new TypeError("failed to fetch")),
        );
        const error = await api.me().catch((e) => e);
        expect(error).toBeInstanceOf(ApiError);
        expect((error as ApiError).status).toBe(0);
        expect((error as ApiError).message).toBe("could not reach the server");
    });

    it("uploads multipart files without a content type", async () => {
        vi.stubGlobal(
            "fetch",
            vi.fn().mockResolvedValue(
                jsonResponse({
                    chat_id: "c1",
                    jobs: [{ job_id: "j1", status: "queued", file: "a.txt" }],
                }),
            ),
        );
        const file = new File(["hello"], "a.txt", { type: "text/plain" });
        await api.uploadFiles([file], "c1");
        const call = vi.mocked(fetch).mock.calls[0];
        const init = call[1] as RequestInit;
        expect(init.body).toBeInstanceOf(FormData);
        expect(
            (init.body as FormData)
                .getAll("files")
                .map((f) => (f as File).name),
        ).toEqual(["a.txt"]);
        const headers = init.headers as Record<string, string> | undefined;
        expect(headers?.["Content-Type"]).toBeUndefined();
    });
});

describe("answer stream wrapper", () => {
    afterEach(() => {
        vi.unstubAllGlobals();
    });

    it("yields parsed frames on the happy path", async () => {
        vi.stubGlobal(
            "fetch",
            vi
                .fn()
                .mockResolvedValue(
                    fakeSseResponse([
                        'event: chat\ndata: {"chat_id": "c1"}\n\n',
                        'event: answer_delta\ndata: {"content": "hello "}\n\n',
                        'event: answer\ndata: {"query": "q", "answer": "hello", "chunk_refs": {}, "chat_id": "c1"}\n\n',
                    ]),
                ),
        );
        const frames = [];
        for await (const frame of api.answerStream("q", "c1"))
            frames.push(frame);
        expect(frames.map((f) => f.event)).toEqual([
            "chat",
            "answer_delta",
            "answer",
        ]);
        expect(frames[1].data).toEqual({ content: "hello " });
    });

    it("throws an ApiError with the detail for error statuses", async () => {
        vi.stubGlobal(
            "fetch",
            vi.fn().mockResolvedValue(
                fakeSseResponse([], {
                    status: 403,
                    body: JSON.stringify({ detail: "Not your ingest job." }),
                }),
            ),
        );
        const frames = [];
        let error: unknown;
        try {
            for await (const frame of api.ingestStream("j1"))
                frames.push(frame);
        } catch (e) {
            error = e;
        }
        expect(error).toBeInstanceOf(ApiError);
        expect((error as ApiError).status).toBe(403);
        expect((error as ApiError).message).toBe("Not your ingest job.");
    });

    it("fails silently when the request was already aborted", async () => {
        const controller = new AbortController();
        controller.abort();
        vi.stubGlobal(
            "fetch",
            vi
                .fn()
                .mockResolvedValue(fakeSseResponse(["event: x\ndata: {}\n\n"])),
        );
        const frames = [];
        for await (const frame of api.answerStream(
            "q",
            null,
            controller.signal,
        ))
            frames.push(frame);
        expect(frames).toEqual([]);
        expect(fetch).not.toHaveBeenCalled();
    });

    it("surfaces transport failures as the reach error", async () => {
        vi.stubGlobal(
            "fetch",
            vi.fn().mockRejectedValue(new TypeError("offline")),
        );
        let error: unknown;
        try {
            for await (const _frame of api.answerStream("q", null)) _frame;
        } catch (e) {
            error = e;
        }
        expect(error).toBeInstanceOf(ApiError);
        expect((error as ApiError).status).toBe(0);
    });

    it("aborts mid-stream without surfacing an error", async () => {
        const controller = new AbortController();
        const encoder = new TextEncoder();
        const abortError = () =>
            new DOMException("The operation was aborted.", "AbortError");
        // The first pull yields one frame; any pull after an abort rejects with
        // an AbortError, like a real fetch body does.
        const stream = new ReadableStream<Uint8Array>({
            pull(streamController) {
                return new Promise<void>((resolve, reject) => {
                    const onAbort = () => {
                        controller.signal.removeEventListener("abort", onAbort);
                        reject(abortError());
                    };
                    if (controller.signal.aborted) {
                        onAbort();
                        return;
                    }
                    controller.signal.addEventListener("abort", onAbort, {
                        once: true,
                    });
                    setTimeout(() => {
                        controller.signal.removeEventListener("abort", onAbort);
                        if (controller.signal.aborted) {
                            reject(abortError());
                            return;
                        }
                        streamController.enqueue(
                            encoder.encode(
                                'event: chat\ndata: {"chat_id": "c1"}\n\n',
                            ),
                        );
                        resolve();
                    }, 5);
                });
            },
        });
        vi.stubGlobal(
            "fetch",
            vi.fn().mockResolvedValue(
                new Response(stream, {
                    headers: { "Content-Type": "text/event-stream" },
                }),
            ),
        );
        const frames: string[] = [];
        let settled = false;
        let thrown: unknown = null;
        void (async () => {
            try {
                for await (const frame of api.answerStream(
                    "q",
                    "c1",
                    controller.signal,
                )) {
                    frames.push(frame.event);
                    if (frames.length === 1) controller.abort();
                }
                settled = true;
            } catch (error) {
                thrown = error;
            }
        })();
        await vi.waitFor(() => expect(settled).toBe(true));
        expect(frames).toEqual(["chat"]);
        expect(thrown).toBeNull();
    });
});
