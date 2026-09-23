import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { api } from "@/api/client";
import { useAppStore } from "@/store/app";
import { framesOf, type FakeFrame } from "../tests/fake-streams";

vi.mock("@/api/client", async () => {
    const { mockClientFactory } = await import("../tests/mock-client");
    return mockClientFactory();
});

function emptyFrames(): AsyncGenerator<FakeFrame> {
    return (async function* () {})();
}

describe("ingest", () => {
    let store: ReturnType<typeof useAppStore>;

    beforeEach(() => {
        window.localStorage.clear();
        setActivePinia(createPinia());
        store = useAppStore();
        store.activeChatId = "c1";
        vi.mocked(api.listChats).mockResolvedValue([
            { chat_id: "c1", created_at: 1000, sources: [] },
        ]);
        // Streams that end without a terminal frame reconcile against the jobs
        // snapshot; default it to empty so the reconcile is a no-op.
        vi.mocked(api.ingestJobs).mockResolvedValue({
            chat_id: "c1",
            jobs: [],
        });
    });

    it("registers a progress row per job and attaches each stream", async () => {
        vi.mocked(api.uploadFiles).mockResolvedValue({
            chat_id: "c1",
            jobs: [
                { job_id: "j1", status: "queued", file: "a.txt" },
                { job_id: "j2", status: "queued", file: "b.txt" },
            ],
        });
        vi.mocked(api.ingestStream).mockImplementation(() => emptyFrames());

        await store.upload([
            new File(["x"], "a.txt"),
            new File(["y"], "b.txt"),
        ]);

        expect(vi.mocked(api.uploadFiles)).toHaveBeenCalledWith(
            expect.any(Array),
            "c1",
        );
        expect(Object.keys(store.ingests).sort()).toEqual(["a.txt", "b.txt"]);
        expect(store.ingests["a.txt"]).toMatchObject({
            status: "queued",
            stage: "queued",
        });
        expect(vi.mocked(api.ingestStream)).toHaveBeenCalledWith(
            "j1",
            expect.any(AbortSignal),
        );
        expect(vi.mocked(api.ingestStream)).toHaveBeenCalledWith(
            "j2",
            expect.any(AbortSignal),
        );
    });

    it("tracks stages and promotes finished files into the durable list", async () => {
        vi.mocked(api.uploadFiles).mockResolvedValue({
            chat_id: "c1",
            jobs: [{ job_id: "j1", status: "queued", file: "a.txt" }],
        });
        vi.mocked(api.ingestStream).mockImplementation(() =>
            framesOf([
                ["chat", { chat_id: "c1" }],
                ["queued", { file: "a.txt" }],
                ["started", { file: "a.txt" }],
                ["stage", { file: "a.txt", stage: "processing" }],
                [
                    "plugin_state",
                    { file: "a.txt", plugin: "text", state: "partitioning" },
                ],
                [
                    "stage",
                    { file: "a.txt", stage: "embedding", chunk_count: 3 },
                ],
                ["file_done", { file: "a.txt", chunk_count: 3 }],
                ["done", { file: "a.txt", total_chunks: 3 }],
            ]),
        );
        vi.mocked(api.listFiles).mockResolvedValue({
            chat_id: "c1",
            files: [
                {
                    source_id: "s1",
                    filename: "a.txt",
                    content_type: "text/plain",
                    chat_id: "c1",
                },
            ],
        });

        await store.upload([new File(["x"], "a.txt")]);

        // The row disappears when the job settles; the file is durable by then.
        await vi.waitFor(() => expect(store.ingests["a.txt"]).toBeUndefined());
        expect(store.activeChatFiles.map((f) => f.filename)).toEqual(["a.txt"]);
        // No filename may appear in both lists at once.
        expect(store.ingests["a.txt"]).toBeUndefined();
        expect(store.activeChatFiles.some((f) => f.filename === "a.txt")).toBe(
            true,
        );
    });

    it("surfaces ingest errors on the row", async () => {
        vi.mocked(api.uploadFiles).mockResolvedValue({
            chat_id: "c1",
            jobs: [{ job_id: "j1", status: "queued", file: "a.txt" }],
        });
        vi.mocked(api.ingestStream).mockImplementation(() =>
            framesOf([
                ["chat", { chat_id: "c1" }],
                ["queued", { file: "a.txt" }],
                ["error", { file: "a.txt", error: "Invalid document." }],
            ]),
        );

        await store.upload([new File(["x"], "a.txt")]);

        await vi.waitFor(() =>
            expect(store.ingests["a.txt"]?.status).toBe("error"),
        );
        expect(store.ingests["a.txt"]?.error).toBe("Invalid document.");
    });

    it("adopts a chat minted by the upload", async () => {
        store.activeChatId = null;
        vi.mocked(api.uploadFiles).mockResolvedValue({
            chat_id: "newc",
            jobs: [{ job_id: "j1", status: "queued", file: "a.txt" }],
        });
        vi.mocked(api.ingestStream).mockImplementation(() => emptyFrames());
        vi.mocked(api.listFiles).mockResolvedValue({
            chat_id: "newc",
            files: [],
        });
        vi.mocked(api.listChats).mockResolvedValue([
            { chat_id: "newc", created_at: 2000, sources: [] },
        ]);

        await store.upload([new File(["x"], "a.txt")]);

        expect(store.activeChatId).toBe("newc");
    });

    it("does not create a stuck row for emitted sub-files", async () => {
        vi.mocked(api.uploadFiles).mockResolvedValue({
            chat_id: "c1",
            jobs: [{ job_id: "j1", status: "queued", file: "doc.pdf" }],
        });
        // The table plugin reports progress under the emitted sub-file name,
        // but only the job's file (doc.pdf) receives the terminal frame.
        vi.mocked(api.ingestStream).mockImplementation(() =>
            framesOf([
                ["chat", { chat_id: "c1" }],
                ["queued", { file: "doc.pdf" }],
                ["started", { file: "doc.pdf" }],
                ["stage", { file: "doc.pdf", stage: "processing" }],
                [
                    "plugin_state",
                    {
                        file: "doc_table_1.csv",
                        plugin: "table",
                        state: "analyzing_tables",
                    },
                ],
                [
                    "stage",
                    { file: "doc.pdf", stage: "embedding", chunk_count: 4 },
                ],
                ["file_done", { file: "doc.pdf", chunk_count: 4 }],
                ["done", { file: "doc.pdf", total_chunks: 4 }],
            ]),
        );
        vi.mocked(api.listFiles).mockResolvedValue({
            chat_id: "c1",
            files: [
                {
                    source_id: "s1",
                    filename: "doc.pdf",
                    content_type: "application/pdf",
                    chat_id: "c1",
                },
            ],
        });

        await store.upload([new File(["x"], "doc.pdf")]);

        await vi.waitFor(() =>
            expect(store.ingests["doc.pdf"]).toBeUndefined(),
        );
        // The sub-file never gets its own (permanently unsettled) row.
        expect(store.ingests["doc_table_1.csv"]).toBeUndefined();
        expect(Object.keys(store.ingests)).toEqual([]);
        expect(store.activeChatFiles.map((f) => f.filename)).toEqual([
            "doc.pdf",
        ]);
    });

    it("reconciles a dropped stream against the jobs snapshot", async () => {
        vi.mocked(api.uploadFiles).mockResolvedValue({
            chat_id: "c1",
            jobs: [{ job_id: "j1", status: "queued", file: "a.txt" }],
        });
        // The stream ends after a stage frame without the terminal frame.
        vi.mocked(api.ingestStream).mockImplementation(() =>
            framesOf([
                ["chat", { chat_id: "c1" }],
                ["queued", { file: "a.txt" }],
                [
                    "stage",
                    { file: "a.txt", stage: "embedding", chunk_count: 2 },
                ],
            ]),
        );
        vi.mocked(api.ingestJobs).mockResolvedValue({
            chat_id: "c1",
            jobs: [
                {
                    job_id: "j1",
                    status: "done",
                    file: "a.txt",
                    created_at: 1000,
                    events: [
                        { name: "chat", payload: { chat_id: "c1" } },
                        { name: "queued", payload: { file: "a.txt" } },
                        {
                            name: "stage",
                            payload: {
                                file: "a.txt",
                                stage: "embedding",
                                chunk_count: 2,
                            },
                        },
                        {
                            name: "done",
                            payload: { file: "a.txt", total_chunks: 2 },
                        },
                    ],
                },
            ],
        });
        vi.mocked(api.listFiles).mockResolvedValue({
            chat_id: "c1",
            files: [
                {
                    source_id: "s1",
                    filename: "a.txt",
                    content_type: "text/plain",
                    chat_id: "c1",
                },
            ],
        });

        await store.upload([new File(["x"], "a.txt")]);

        // The row settles without a page refresh.
        await vi.waitFor(() => expect(store.ingests["a.txt"]).toBeUndefined());
        expect(vi.mocked(api.ingestJobs)).toHaveBeenCalledWith("c1");
        expect(store.activeChatFiles.map((f) => f.filename)).toEqual(["a.txt"]);
    });

    it("shares one progress row per filename (known limitation)", async () => {
        vi.mocked(api.uploadFiles).mockResolvedValue({
            chat_id: "c1",
            jobs: [
                { job_id: "j1", status: "queued", file: "a.txt" },
                { job_id: "j2", status: "queued", file: "a.txt" },
            ],
        });
        vi.mocked(api.ingestStream).mockImplementation(() => emptyFrames());

        await store.upload([
            new File(["x"], "a.txt"),
            new File(["y"], "a.txt"),
        ]);

        expect(Object.keys(store.ingests)).toEqual(["a.txt"]);
    });
});

describe("ingest recovery", () => {
    let store: ReturnType<typeof useAppStore>;

    beforeEach(() => {
        window.localStorage.clear();
        setActivePinia(createPinia());
        store = useAppStore();
        store.activeChatId = "c1";
        vi.mocked(api.ingestStream).mockImplementation(() => emptyFrames());
        vi.mocked(api.listFiles).mockResolvedValue({
            chat_id: "c1",
            files: [
                {
                    source_id: "s2",
                    filename: "b.txt",
                    content_type: "text/plain",
                    chat_id: "c1",
                },
            ],
        });
    });

    it("replays the snapshot and re-attaches to running jobs", async () => {
        vi.mocked(api.ingestJobs).mockResolvedValue({
            chat_id: "c1",
            jobs: [
                {
                    job_id: "j1",
                    status: "running",
                    file: "a.txt",
                    created_at: 1000,
                    events: [
                        { name: "chat", payload: { chat_id: "c1" } },
                        { name: "queued", payload: { file: "a.txt" } },
                        { name: "started", payload: { file: "a.txt" } },
                        {
                            name: "stage",
                            payload: {
                                file: "a.txt",
                                stage: "embedding",
                                chunk_count: 2,
                            },
                        },
                    ],
                },
                {
                    job_id: "j2",
                    status: "done",
                    file: "b.txt",
                    created_at: 1000,
                    events: [
                        { name: "chat", payload: { chat_id: "c1" } },
                        {
                            name: "done",
                            payload: { file: "b.txt", total_chunks: 1 },
                        },
                    ],
                },
            ],
        });

        await store.recoverIngests();

        expect(store.ingests["a.txt"]).toMatchObject({
            status: "running",
            stage: "embedding",
            chunkCount: 2,
            progress: 85,
        });
        expect(store.ingests["b.txt"]).toBeUndefined();
        expect(vi.mocked(api.ingestStream)).toHaveBeenCalledWith(
            "j1",
            expect.any(AbortSignal),
        );
        expect(vi.mocked(api.ingestStream)).not.toHaveBeenCalledWith(
            "j2",
            expect.anything(),
        );
    });

    it("de-duplicates a job replayed by both snapshot and live stream", async () => {
        vi.mocked(api.ingestJobs).mockResolvedValue({
            chat_id: "c1",
            jobs: [
                {
                    job_id: "j1",
                    status: "running",
                    file: "a.txt",
                    created_at: 1000,
                    events: [
                        { name: "chat", payload: { chat_id: "c1" } },
                        { name: "queued", payload: { file: "a.txt" } },
                        {
                            name: "stage",
                            payload: { file: "a.txt", stage: "processing" },
                        },
                    ],
                },
            ],
        });
        // The live stream replays the same frames (late subscribers get the
        // buffered ones) plus newer progress.
        vi.mocked(api.ingestStream).mockImplementation(() =>
            framesOf([
                ["chat", { chat_id: "c1" }],
                ["queued", { file: "a.txt" }],
                ["stage", { file: "a.txt", stage: "processing" }],
                [
                    "stage",
                    { file: "a.txt", stage: "embedding", chunk_count: 5 },
                ],
            ]),
        );

        await store.recoverIngests();

        await vi.waitFor(() =>
            expect(store.ingests["a.txt"]?.stage).toBe("embedding"),
        );
        expect(store.ingests["a.txt"]?.chunkCount).toBe(5);
        // One row per filename, never a duplicate.
        expect(Object.keys(store.ingests)).toEqual(["a.txt"]);
    });
});
