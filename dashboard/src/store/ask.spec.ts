import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { ApiError, api } from "@/api/client";
import { useAppStore } from "@/store/app";
import {
    framesOf,
    framesUntilAbort,
    type FakeFrame,
} from "../tests/fake-streams";

vi.mock("@/api/client", async () => {
    const { mockClientFactory } = await import("../tests/mock-client");
    return mockClientFactory();
});

function mockAnswerStream(frames: Array<[string, unknown]>): void {
    vi.mocked(api.answerStream).mockImplementation(
        () => framesOf(frames) as unknown as AsyncGenerator<FakeFrame>,
    );
}

describe("ask flow", () => {
    let store: ReturnType<typeof useAppStore>;

    beforeEach(() => {
        window.localStorage.clear();
        setActivePinia(createPinia());
        store = useAppStore();
        store.activeChatId = "c1";
    });

    it("streams deltas, records tool steps, and finalizes from the terminal frame", async () => {
        mockAnswerStream([
            ["chat", { chat_id: "c1" }],
            [
                "tool_call",
                { name: "search_documents", arguments: { query: "x" } },
            ],
            ["tool_result", { name: "search_documents", content: "found it" }],
            ["answer_delta", { content: "Hello " }],
            ["answer_delta", { content: "world" }],
            [
                "answer",
                {
                    query: "q",
                    answer: "Hello world",
                    chunk_refs: {},
                    chat_id: "c1",
                },
            ],
        ]);
        vi.mocked(api.listChats).mockResolvedValue([
            { chat_id: "c1", created_at: 1000, sources: [] },
        ]);

        await store.ask("q");

        const turns = store.messages;
        expect(turns.map((t) => t.kind)).toEqual(["user", "assistant"]);
        const assistant = turns[1];
        if (assistant.kind === "assistant") {
            expect(assistant.toolSteps).toEqual([
                {
                    name: "search_documents",
                    arguments: { query: "x" },
                    result: "found it",
                },
            ]);
            // The terminal answer overrides the streamed partials.
            expect(assistant.answer).toBe("Hello world");
            expect(assistant.streaming).toBe(false);
            expect(assistant.failed).toBe(false);
        }
        expect(store.streaming).toBe(false);
        expect(window.localStorage.getItem("rag:draft")).toBeNull();
    });

    it("logs the raw conversation to the console, deltas skipped", async () => {
        const log = vi.spyOn(console, "log").mockImplementation(() => {});
        mockAnswerStream([
            ["chat", { chat_id: "c1" }],
            ["answer_delta", { content: "Hello " }],
            ["answer_delta", { content: "world" }],
            [
                "answer",
                {
                    query: "q",
                    answer: "Hello world",
                    chunk_refs: {
                        "#[1]": { origin_source_id: "s1", chunk_id: "cx" },
                    },
                    chat_id: "c1",
                },
            ],
        ]);
        vi.mocked(api.listChats).mockResolvedValue([
            { chat_id: "c1", created_at: 1000, sources: [] },
        ]);
        vi.mocked(api.listFileChunks).mockResolvedValue({
            chat_id: "c1",
            origin_source_id: "s1",
            chunks: [],
        });

        await store.ask("q");

        const labels = log.mock.calls.map((args) => args[0]);
        expect(labels).toEqual([
            "[conversation] user",
            "[conversation] chat",
            "[conversation] answer",
        ]);
        // The payloads are logged raw, untransformed.
        expect(log.mock.calls[0][1]).toBe("q");
        expect(log.mock.calls[2][1]).toMatchObject({
            answer: "Hello world",
            chunk_refs: { "#[1]": { origin_source_id: "s1", chunk_id: "cx" } },
        });
        log.mockRestore();
    });

    it("resolves citations from fetched chunks", async () => {
        mockAnswerStream([
            ["chat", { chat_id: "c1" }],
            [
                "answer",
                {
                    query: "q",
                    answer: "fact #[1]",
                    chunk_refs: {
                        "#[1]": { origin_source_id: "s1", chunk_id: "cx" },
                    },
                    chat_id: "c1",
                },
            ],
        ]);
        store.activeChatFiles = [
            {
                source_id: "s1",
                filename: "f.txt",
                content_type: "text/plain",
                chat_id: "c1",
            },
        ];
        vi.mocked(api.listFileChunks).mockResolvedValue({
            chat_id: "c1",
            origin_source_id: "s1",
            chunks: [
                {
                    chunk_id: "cx",
                    source_id: "s1",
                    parent_source_id: null,
                    origin_source_id: "s1",
                    plugin: "text",
                    text: "the cited fact",
                    metadata: { source_page_number: 7 },
                    chat_id: "c1",
                },
            ],
        });
        vi.mocked(api.listChats).mockResolvedValue([
            { chat_id: "c1", created_at: 1000, sources: ["s1"] },
        ]);

        await store.ask("q");

        const assistant = store.messages[1];
        if (assistant.kind === "assistant") {
            expect(assistant.citations?.[0]).toMatchObject({
                number: 1,
                chunkId: "cx",
                filename: "f.txt",
                page: 7,
                plugin: "text",
                excerpt: "the cited fact",
            });
        }
    });

    it("adopts a chat id minted during the stream", async () => {
        store.activeChatId = null;
        mockAnswerStream([
            ["chat", { chat_id: "newc" }],
            [
                "answer",
                {
                    query: "q",
                    answer: "a",
                    chunk_refs: {},
                    chat_id: "newc",
                },
            ],
        ]);
        vi.mocked(api.listChats).mockResolvedValue([
            { chat_id: "newc", created_at: 2000, sources: [] },
        ]);

        await store.ask("q");

        expect(store.activeChatId).toBe("newc");
        expect(window.localStorage.getItem("rag:active-chat")).toBe("newc");
    });

    it("marks the run failed on real errors and keeps the draft", async () => {
        vi.mocked(api.answerStream).mockImplementation(() =>
            (async function* () {
                yield { event: "chat", data: { chat_id: "c1" } };
                throw new ApiError(500, "The agent produced no final answer.");
            })(),
        );

        await store.ask("q");

        const assistant = store.messages[1];
        if (assistant.kind === "assistant") {
            expect(assistant.failed).toBe(true);
            expect(assistant.error).toBe("The agent produced no final answer.");
            expect(assistant.streaming).toBe(false);
        }
        const draft = JSON.parse(
            window.localStorage.getItem("rag:draft") ?? "{}",
        ) as { question: string; chatId: string | null };
        expect(draft).toEqual({ question: "q", chatId: "c1" });
    });

    it("does not mark the run failed when the user stops it", async () => {
        vi.mocked(api.stopAnswer).mockResolvedValue({
            chat_id: "c1",
            stopped: true,
        });
        vi.mocked(api.answerStream).mockImplementation(
            (
                _query: string,
                _chatId: string | null | undefined,
                signal?: AbortSignal,
            ) =>
                framesUntilAbort(signal ?? new AbortController().signal, [
                    ["chat", { chat_id: "c1" }],
                    ["answer_delta", { content: "partial" }],
                ]),
        );

        const pending = store.ask("q");
        await vi.waitFor(() => expect(store.streaming).toBe(true));
        await store.stop();
        await pending;

        const assistant = store.messages[1];
        if (assistant.kind === "assistant") {
            expect(assistant.failed).toBe(false);
            expect(assistant.streaming).toBe(false);
            expect(assistant.answer).toBe("partial");
        }
        expect(vi.mocked(api.stopAnswer)).toHaveBeenCalledWith("c1");
        expect(window.localStorage.getItem("rag:draft")).toBeNull();
    });

    it("ignores frames after the chat switches mid-stream", async () => {
        // The stream pauses on a gate after the first delta so the test can
        // switch chats before the next frame arrives.
        let release: () => void = () => undefined;
        const gate = new Promise<void>((resolve) => (release = resolve));
        vi.mocked(api.answerStream).mockImplementation(() =>
            (async function* () {
                yield { event: "chat", data: { chat_id: "c1" } };
                yield { event: "answer_delta", data: { content: "A" } };
                await gate;
                yield { event: "answer_delta", data: { content: "B" } };
                yield {
                    event: "answer",
                    data: {
                        query: "q",
                        answer: "AB",
                        chunk_refs: {},
                        chat_id: "c1",
                    },
                };
            })(),
        );
        vi.mocked(api.listChats).mockResolvedValue([
            { chat_id: "c1", created_at: 1000, sources: [] },
        ]);

        const pending = store.ask("q");
        await vi.waitFor(() => {
            const last = store.messages.at(-1);
            expect(last && last.kind === "assistant" && last.answer).toBe("A");
        });
        // Simulate a chat switch: the messages array is replaced.
        store.messages = [{ kind: "user", id: 99, content: "other" }];
        release();
        await pending;
        // The switched-away partial message is dropped; 'B' never lands.
        expect(store.messages).toEqual([
            { kind: "user", id: 99, content: "other" },
        ]);
    });
});
