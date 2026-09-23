import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { api } from "@/api/client";
import { groupHistory, useAppStore } from "@/store/app";

vi.mock("@/api/client", async () => {
    const { mockClientFactory } = await import("../tests/mock-client");
    return mockClientFactory();
});

describe("groupHistory", () => {
    it("groups consecutive non-user items into one assistant turn", () => {
        const turns = groupHistory([
            { type: "user", content: "q" },
            { type: "tool_call", name: "search", arguments: { a: 1 } },
            { type: "tool_result", name: "search", content: "found" },
            { type: "answer", answer: "A", chunk_refs: {} },
        ]);
        expect(turns.map((t) => t.kind)).toEqual(["user", "assistant"]);
        const assistant = turns[1];
        if (assistant.kind === "assistant") {
            expect(assistant.toolSteps).toEqual([
                { name: "search", arguments: { a: 1 }, result: "found" },
            ]);
            expect(assistant.answer).toBe("A");
            expect(assistant.streaming).toBe(false);
        }
    });

    it("closes the group at each user item", () => {
        const turns = groupHistory([
            { type: "user", content: "q1" },
            { type: "answer", answer: "a1", chunk_refs: {} },
            { type: "user", content: "q2" },
            { type: "tool_call", name: "t", arguments: {} },
            { type: "answer", answer: "a2", chunk_refs: {} },
        ]);
        expect(turns.map((t) => t.kind)).toEqual([
            "user",
            "assistant",
            "user",
            "assistant",
        ]);
    });

    it("handles leading non-user items", () => {
        const turns = groupHistory([
            { type: "tool_call", name: "t", arguments: {} },
            { type: "answer", answer: "a", chunk_refs: {} },
        ]);
        expect(turns).toHaveLength(1);
        expect(turns[0].kind).toBe("assistant");
    });

    it("attaches citations from the answer item", () => {
        const turns = groupHistory([
            {
                type: "answer",
                answer: "see #[1]",
                chunk_refs: {
                    "#[1]": { origin_source_id: "o", chunk_id: "c" },
                },
            },
        ]);
        const assistant = turns[0];
        if (assistant.kind === "assistant") {
            expect(assistant.citations?.[0]).toMatchObject({
                number: 1,
                chunkId: "c",
                excerpt: null,
            });
        }
    });
});

describe("chat management", () => {
    let store: ReturnType<typeof useAppStore>;

    beforeEach(() => {
        window.localStorage.clear();
        setActivePinia(createPinia());
        store = useAppStore();
        store.chats = [
            { chat_id: "c1", created_at: 1000, sources: [] },
            { chat_id: "c2", created_at: 2000, sources: ["s1"] },
        ];
    });

    it("selecting a chat loads its history and files", async () => {
        vi.mocked(api.chatMessages).mockResolvedValue([
            { type: "user", content: "q" },
        ]);
        vi.mocked(api.listFiles).mockResolvedValue({
            chat_id: "c2",
            files: [
                {
                    source_id: "s1",
                    filename: "f.txt",
                    content_type: "text/plain",
                    chat_id: "c2",
                },
            ],
        });
        vi.mocked(api.ingestJobs).mockResolvedValue({
            chat_id: "c2",
            jobs: [],
        });

        await store.selectChat("c2");

        expect(store.activeChatId).toBe("c2");
        expect(store.messages[0]).toMatchObject({ kind: "user", content: "q" });
        expect(store.activeChatFiles).toHaveLength(1);
        expect(window.localStorage.getItem("rag:active-chat")).toBe("c2");
    });

    it("reconstructs citations when loading history", async () => {
        vi.mocked(api.chatMessages).mockResolvedValue([
            { type: "user", content: "q" },
            {
                type: "answer",
                answer: "fact #[1]",
                chunk_refs: {
                    "#[1]": { origin_source_id: "s1", chunk_id: "cx" },
                },
            },
        ]);
        vi.mocked(api.listFiles).mockResolvedValue({
            chat_id: "c2",
            files: [
                {
                    source_id: "s1",
                    filename: "f.txt",
                    content_type: "text/plain",
                    chat_id: "c2",
                },
            ],
        });
        vi.mocked(api.listFileChunks).mockResolvedValue({
            chat_id: "c2",
            origin_source_id: "s1",
            chunks: [
                {
                    chunk_id: "cx",
                    source_id: "s1",
                    parent_source_id: null,
                    origin_source_id: "s1",
                    plugin: "text",
                    text: "the cited fact",
                    metadata: { source_page_number: 4 },
                    chat_id: "c2",
                },
            ],
        });
        vi.mocked(api.ingestJobs).mockResolvedValue({
            chat_id: "c2",
            jobs: [],
        });

        await store.selectChat("c2");

        // Citation details resolve in the background after the context load.
        await vi.waitFor(() => {
            const assistant = store.messages[1];
            expect(
                assistant &&
                    assistant.kind === "assistant" &&
                    assistant.citations?.[0],
            ).toMatchObject({
                number: 1,
                excerpt: "the cited fact",
                page: 4,
                plugin: "text",
                filename: "f.txt",
            });
        });
    });

    it("deleting the active chat selects the first remaining one", async () => {
        vi.mocked(api.deleteChat).mockResolvedValue(undefined);
        vi.mocked(api.chatMessages).mockResolvedValue([]);
        vi.mocked(api.listFiles).mockResolvedValue({
            chat_id: "c1",
            files: [],
        });
        vi.mocked(api.ingestJobs).mockResolvedValue({
            chat_id: "c1",
            jobs: [],
        });
        store.activeChatId = "c2";

        await store.deleteChat("c2");

        expect(store.activeChatId).toBe("c1");
        expect(store.chats.map((c) => c.chat_id)).toEqual(["c1"]);
    });

    it("deleting the last chat resets to a fresh state", async () => {
        vi.mocked(api.deleteChat).mockResolvedValue(undefined);
        store.activeChatId = "c1";
        store.chats = [{ chat_id: "c1", created_at: 1000, sources: [] }];

        await store.deleteChat("c1");

        expect(store.activeChatId).toBeNull();
        expect(store.messages).toEqual([]);
        expect(store.activeChatFiles).toEqual([]);
        expect(window.localStorage.getItem("rag:active-chat")).toBeNull();
    });

    it("newChat clears the context without leaving a chat selected", async () => {
        store.activeChatId = "c2";
        store.messages = [{ kind: "user", id: 1, content: "x" }];
        store.activeChatFiles = [
            {
                source_id: "s1",
                filename: "f",
                content_type: "text/plain",
                chat_id: "c2",
            },
        ];

        store.newChat();

        expect(store.activeChatId).toBeNull();
        expect(store.messages).toEqual([]);
        expect(store.activeChatFiles).toEqual([]);
    });
});

describe("draft recovery", () => {
    let store: ReturnType<typeof useAppStore>;

    beforeEach(() => {
        window.localStorage.clear();
        setActivePinia(createPinia());
        store = useAppStore();
    });

    it("offers the draft when the run is unanswered (mid-tool history)", async () => {
        store.chats = [{ chat_id: "c1", created_at: 1000, sources: [] }];
        store.activeChatId = "c1";
        window.localStorage.setItem(
            "rag:draft",
            JSON.stringify({ question: "q", chatId: "c1" }),
        );
        // Checkpoints land after every node: a mid-run history can end in a
        // tool_result, not the user turn.
        vi.mocked(api.chatMessages).mockResolvedValue([
            { type: "user", content: "q" },
            { type: "tool_call", name: "search", arguments: {} },
            { type: "tool_result", name: "search", content: "results" },
        ]);
        vi.mocked(api.listFiles).mockResolvedValue({
            chat_id: "c1",
            files: [],
        });
        vi.mocked(api.ingestJobs).mockResolvedValue({
            chat_id: "c1",
            jobs: [],
        });

        await store.recoverDraft();

        expect(store.recoveredDraft).toBe("q");
    });

    it("offers the draft when the last stored item is the user turn", async () => {
        store.chats = [{ chat_id: "c1", created_at: 1000, sources: [] }];
        store.activeChatId = "c1";
        window.localStorage.setItem(
            "rag:draft",
            JSON.stringify({ question: "q", chatId: "c1" }),
        );
        vi.mocked(api.chatMessages).mockResolvedValue([
            { type: "user", content: "q" },
        ]);
        vi.mocked(api.listFiles).mockResolvedValue({
            chat_id: "c1",
            files: [],
        });
        vi.mocked(api.ingestJobs).mockResolvedValue({
            chat_id: "c1",
            jobs: [],
        });

        await store.recoverDraft();

        expect(store.recoveredDraft).toBe("q");
    });

    it("clears the draft when the answer already exists", async () => {
        store.chats = [{ chat_id: "c1", created_at: 1000, sources: [] }];
        store.activeChatId = "c1";
        window.localStorage.setItem(
            "rag:draft",
            JSON.stringify({ question: "q", chatId: "c1" }),
        );
        vi.mocked(api.chatMessages).mockResolvedValue([
            { type: "user", content: "q" },
            { type: "answer", answer: "a", chunk_refs: {} },
        ]);

        await store.recoverDraft();

        expect(store.recoveredDraft).toBeNull();
        expect(window.localStorage.getItem("rag:draft")).toBeNull();
    });

    it("recovers a draft recorded before the chat id existed", async () => {
        store.chats = [
            { chat_id: "old", created_at: 1000, sources: [] },
            { chat_id: "new", created_at: 2000, sources: [] },
        ];
        store.activeChatId = "old";
        window.localStorage.setItem(
            "rag:draft",
            JSON.stringify({ question: "q", chatId: null }),
        );
        vi.mocked(api.chatMessages).mockImplementation(
            async (chatId: string) => {
                if (chatId === "new") return [{ type: "user", content: "q" }];
                return [];
            },
        );
        vi.mocked(api.listFiles).mockResolvedValue({
            chat_id: "new",
            files: [],
        });
        vi.mocked(api.ingestJobs).mockResolvedValue({
            chat_id: "new",
            jobs: [],
        });

        await store.recoverDraft();

        expect(store.recoveredDraft).toBe("q");
        expect(store.activeChatId).toBe("new");
    });
});
