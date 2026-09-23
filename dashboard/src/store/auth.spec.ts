import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { ApiError, api } from "@/api/client";
import { useAppStore } from "@/store/app";

// The factory body runs lazily, so the helper is imported inside it to
// avoid the temporal dead zone of the hoisted vi.mock call.
vi.mock("@/api/client", async () => {
    const { mockClientFactory } = await import("../tests/mock-client");
    return mockClientFactory();
});

describe("auth store", () => {
    let store: ReturnType<typeof useAppStore>;

    beforeEach(() => {
        window.localStorage.clear();
        setActivePinia(createPinia());
        store = useAppStore();
    });

    it("lands on the auth screen when the session is missing", async () => {
        vi.mocked(api.me).mockRejectedValue(
            new ApiError(401, "Missing or invalid session."),
        );
        await store.init();
        expect(store.authStatus).toBe("guest");
    });

    it("restores the session and chat list on load", async () => {
        vi.mocked(api.me).mockResolvedValue({ username: "u" });
        vi.mocked(api.listChats).mockResolvedValue([
            { chat_id: "c1", created_at: 1000, sources: [] },
        ]);
        await store.init();
        expect(store.authStatus).toBe("authed");
        expect(store.username).toBe("u");
        expect(store.chats.map((c) => c.chat_id)).toEqual(["c1"]);
    });

    it("adopts the persisted chat when it still exists", async () => {
        window.localStorage.setItem("rag:active-chat", "c1");
        vi.mocked(api.me).mockResolvedValue({ username: "u" });
        vi.mocked(api.listChats).mockResolvedValue([
            { chat_id: "c1", created_at: 1000, sources: [] },
            { chat_id: "c2", created_at: 2000, sources: [] },
        ]);
        vi.mocked(api.chatMessages).mockResolvedValue([]);
        vi.mocked(api.listFiles).mockResolvedValue({
            chat_id: "c1",
            files: [],
        });
        vi.mocked(api.ingestJobs).mockResolvedValue({
            chat_id: "c1",
            jobs: [],
        });
        await store.init();
        expect(store.activeChatId).toBe("c1");
        expect(vi.mocked(api.chatMessages)).toHaveBeenCalledWith("c1");
    });

    it("signs in with valid credentials", async () => {
        vi.mocked(api.login).mockResolvedValue({ username: "u" });
        vi.mocked(api.listChats).mockResolvedValue([]);
        const detail = await store.login("u", "pass");
        expect(detail).toBeNull();
        expect(store.authStatus).toBe("authed");
        expect(store.username).toBe("u");
    });

    it("returns the error detail for a bad login instead of toasting", async () => {
        vi.mocked(api.login).mockRejectedValue(
            new ApiError(401, "Invalid username or password."),
        );
        const detail = await store.login("u", "nope");
        expect(detail).toBe("Invalid username or password.");
        expect(store.authStatus).toBe("loading");
    });

    it("registers and then logs in", async () => {
        vi.mocked(api.register).mockResolvedValue({ username: "newu" });
        vi.mocked(api.login).mockResolvedValue({ username: "newu" });
        vi.mocked(api.listChats).mockResolvedValue([]);
        const detail = await store.register("newu", "pass");
        expect(detail).toBeNull();
        expect(vi.mocked(api.register)).toHaveBeenCalledWith("newu", "pass");
        expect(vi.mocked(api.login)).toHaveBeenCalledWith("newu", "pass");
        expect(store.authStatus).toBe("authed");
    });

    it("signs out and clears all local state", async () => {
        vi.mocked(api.me).mockResolvedValue({ username: "u" });
        vi.mocked(api.listChats).mockResolvedValue([
            { chat_id: "c1", created_at: 1000, sources: [] },
        ]);
        await store.init();
        store.messages.push({ kind: "user", id: 1, content: "hi" });
        vi.mocked(api.logout).mockResolvedValue({ detail: "Logged out." });
        await store.logout();
        expect(store.authStatus).toBe("guest");
        expect(store.username).toBe("");
        expect(store.chats).toEqual([]);
        expect(store.messages).toEqual([]);
        expect(store.activeChatId).toBeNull();
    });
});
