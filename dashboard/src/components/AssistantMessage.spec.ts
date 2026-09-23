import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { api } from "@/api/client";
import { useAppStore, type AssistantTurn } from "@/store/app";
import AssistantMessage from "./AssistantMessage.vue";
import { mount } from "../tests/mount";

vi.mock("@/api/client", async () => {
    const { mockClientFactory } = await import("../tests/mock-client");
    return mockClientFactory();
});

function makeTurn(patch: Partial<AssistantTurn> = {}): AssistantTurn {
    return {
        kind: "assistant",
        id: 1,
        toolSteps: [],
        answer: null,
        chunkRefs: null,
        citations: null,
        streaming: false,
        failed: false,
        error: null,
        ...patch,
    };
}

describe("AssistantMessage", () => {
    let store: ReturnType<typeof useAppStore>;

    beforeEach(() => {
        window.localStorage.clear();
        setActivePinia(createPinia());
        store = useAppStore();
        store.authStatus = "authed";
        store.activeChatId = "c1";
    });

    it("renders markdown content", () => {
        const wrapper = mount(AssistantMessage, {
            props: {
                turn: makeTurn({ answer: "# Title\n\n**bold** and `code`" }),
            },
        });
        const html = wrapper.find('[data-testid="markdown-body"]').html();
        expect(html).toContain("<h1>Title</h1>");
        expect(html).toContain("<strong>bold</strong>");
        expect(html).toContain("<code>code</code>");
    });

    it("renders citation badges numbered by their markers", () => {
        const answer = "A #[1] and B #[2]";
        const turn = makeTurn({
            answer,
            chunkRefs: {
                "#[1]": { origin_source_id: "o1", chunk_id: "c1" },
                "#[2]": { origin_source_id: "o2", chunk_id: "c2" },
            },
            citations: [
                {
                    refKey: "#[1]",
                    number: 1,
                    originSourceId: "o1",
                    chunkId: "c1",
                    filename: "a.txt",
                    page: 2,
                    plugin: "text",
                    excerpt: "known excerpt",
                },
                {
                    refKey: "#[2]",
                    number: 2,
                    originSourceId: "o2",
                    chunkId: "c2",
                    filename: null,
                    page: null,
                    plugin: null,
                    excerpt: null,
                },
            ],
        });
        const wrapper = mount(AssistantMessage, { props: { turn } });
        const badges = wrapper
            .find('[data-testid="markdown-body"]')
            .findAll(".citation-badge");
        expect(badges.map((b) => b.text())).toEqual(["1", "2"]);

        const sources = wrapper.findAll('[data-testid="source-item"]');
        expect(sources).toHaveLength(2);
        expect(wrapper.find('[data-testid="source-filename"]').text()).toBe(
            "a.txt",
        );
        expect(wrapper.find('[data-testid="source-number"]').text()).toBe("1");
        // Unknown chunks fall back to a "cited chunk" label with a short id.
        expect(
            sources[1].find('[data-testid="source-excerpt"]').text(),
        ).toContain("cited chunk c2");
    });

    it("hides inline badges for a single-citation answer but keeps the sources block", () => {
        const turn = makeTurn({
            answer: "the answer is 42 #[1]",
            chunkRefs: {
                "#[1]": { origin_source_id: "o1", chunk_id: "c1" },
            },
            citations: [
                {
                    refKey: "#[1]",
                    number: 1,
                    originSourceId: "o1",
                    chunkId: "c1",
                    filename: "a.txt",
                    page: null,
                    plugin: null,
                    excerpt: null,
                },
            ],
        });
        const wrapper = mount(AssistantMessage, { props: { turn } });
        expect(
            wrapper
                .find('[data-testid="markdown-body"]')
                .find(".citation-badge")
                .exists(),
        ).toBe(false);
        expect(wrapper.find('[data-testid="markdown-body"]').text()).toContain(
            "the answer is 42",
        );
        expect(wrapper.find('[data-testid="sources-block"]').exists()).toBe(
            true,
        );
        expect(wrapper.find('[data-testid="source-filename"]').text()).toBe(
            "a.txt",
        );
    });

    it("hides the badge for a lone streaming marker, and shows both when a second arrives", async () => {
        // One marker so far: the badge is hidden already.
        const wrapper = mount(AssistantMessage, {
            props: {
                turn: makeTurn({
                    answer: "first #[1]",
                    chunkRefs: null,
                    streaming: true,
                }),
            },
        });
        expect(
            wrapper
                .find('[data-testid="markdown-body"]')
                .find(".citation-badge")
                .exists(),
        ).toBe(false);
        // The second marker arrives: both badges show again.
        await wrapper.setProps({
            turn: makeTurn({
                answer: "first #[1] then #[2]",
                chunkRefs: null,
                streaming: true,
            }),
        });
        expect(
            wrapper
                .find('[data-testid="markdown-body"]')
                .findAll(".citation-badge")
                .map((b) => b.text()),
        ).toEqual(["1", "2"]);
    });

    it("numbers badges from the markers while streaming without refs", () => {
        const wrapper = mount(AssistantMessage, {
            props: {
                turn: makeTurn({
                    answer: "first #[1] then #[2]",
                    chunkRefs: null,
                    streaming: true,
                }),
            },
        });
        expect(
            wrapper
                .find('[data-testid="markdown-body"]')
                .findAll(".citation-badge")
                .map((b) => b.text()),
        ).toEqual(["1", "2"]);
    });

    it("opens the cited chunk when a badge is clicked", async () => {
        // Two citations: a single-citation answer renders no badge at all.
        const turn = makeTurn({
            answer: "see #[1] and #[2]",
            chunkRefs: {
                "#[1]": { origin_source_id: "o1", chunk_id: "c1" },
                "#[2]": { origin_source_id: "o2", chunk_id: "c2" },
            },
            citations: [
                {
                    refKey: "#[1]",
                    number: 1,
                    originSourceId: "o1",
                    chunkId: "c1",
                    filename: "a.txt",
                    page: null,
                    plugin: null,
                    excerpt: null,
                },
                {
                    refKey: "#[2]",
                    number: 2,
                    originSourceId: "o2",
                    chunkId: "c2",
                    filename: "b.txt",
                    page: null,
                    plugin: null,
                    excerpt: null,
                },
            ],
        });
        vi.mocked(api.listFileChunks).mockResolvedValue({
            chat_id: "c1",
            origin_source_id: "o1",
            chunks: [],
        });
        const wrapper = mount(AssistantMessage, { props: { turn } });
        await wrapper.find(".citation-badge").trigger("click");
        expect(store.citedChunk).toEqual({
            originSourceId: "o1",
            chunkId: "c1",
        });
    });

    it("shows the streaming placeholder before any content arrives", () => {
        const wrapper = mount(AssistantMessage, {
            props: { turn: makeTurn({ streaming: true, answer: null }) },
        });
        expect(wrapper.find('[data-testid="thinking"]').text()).toBe(
            "Thinking",
        );
        expect(wrapper.find(".caret").exists()).toBe(true);
    });

    it("shows an error banner when the run failed", () => {
        const wrapper = mount(AssistantMessage, {
            props: {
                turn: makeTurn({ failed: true, error: "boom", answer: null }),
            },
        });
        expect(
            wrapper.find('[data-testid="assistant-error"]').text(),
        ).toContain("boom");
    });
});
