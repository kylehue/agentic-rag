import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { api } from "@/api/client";
import { useAppStore } from "@/store/app";
import { mount } from "../tests/mount";
import {
    framesOf,
    framesUntilAbort,
    type FakeFrame,
} from "../tests/fake-streams";
import Composer from "./Composer.vue";

vi.mock("@/api/client", async () => {
    const { mockClientFactory } = await import("../tests/mock-client");
    return mockClientFactory();
});

describe("Composer", () => {
    let store: ReturnType<typeof useAppStore>;

    beforeEach(() => {
        window.localStorage.clear();
        setActivePinia(createPinia());
        store = useAppStore();
        store.authStatus = "authed";
        vi.mocked(api.listChats).mockResolvedValue([]);
    });

    it("sends on Enter", async () => {
        vi.mocked(api.answerStream).mockImplementation(
            () => framesOf([]) as unknown as AsyncGenerator<FakeFrame>,
        );
        const wrapper = mount(Composer);
        const input = wrapper.find('[data-testid="composer-input"]');
        await input.setValue("hello");
        await input.trigger("keydown", { key: "Enter" });
        expect(vi.mocked(api.answerStream)).toHaveBeenCalledWith(
            "hello",
            null,
            expect.any(AbortSignal),
        );
        await vi.waitFor(() =>
            expect(
                (input.element as unknown as HTMLTextAreaElement).value,
            ).toBe(""),
        );
    });

    it("does not send on Shift+Enter", async () => {
        vi.mocked(api.answerStream).mockImplementation(
            () => framesOf([]) as unknown as AsyncGenerator<FakeFrame>,
        );
        const wrapper = mount(Composer);
        const input = wrapper.find('[data-testid="composer-input"]');
        await input.setValue("hello");
        await input.trigger("keydown", { key: "Enter", shiftKey: true });
        expect(vi.mocked(api.answerStream)).not.toHaveBeenCalled();
    });

    it("disables send while the text is empty", async () => {
        const wrapper = mount(Composer);
        const button = wrapper.find('[data-testid="composer-send"]');
        expect(button.attributes("disabled")).toBeDefined();
        await wrapper.find('[data-testid="composer-input"]').setValue("x");
        expect(button.attributes("disabled")).toBeUndefined();
    });

    it("uses different placeholders with and without an active chat", async () => {
        const wrapper = mount(Composer);
        expect(
            wrapper
                .find('[data-testid="composer-input"]')
                .attributes("placeholder"),
        ).toBe("Start a new conversation");
        store.activeChatId = "c1";
        await wrapper.vm.$nextTick();
        expect(
            wrapper
                .find('[data-testid="composer-input"]')
                .attributes("placeholder"),
        ).toBe("Ask a question about your documents");
    });

    it("pre-fills the recovered draft and clears it", async () => {
        store.recoveredDraft = "my recovered question";
        const wrapper = mount(Composer);
        await wrapper.vm.$nextTick();
        const input = wrapper.find('[data-testid="composer-input"]');
        expect((input.element as unknown as HTMLTextAreaElement).value).toBe(
            "my recovered question",
        );
        expect(store.recoveredDraft).toBeNull();
    });

    it("shows the stop button while streaming and stops the run", async () => {
        store.activeChatId = "c1";
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
                    ["answer_delta", { content: "x" }],
                ]),
        );
        const wrapper = mount(Composer);
        const input = wrapper.find('[data-testid="composer-input"]');
        await input.setValue("go");
        await input.trigger("keydown", { key: "Enter" });
        await vi.waitFor(() => expect(store.streaming).toBe(true));
        expect(wrapper.find('[data-testid="composer-stop"]').exists()).toBe(
            true,
        );
        expect(wrapper.find('[data-testid="composer-send"]').exists()).toBe(
            false,
        );
        await wrapper.find('[data-testid="composer-stop"]').trigger("click");
        await vi.waitFor(() => expect(store.streaming).toBe(false));
        expect(vi.mocked(api.stopAnswer)).toHaveBeenCalledWith("c1");
    });
});
