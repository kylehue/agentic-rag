import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { api } from "@/api/client";
import { useAppStore } from "@/store/app";
import SourcesPanel from "./SourcesPanel.vue";
import { mount } from "../tests/mount";

vi.mock("@/api/client", async () => {
    const { mockClientFactory } = await import("../tests/mock-client");
    return mockClientFactory();
});

describe("SourcesPanel", () => {
    let store: ReturnType<typeof useAppStore>;

    beforeEach(() => {
        window.localStorage.clear();
        setActivePinia(createPinia());
        store = useAppStore();
        store.authStatus = "authed";
        store.activeChatId = "c1";
    });

    it("shows the total count (in-flight plus durable)", () => {
        store.activeChatFiles = [
            {
                source_id: "s1",
                filename: "a.txt",
                content_type: "text/plain",
                chat_id: "c1",
            },
        ];
        store.ingests = {
            "b.txt": {
                filename: "b.txt",
                jobId: "j1",
                status: "running",
                stage: "embedding",
                pluginState: "text: partitioning",
                chunkCount: null,
                error: null,
                progress: 85,
            },
        };
        const wrapper = mount(SourcesPanel);
        expect(wrapper.find('[data-testid="sources-total"]').text()).toBe("2");
    });

    it("renders ingest rows with state, stage, and plugin detail", () => {
        store.ingests = {
            "b.txt": {
                filename: "b.txt",
                jobId: "j1",
                status: "running",
                stage: "embedding",
                pluginState: "text: partitioning",
                chunkCount: null,
                error: null,
                progress: 85,
            },
        };
        const wrapper = mount(SourcesPanel);
        const row = wrapper.find('[data-testid="ingest-row"]');
        expect(row.attributes("data-status")).toBe("running");
        expect(row.find('[data-testid="ingest-filename"]').text()).toBe(
            "b.txt",
        );
        expect(row.find('[data-testid="ingest-stage"]').text()).toBe(
            "embedding · text: partitioning",
        );
        expect(row.find('[data-testid="ingest-spinner"]').exists()).toBe(true);
    });

    it("renders failed rows with the error message", () => {
        store.ingests = {
            "bad.txt": {
                filename: "bad.txt",
                jobId: "j1",
                status: "error",
                stage: "failed",
                pluginState: null,
                chunkCount: null,
                error: "Invalid document.",
                progress: 100,
            },
        };
        const wrapper = mount(SourcesPanel);
        const row = wrapper.find('[data-testid="ingest-row"]');
        expect(row.attributes("data-status")).toBe("error");
        expect(row.find('[data-testid="ingest-cross"]').exists()).toBe(true);
        expect(row.find('[data-testid="ingest-error"]').text()).toBe(
            "Invalid document.",
        );
    });

    it("renders file rows with kind label and short id", () => {
        store.activeChatFiles = [
            {
                source_id: "abcdef12-3456-7890",
                filename: "report.pdf",
                content_type: "application/pdf",
                chat_id: "c1",
            },
        ];
        const wrapper = mount(SourcesPanel);
        const row = wrapper.find('[data-testid="file-row"]');
        expect(row.find('[data-testid="file-name"]').text()).toBe("report.pdf");
        expect(row.find('[data-testid="file-kind"]').text()).toBe(
            "PDF · abcdef12",
        );
    });

    it("opens the preview when a filename is clicked", async () => {
        store.activeChatFiles = [
            {
                source_id: "s1",
                filename: "a.txt",
                content_type: "text/plain",
                chat_id: "c1",
            },
        ];
        const wrapper = mount(SourcesPanel);
        await wrapper.find('[data-testid="file-name"]').trigger("click");
        expect(store.filePreview).toEqual({ sourceId: "s1" });
    });

    it("asks for confirmation before deleting a file", async () => {
        store.activeChatFiles = [
            {
                source_id: "s1",
                filename: "a.txt",
                content_type: "text/plain",
                chat_id: "c1",
            },
        ];
        vi.mocked(api.deleteFile).mockResolvedValue({
            chat_id: "c1",
            origin_source_id: "s1",
            deleted_files: 1,
        });
        const wrapper = mount(SourcesPanel);
        await wrapper.find('[data-testid="file-delete"]').trigger("click");
        expect(store.confirm).toMatchObject({
            title: "Delete file",
            destructive: true,
        });
        await store.executeConfirm();
        expect(store.confirm).toBeNull();
        expect(vi.mocked(api.deleteFile)).toHaveBeenCalledWith("s1", "c1");
        expect(store.activeChatFiles).toEqual([]);
    });

    it("shows the empty state", () => {
        const wrapper = mount(SourcesPanel);
        expect(wrapper.text()).toContain("No files yet.");
    });
});
