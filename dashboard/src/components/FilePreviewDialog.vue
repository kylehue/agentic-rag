<script setup lang="ts">
import { Download, LoaderCircle } from "@lucide/vue";
import { computed, onBeforeUnmount, ref, watch } from "vue";
import type { FileMetadata } from "@/api/types";
import { api } from "@/api/client";
import { useAppStore } from "@/store/app";
import { FILE_KIND_LABELS, byteSize, fileKind } from "@/lib/format";
import ChatFileIcon from "./ChatFileIcon.vue";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "./ui/dialog";

const store = useAppStore();

const MAX_TEXT_CHARS = 200_000;
const TEXT_KINDS = new Set(["text", "spreadsheet"]);

const state = ref<"loading" | "ready" | "error">("loading");
const error = ref("");
const metadata = ref<FileMetadata | null>(null);
const blobSize = ref(0);
const objectUrl = ref<string | null>(null);
const textContent = ref("");
const textTruncated = ref(false);

const kind = computed(() =>
    metadata.value
        ? fileKind(metadata.value.content_type, metadata.value.filename)
        : "other",
);
const label = computed(() => FILE_KIND_LABELS[kind.value] ?? "File");

function releaseUrl(): void {
    if (objectUrl.value) {
        URL.revokeObjectURL(objectUrl.value);
        objectUrl.value = null;
    }
}

// The bytes load once per open; switching the target reloads them.
async function load(sourceId: string): Promise<void> {
    releaseUrl();
    state.value = "loading";
    error.value = "";
    metadata.value = null;
    blobSize.value = 0;
    textContent.value = "";
    textTruncated.value = false;
    try {
        const [fileMetadata, blob] = await Promise.all([
            api.getFile(sourceId),
            api.fileBlob(sourceId),
        ]);
        if (store.filePreview?.sourceId !== sourceId) return;
        metadata.value = fileMetadata;
        blobSize.value = blob.size;
        objectUrl.value = URL.createObjectURL(blob);
        if (TEXT_KINDS.has(kind.value)) {
            const full = await blob.text();
            textTruncated.value = full.length > MAX_TEXT_CHARS;
            textContent.value = full.slice(0, MAX_TEXT_CHARS);
        }
        state.value = "ready";
    } catch (e) {
        if (store.filePreview?.sourceId !== sourceId) return;
        state.value = "error";
        error.value = e instanceof Error ? e.message : "Preview failed";
    }
}

watch(
    () => store.filePreview?.sourceId,
    (sourceId, previous) => {
        if (sourceId && sourceId !== previous) void load(sourceId);
        if (!sourceId) releaseUrl();
    },
    { immediate: true },
);

onBeforeUnmount(releaseUrl);
</script>

<template>
    <Dialog
        v-if="store.filePreview"
        :open="true"
        @update:open="store.filePreview = null"
    >
        <DialogContent class="max-w-2xl" data-testid="preview-dialog">
            <DialogHeader>
                <div class="flex items-center gap-2.5">
                    <ChatFileIcon
                        v-if="metadata"
                        :filename="metadata.filename"
                        :content-type="metadata.content_type"
                        class="h-5 w-5"
                    />
                    <DialogTitle
                        class="min-w-0 flex-1 truncate"
                        data-testid="preview-filename"
                    >
                        {{ metadata?.filename ?? "…" }}
                    </DialogTitle>
                    <span
                        v-if="metadata"
                        class="shrink-0 text-xs text-muted-foreground"
                        data-testid="preview-kind"
                    >
                        {{ label }} · {{ byteSize(blobSize) }}
                    </span>
                </div>
            </DialogHeader>

            <div class="min-h-40">
                <div
                    v-if="state === 'loading'"
                    class="flex justify-center p-10"
                >
                    <LoaderCircle
                        class="spin-icon h-6 w-6 text-muted-foreground"
                    />
                </div>
                <div
                    v-else-if="state === 'error'"
                    class="rounded-md border border-destructive/40 bg-destructive/10 p-4 text-center text-sm text-destructive"
                    data-testid="preview-error"
                >
                    {{ error }}
                </div>
                <template v-else-if="objectUrl">
                    <img
                        v-if="kind === 'image'"
                        :src="objectUrl"
                        :alt="metadata?.filename ?? 'image'"
                        class="mx-auto max-h-[60vh] rounded-md"
                        data-testid="preview-image"
                    />
                    <iframe
                        v-else-if="kind === 'pdf'"
                        :src="objectUrl"
                        title="PDF preview"
                        class="h-[60vh] w-full rounded-md border"
                        data-testid="preview-pdf"
                    />
                    <pre
                        v-else-if="TEXT_KINDS.has(kind)"
                        class="scrollbar-thin max-h-[60vh] overflow-auto whitespace-pre-wrap rounded-md border bg-muted/40 p-3 text-xs leading-relaxed"
                        data-testid="preview-text"
                    >
            {{ textContent }}{{ textTruncated ? "…" : "" }}</pre>
                    <p
                        v-else
                        class="rounded-md border p-10 text-center text-sm text-muted-foreground"
                        data-testid="preview-none"
                    >
                        No inline preview available for this file type.
                    </p>
                </template>
            </div>

            <div v-if="state === 'ready' && objectUrl" class="flex justify-end">
                <a
                    :href="objectUrl"
                    :download="metadata?.filename"
                    class="inline-flex items-center gap-1.5 text-xs text-primary underline underline-offset-2"
                    data-testid="preview-download"
                >
                    <Download class="h-3.5 w-3.5" />
                    Download
                </a>
            </div>
        </DialogContent>
    </Dialog>
</template>
