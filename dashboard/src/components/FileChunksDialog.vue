<script setup lang="ts">
import { ChevronDown, LoaderCircle } from "@lucide/vue";
import { computed, ref } from "vue";
import type { StoredChunk } from "@/api/types";
import { useAppStore } from "@/store/app";
import { shortId } from "@/lib/format";
import Badge from "./ui/badge.vue";
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
} from "./ui/dialog";

const store = useAppStore();

const expanded = ref<Record<number, boolean>>({});

const entry = computed(() =>
    store.fileChunks
        ? store.chunkCache[store.fileChunks.originSourceId]
        : undefined,
);
const chunks = computed<StoredChunk[]>(() => entry.value?.chunks ?? []);
const loading = computed(() => entry.value?.loading === true);

function pageOf(chunk: StoredChunk): number | null {
    const value = chunk.metadata?.source_page_number;
    return typeof value === "number" ? value : null;
}
</script>

<template>
    <Dialog
        v-if="store.fileChunks"
        :open="true"
        @update:open="store.fileChunks = null"
    >
        <DialogContent class="max-w-2xl" data-testid="chunks-dialog">
            <DialogHeader>
                <DialogTitle class="truncate" data-testid="chunks-filename">
                    {{
                        store.fileChunks.filename ||
                        shortId(store.fileChunks.originSourceId)
                    }}
                </DialogTitle>
                <DialogDescription data-testid="chunks-count">
                    {{ chunks.length }} chunk{{
                        chunks.length === 1 ? "" : "s"
                    }}
                </DialogDescription>
            </DialogHeader>

            <div
                class="scrollbar-thin max-h-96 space-y-2 overflow-y-auto"
                data-testid="chunks-list"
            >
                <div v-if="loading" class="flex justify-center p-6">
                    <LoaderCircle
                        class="spin-icon h-5 w-5 text-muted-foreground"
                    />
                </div>
                <p
                    v-else-if="chunks.length === 0"
                    class="p-6 text-center text-sm text-muted-foreground"
                >
                    No chunks stored for this file.
                </p>
                <div
                    v-for="(chunk, index) in chunks"
                    v-else
                    :key="chunk.chunk_id"
                    class="rounded-md border"
                    data-testid="chunk-row"
                >
                    <button
                        type="button"
                        class="flex w-full items-center gap-2 p-2 text-left text-xs hover:bg-accent/40"
                        :aria-expanded="expanded[index] === true"
                        data-testid="chunk-preview"
                        @click="expanded[index] = !expanded[index]"
                    >
                        <span
                            class="w-5 shrink-0 text-right font-mono text-muted-foreground"
                        >
                            {{ index + 1 }}
                        </span>
                        <span class="min-w-0 flex-1 truncate">{{
                            chunk.text
                        }}</span>
                        <ChevronDown
                            class="h-3.5 w-3.5 shrink-0 text-muted-foreground transition-transform"
                            :class="{ 'rotate-180': expanded[index] }"
                        />
                    </button>
                    <div v-if="expanded[index]" class="space-y-2 border-t p-2">
                        <pre
                            class="scrollbar-thin max-h-64 overflow-auto whitespace-pre-wrap rounded bg-muted/40 p-2 text-xs leading-relaxed"
                            data-testid="chunk-full"
                        >
              {{ chunk.text }}</pre>
                        <div class="flex flex-wrap gap-1.5">
                            <Badge
                                variant="secondary"
                                data-testid="chunk-plugin"
                            >
                                {{ chunk.plugin }}
                            </Badge>
                            <Badge variant="secondary" data-testid="chunk-id">
                                {{ shortId(chunk.chunk_id) }}
                            </Badge>
                            <Badge
                                v-if="pageOf(chunk) !== null"
                                variant="secondary"
                                data-testid="chunk-page"
                            >
                                page {{ pageOf(chunk) }}
                            </Badge>
                        </div>
                    </div>
                </div>
            </div>
        </DialogContent>
    </Dialog>
</template>
