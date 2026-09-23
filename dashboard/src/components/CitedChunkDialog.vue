<script setup lang="ts">
import { LoaderCircle } from "@lucide/vue";
import { computed } from "vue";
import { useAppStore } from "@/store/app";
import { shortId } from "@/lib/format";
import Badge from "./ui/badge.vue";
import Button from "./ui/button.vue";
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
} from "./ui/dialog";

const store = useAppStore();

const chunk = computed(() => {
    const target = store.citedChunk;
    if (!target) return null;
    const entry = store.chunkCache[target.originSourceId];
    return entry?.chunks?.find((c) => c.chunk_id === target.chunkId) ?? null;
});

const loading = computed(() => {
    const target = store.citedChunk;
    if (!target) return false;
    return store.chunkCache[target.originSourceId]?.loading === true;
});

const filename = computed(() => {
    const target = store.citedChunk;
    if (!target) return null;
    return (
        store.activeChatFiles.find((f) => f.source_id === target.originSourceId)
            ?.filename ?? null
    );
});

const page = computed(() => {
    const value = chunk.value?.metadata?.source_page_number;
    return typeof value === "number" ? value : null;
});

function viewFullFile(): void {
    const target = store.citedChunk;
    store.citedChunk = null;
    if (target) store.openFilePreview(target.originSourceId);
}
</script>

<template>
    <Dialog
        v-if="store.citedChunk"
        :open="true"
        @update:open="store.citedChunk = null"
    >
        <DialogContent class="max-w-lg" data-testid="cited-chunk-dialog">
            <DialogHeader>
                <DialogTitle
                    class="truncate"
                    data-testid="cited-chunk-filename"
                >
                    {{ filename ?? shortId(store.citedChunk.originSourceId) }}
                </DialogTitle>
                <DialogDescription data-testid="cited-chunk-label">
                    cited chunk {{ shortId(store.citedChunk.chunkId) }}
                </DialogDescription>
            </DialogHeader>

            <div class="space-y-3">
                <div
                    v-if="loading"
                    class="flex justify-center p-6"
                    data-testid="cited-chunk-loading"
                >
                    <LoaderCircle
                        class="spin-icon h-5 w-5 text-muted-foreground"
                    />
                </div>
                <pre
                    v-else-if="chunk"
                    class="scrollbar-thin max-h-80 overflow-y-auto whitespace-pre-wrap rounded-md border bg-muted/40 p-3 text-sm leading-relaxed"
                    data-testid="cited-chunk-text"
                >
          {{ chunk.text }}</pre>
                <p
                    v-else
                    class="rounded-md border p-4 text-center text-sm text-muted-foreground"
                    data-testid="cited-chunk-unavailable"
                >
                    This chunk is unavailable (it may have been deleted).
                </p>

                <div class="flex flex-wrap gap-1.5 text-xs">
                    <Badge
                        v-if="page !== null"
                        variant="secondary"
                        data-testid="cited-chunk-page"
                    >
                        page {{ page }}
                    </Badge>
                    <Badge
                        v-if="chunk?.plugin"
                        variant="secondary"
                        data-testid="cited-chunk-plugin"
                    >
                        {{ chunk.plugin }}
                    </Badge>
                </div>
            </div>

            <DialogFooter>
                <Button
                    variant="outline"
                    data-testid="cited-chunk-view-file"
                    :disabled="loading"
                    @click="viewFullFile"
                >
                    View full file
                </Button>
            </DialogFooter>
        </DialogContent>
    </Dialog>
</template>
