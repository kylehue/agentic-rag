<script setup lang="ts">
import { useAppStore, type AssistantTurn } from "@/store/app";
import type { CitationEntry } from "@/lib/citations";
import { shortId } from "@/lib/format";

const props = defineProps<{
    turn: AssistantTurn;
}>();

const store = useAppStore();

function openChunk(entry: CitationEntry): void {
    void store.openCitedChunk(entry.originSourceId, entry.chunkId);
}

function viewFile(entry: CitationEntry): void {
    store.openFilePreview(entry.originSourceId);
}
</script>

<template>
    <div
        v-if="turn.citations && turn.citations.length > 0"
        class="mt-3 space-y-1.5"
        data-testid="sources-block"
    >
        <p class="text-xs font-medium text-muted-foreground">Sources</p>
        <div
            v-for="entry in turn.citations"
            :key="entry.refKey"
            class="flex cursor-pointer items-start gap-2 rounded-md border p-2 text-xs transition-colors hover:bg-accent/40"
            data-testid="source-item"
            @click="openChunk(entry)"
        >
            <span
                class="citation-badge pointer-events-none"
                data-testid="source-number"
            >
                {{ entry.number }}
            </span>
            <div class="min-w-0 flex-1">
                <p class="truncate font-medium" data-testid="source-filename">
                    {{ entry.filename ?? shortId(entry.originSourceId) }}
                </p>
                <p
                    v-if="entry.page !== null || entry.plugin !== null"
                    class="text-muted-foreground"
                >
                    <template v-if="entry.page !== null"
                        >page {{ entry.page }}</template
                    >
                    <template
                        v-if="entry.page !== null && entry.plugin !== null"
                    >
                        ·
                    </template>
                    <template v-if="entry.plugin !== null">{{
                        entry.plugin
                    }}</template>
                </p>
                <p
                    class="mt-0.5 line-clamp-2 whitespace-pre-wrap text-muted-foreground"
                    data-testid="source-excerpt"
                >
                    {{
                        entry.excerpt ?? `cited chunk ${shortId(entry.chunkId)}`
                    }}
                </p>
                <button
                    type="button"
                    class="mt-1 text-primary underline underline-offset-2"
                    data-testid="source-view-file"
                    @click.stop="viewFile(entry)"
                >
                    View full file
                </button>
            </div>
        </div>
    </div>
</template>
