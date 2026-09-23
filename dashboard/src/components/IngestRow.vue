<script setup lang="ts">
import { Check, LoaderCircle, X } from "@lucide/vue";
import { computed } from "vue";
import type { IngestRow as IngestRowState } from "@/store/app";
import ChatFileIcon from "./ChatFileIcon.vue";
import { Progress, ProgressIndicator } from "./ui/progress";

const props = defineProps<{
    row: IngestRowState;
}>();

const running = computed(
    () => props.row.status === "queued" || props.row.status === "running",
);

const stageLabel = computed(() => {
    const label =
        props.row.stage === "failed"
            ? "failed"
            : props.row.stage === "done"
              ? "done"
              : props.row.stage;
    if (props.row.pluginState) return `${label} · ${props.row.pluginState}`;
    return label;
});
</script>

<template>
    <div
        class="flex items-center gap-2 rounded-md border p-2 text-xs"
        data-testid="ingest-row"
        :data-status="row.status"
    >
        <ChatFileIcon :filename="row.filename" />
        <div class="min-w-0 flex-1">
            <div class="flex items-center gap-2">
                <span
                    class="truncate font-medium"
                    data-testid="ingest-filename"
                >
                    {{ row.filename }}
                </span>
                <LoaderCircle
                    v-if="running"
                    class="spin-icon h-3.5 w-3.5 shrink-0 text-muted-foreground"
                    data-testid="ingest-spinner"
                />
                <Check
                    v-else-if="row.status === 'done'"
                    class="h-3.5 w-3.5 shrink-0 text-primary"
                    data-testid="ingest-check"
                />
                <X
                    v-else
                    class="h-3.5 w-3.5 shrink-0 text-destructive"
                    data-testid="ingest-cross"
                />
            </div>
            <p class="text-muted-foreground" data-testid="ingest-stage">
                {{ stageLabel }}
                <template
                    v-if="row.status === 'done' && row.chunkCount !== null"
                >
                    · {{ row.chunkCount }} chunk{{
                        row.chunkCount === 1 ? "" : "s"
                    }}
                </template>
            </p>
            <p
                v-if="row.error"
                class="mt-0.5 text-destructive"
                data-testid="ingest-error"
            >
                {{ row.error }}
            </p>
            <Progress :model-value="row.progress" class="mt-1.5 h-1">
                <ProgressIndicator
                    :style="{
                        transformOrigin: 'left',
                        transform: `scaleX(${Math.min(row.progress, 100) / 100})`,
                    }"
                />
            </Progress>
        </div>
    </div>
</template>
