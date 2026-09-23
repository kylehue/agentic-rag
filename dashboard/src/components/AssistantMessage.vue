<script setup lang="ts">
import { computed } from "vue";
import { CircleAlert } from "@lucide/vue";
import { useAppStore, type AssistantTurn } from "@/store/app";
import { citationCount } from "@/lib/citations";
import MarkdownBody from "./MarkdownBody.vue";
import ToolSteps from "./ToolSteps.vue";
import SourcesBlock from "./SourcesBlock.vue";

const props = defineProps<{
    turn: AssistantTurn;
}>();

const store = useAppStore();

// A single-citation answer shows no inline badge; the sources footer is
// enough. (While streaming, the count is the unique markers seen so far,
// so a badge can appear and then vanish when the answer settles on one.)
const hideCitations = computed(
    () => citationCount(props.turn.answer, props.turn.chunkRefs) === 1,
);

function citationClick(number: number): void {
    void store.openCitation(props.turn, number);
}
</script>

<template>
    <div class="animate-rise w-full" data-testid="assistant-message">
        <div class="rounded-lg border bg-card p-4 shadow-sm">
            <ToolSteps :steps="turn.toolSteps" />
            <div
                v-if="turn.failed"
                class="mb-3 flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/10 p-2 text-sm text-destructive"
                data-testid="assistant-error"
            >
                <CircleAlert class="mt-0.5 h-4 w-4 shrink-0" />
                <span>{{ turn.error ?? "This run failed." }}</span>
            </div>

            <div
                v-if="turn.streaming && !turn.answer"
                class="flex items-center gap-1 text-sm text-muted-foreground"
            >
                <span data-testid="thinking">Thinking</span>
                <span class="caret" aria-hidden="true" />
            </div>
            <MarkdownBody
                v-else-if="turn.answer !== null"
                :content="turn.answer"
                :streaming="turn.streaming"
                :hide-citations="hideCitations"
                @citation="citationClick"
            />

            <SourcesBlock :turn="turn" />
        </div>
    </div>
</template>
