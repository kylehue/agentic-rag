<script setup lang="ts">
import { ref } from "vue";
import { Bot, ChevronDown, LoaderCircle } from "@lucide/vue";
import type { ToolStep } from "@/store/app";
import { truncateArgs } from "@/lib/format";

const props = defineProps<{
    steps: ToolStep[];
}>();

// Open by default, collapsible.
const open = ref(true);
const openResults = ref<Record<number, boolean>>({});

function toggleResult(index: number): void {
    openResults.value[index] = !openResults.value[index];
}

function argsFull(step: ToolStep): string {
    try {
        return JSON.stringify(step.arguments ?? {});
    } catch {
        return String(step.arguments);
    }
}
</script>

<template>
    <div
        v-if="steps.length > 0"
        class="mb-3 overflow-hidden rounded-md border bg-muted/40 text-xs"
        data-testid="tool-steps"
    >
        <button
            type="button"
            class="flex w-full items-center gap-2 px-3 py-2 text-left text-muted-foreground hover:bg-accent/50"
            :aria-expanded="open"
            data-testid="tool-steps-toggle"
            @click="open = !open"
        >
            <ChevronDown
                class="h-3.5 w-3.5 transition-transform"
                :class="{ 'rotate-180': !open }"
            />
            <Bot class="h-3.5 w-3.5" />
            <span class="font-medium">Agent activity</span>
            <span
                >({{ steps.length }} step{{
                    steps.length === 1 ? "" : "s"
                }})</span
            >
        </button>
        <ul v-if="open" class="space-y-2.5 border-t px-3 py-2.5">
            <li v-for="(step, index) in steps" :key="index" class="space-y-1">
                <div class="flex items-center gap-2">
                    <span
                        class="shrink-0 font-mono font-medium"
                        data-testid="tool-step-name"
                    >
                        {{ step.name }}
                    </span>
                    <span
                        v-if="step.arguments !== null"
                        class="min-w-0 truncate font-mono text-muted-foreground"
                        :title="argsFull(step)"
                        data-testid="tool-step-args"
                    >
                        {{ truncateArgs(step.arguments) }}
                    </span>
                    <LoaderCircle
                        v-if="step.result === null"
                        class="spin-icon ml-auto h-3 w-3 shrink-0 text-muted-foreground"
                    />
                </div>
                <template v-if="step.result !== null">
                    <button
                        type="button"
                        class="text-muted-foreground underline underline-offset-2 hover:text-foreground"
                        :aria-expanded="openResults[index] === true"
                        data-testid="tool-step-result-toggle"
                        @click="toggleResult(index)"
                    >
                        result ({{ step.result.length }} chars)
                    </button>
                    <pre
                        v-if="openResults[index]"
                        class="scrollbar-thin max-h-48 overflow-auto whitespace-pre-wrap rounded bg-background p-2 font-mono text-[11px] leading-relaxed"
                        data-testid="tool-step-result"
                    >
            {{ step.result }}</pre>
                </template>
            </li>
        </ul>
    </div>
</template>
