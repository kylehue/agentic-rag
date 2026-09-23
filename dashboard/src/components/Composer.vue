<script setup lang="ts">
import { Send, Square } from "@lucide/vue";
import { computed, nextTick, onMounted, ref } from "vue";
import { useAppStore } from "@/store/app";
import Button from "./ui/button.vue";

const store = useAppStore();

const text = ref("");
const textareaRef = ref<HTMLTextAreaElement | null>(null);

const placeholder = computed(() =>
    store.activeChatId
        ? "Ask a question about your documents"
        : "Start a new conversation",
);

// Auto-growing textarea: one line up to a max height.
function autoGrow(): void {
    const el = textareaRef.value;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
}

function submit(): void {
    const question = text.value.trim();
    if (!question || store.isStreaming) return;
    text.value = "";
    nextTick(autoGrow);
    void store.ask(question);
}

function onKeydown(event: KeyboardEvent): void {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        submit();
    }
}

onMounted(() => {
    // A recovered draft pre-fills the composer.
    if (store.recoveredDraft) {
        text.value = store.recoveredDraft;
        store.consumeRecoveredDraft();
        nextTick(autoGrow);
    }
});
</script>

<template>
    <div class="border-t bg-background p-3 md:p-4">
        <div class="mx-auto w-full max-w-3xl">
            <div class="rounded-lg border bg-card p-2 shadow-sm">
                <div class="flex items-end gap-2">
                    <textarea
                        ref="textareaRef"
                        v-model="text"
                        data-testid="composer-input"
                        rows="1"
                        :placeholder="placeholder"
                        :disabled="store.isStreaming"
                        class="scrollbar-thin max-h-40 min-h-8 flex-1 resize-none bg-transparent py-1 text-sm outline-none placeholder:text-muted-foreground disabled:cursor-not-allowed"
                        @input="autoGrow"
                        @keydown="onKeydown"
                    />
                    <Button
                        v-if="!store.isStreaming"
                        data-testid="composer-send"
                        size="icon"
                        :disabled="!text.trim()"
                        aria-label="Send"
                        @click="submit"
                    >
                        <Send class="h-4 w-4" />
                    </Button>
                    <Button
                        v-else
                        data-testid="composer-stop"
                        size="icon"
                        variant="outline"
                        aria-label="Stop"
                        :disabled="store.stopping"
                        @click="store.stop()"
                    >
                        <Square class="h-4 w-4" />
                    </Button>
                </div>
            </div>
            <p class="mt-1.5 px-1 text-xs text-muted-foreground">
                Enter to send · Shift+Enter for a new line
            </p>
        </div>
    </div>
</template>
