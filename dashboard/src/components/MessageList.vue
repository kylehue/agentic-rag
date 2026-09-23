<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from "vue";
import { MessagesSquare } from "@lucide/vue";
import { useAppStore } from "@/store/app";
import AssistantMessage from "./AssistantMessage.vue";
import UserMessage from "./UserMessage.vue";
import Button from "./ui/button.vue";
import Skeleton from "./ui/skeleton.vue";

const store = useAppStore();

const scrollRef = ref<HTMLElement | null>(null);
// Sticky to the bottom while the user stays near it; detaches on scroll up.
const stickToBottom = ref(true);

const NEAR_BOTTOM = 48;

const signature = computed(() => {
    const last = store.messages[store.messages.length - 1];
    const lastLength =
        last && last.kind === "assistant" ? (last.answer?.length ?? 0) : 0;
    return `${store.messages.length}:${lastLength}:${store.streaming}`;
});

function onScroll(): void {
    const el = scrollRef.value;
    if (!el) return;
    stickToBottom.value =
        el.scrollHeight - el.scrollTop - el.clientHeight < NEAR_BOTTOM;
}

async function scrollDown(force: boolean): Promise<void> {
    const el = scrollRef.value;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
    if (force) stickToBottom.value = true;
    await nextTick();
}

watch(signature, () => {
    if (stickToBottom.value) void scrollDown(false);
});

// New questions and finished history loads force the view back down.
watch(
    () => store.scrollForce,
    () => {
        void scrollDown(true);
    },
);

watch(
    () => store.historyLoading,
    (loading) => {
        if (!loading) void scrollDown(true);
    },
);

onMounted(() => {
    void scrollDown(true);
});
</script>

<template>
    <div
        ref="scrollRef"
        class="scrollbar-thin min-h-0 flex-1 overflow-y-auto"
        data-testid="message-list"
        @scroll.passive="onScroll"
    >
        <div class="mx-auto flex w-full max-w-3xl flex-col gap-4 p-4 md:p-6">
            <template v-if="store.historyLoading">
                <div class="flex justify-end">
                    <Skeleton class="h-10 w-1/3" />
                </div>
                <div class="flex flex-col gap-2">
                    <Skeleton class="h-4 w-1/2" />
                    <Skeleton class="h-4 w-2/3" />
                    <Skeleton class="h-4 w-1/3" />
                </div>
                <div class="flex justify-end">
                    <Skeleton class="h-10 w-1/4" />
                </div>
                <div class="flex flex-col gap-2">
                    <Skeleton class="h-4 w-3/4" />
                    <Skeleton class="h-4 w-2/3" />
                </div>
            </template>

            <div
                v-else-if="store.messages.length === 0"
                class="flex flex-1 flex-col items-center justify-center gap-3 py-16 text-center"
                data-testid="empty-state"
            >
                <div
                    class="flex h-12 w-12 items-center justify-center rounded-full bg-muted"
                >
                    <MessagesSquare class="h-6 w-6 text-muted-foreground" />
                </div>
                <h2 class="text-base font-medium">Ask your documents</h2>
                <p class="max-w-xs text-sm text-muted-foreground">
                    Upload files in the sources panel, then ask a question about
                    them.
                </p>
                <Button
                    variant="outline"
                    size="sm"
                    data-testid="add-files"
                    @click="store.setSourcesOpen(true)"
                >
                    Add files
                </Button>
            </div>

            <template v-else>
                <template v-for="turn in store.messages" :key="turn.id">
                    <UserMessage v-if="turn.kind === 'user'" :turn="turn" />
                    <AssistantMessage v-else :turn="turn" />
                </template>
            </template>
        </div>
    </div>
</template>
