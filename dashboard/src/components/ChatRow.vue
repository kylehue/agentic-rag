<script setup lang="ts">
import { Trash2 } from "@lucide/vue";
import { computed } from "vue";
import type { ChatInfo } from "@/api/types";
import { relativeDate, shortId } from "@/lib/format";
import Button from "./ui/button.vue";

const props = defineProps<{
    chat: ChatInfo;
    active: boolean;
}>();

const emit = defineEmits<{
    select: [];
    delete: [];
}>();

const label = computed(() => `Chat ${shortId(props.chat.chat_id)}`);
const date = computed(() => relativeDate(props.chat.created_at));
</script>

<template>
    <div
        data-testid="chat-row"
        class="group relative flex w-full cursor-pointer items-center gap-2 rounded-md px-2 py-2 text-sm outline-none transition-colors hover:bg-accent focus-visible:bg-accent focus-visible:ring-2 focus-visible:ring-ring"
        :class="{ 'bg-accent text-accent-foreground': active }"
        role="button"
        :aria-current="active ? 'true' : undefined"
        tabindex="0"
        @click="emit('select')"
        @keydown.enter.prevent="emit('select')"
        @keydown.space.prevent="emit('select')"
    >
        <div class="min-w-0 flex-1">
            <p class="truncate font-medium">{{ label }}</p>
            <p class="truncate text-xs text-muted-foreground">
                {{ chat.sources.length }} source{{
                    chat.sources.length === 1 ? "" : "s"
                }}
                · {{ date }}
            </p>
        </div>
        <Button
            data-testid="chat-delete"
            variant="ghost"
            size="icon"
            class="absolute right-1 h-7 w-7 opacity-0 text-muted-foreground transition-opacity group-hover:opacity-100 group-focus-within:opacity-100 focus-visible:opacity-100 hover:text-destructive"
            :aria-label="`Delete ${label}`"
            @click.stop="emit('delete')"
            @keydown.enter.stop.prevent="emit('delete')"
        >
            <Trash2 class="h-3.5 w-3.5" />
        </Button>
    </div>
</template>
