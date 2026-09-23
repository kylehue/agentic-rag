<script setup lang="ts">
import type { UserTurn } from "@/store/app";

defineProps<{
    turn: UserTurn;
}>();
</script>

<template>
    <div class="animate-rise flex justify-end" data-testid="user-message">
        <div
            class="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-sm bg-primary px-4 py-2 text-sm text-primary-foreground md:max-w-[70%]"
        >
            {{ turn.content }}
        </div>
    </div>
</template>
