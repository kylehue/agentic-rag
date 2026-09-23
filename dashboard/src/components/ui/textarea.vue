<script setup lang="ts">
import { computed } from "vue";
import { cn } from "@/lib/utils";

const props = defineProps<{
    modelValue?: string | number | null;
}>();

const emit = defineEmits<{
    "update:modelValue": [value: string];
}>();

const classes = computed(() =>
    cn(
        "flex min-h-[60px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50",
    ),
);

function onInput(event: Event): void {
    emit("update:modelValue", (event.target as HTMLTextAreaElement).value);
}
</script>

<template>
    <textarea
        v-bind="$attrs"
        :class="classes"
        :value="props.modelValue ?? ''"
        @input="onInput"
    />
</template>
