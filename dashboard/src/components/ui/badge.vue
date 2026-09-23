<script setup lang="ts">
import { computed } from "vue";
import { cn } from "@/lib/utils";

export type BadgeVariant = "default" | "secondary" | "destructive" | "outline";

const props = withDefaults(
    defineProps<{
        variant?: BadgeVariant;
    }>(),
    { variant: "default" },
);

const variantClasses: Record<BadgeVariant, string> = {
    default: "border-transparent bg-primary text-primary-foreground",
    secondary: "border-transparent bg-secondary text-secondary-foreground",
    destructive: "border-transparent bg-destructive text-primary-foreground",
    outline: "text-foreground",
};

const classes = computed(() =>
    cn(
        "inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-semibold transition-colors",
        variantClasses[props.variant],
    ),
);
</script>

<template>
    <span v-bind="$attrs" :class="classes">
        <slot />
    </span>
</template>
