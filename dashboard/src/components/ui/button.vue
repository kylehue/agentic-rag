<script setup lang="ts">
import { computed } from "vue";
import { cn } from "@/lib/utils";

export type ButtonVariant =
    "default" | "secondary" | "destructive" | "outline" | "ghost" | "link";

export type ButtonSize = "default" | "sm" | "lg" | "icon";

const props = withDefaults(
    defineProps<{
        variant?: ButtonVariant;
        size?: ButtonSize;
    }>(),
    {
        variant: "default",
        size: "default",
    },
);

const variantClasses: Record<ButtonVariant, string> = {
    default: "bg-primary text-primary-foreground hover:bg-primary/90",
    secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80",
    destructive:
        "bg-destructive text-primary-foreground hover:bg-destructive/90",
    outline:
        "border border-input bg-background hover:bg-accent hover:text-accent-foreground",
    ghost: "hover:bg-accent hover:text-accent-foreground",
    link: "text-primary underline-offset-4 hover:underline",
};

const sizeClasses: Record<ButtonSize, string> = {
    default: "h-9 px-4 py-2",
    sm: "h-8 rounded-md px-3 text-xs",
    lg: "h-10 rounded-md px-8",
    icon: "h-9 w-9",
};

const classes = computed(() =>
    cn(
        "inline-flex cursor-pointer items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:shrink-0",
        variantClasses[props.variant],
        sizeClasses[props.size],
    ),
);
</script>

<template>
    <button v-bind="$attrs" :class="classes">
        <slot />
    </button>
</template>
