<script setup lang="ts">
import { X } from "@lucide/vue";
import { computed } from "vue";
import {
    Toast,
    ToastClose,
    ToastDescription,
    ToastProvider,
    ToastTitle,
    ToastViewport,
} from "./toast";
import { useToast } from "./use-toast";

const { toasts } = useToast();

const items = computed(() =>
    toasts.map((t) => ({
        id: t.id,
        title: t.title,
        description: t.description,
        destructive: t.variant === "destructive",
    })),
);
</script>

<template>
    <ToastProvider :max="20">
        <template v-for="item of items" :key="item.id">
            <Toast
                :class="
                    item.destructive
                        ? 'border-destructive/50 bg-destructive/10 text-destructive'
                        : ''
                "
            >
                <div class="flex w-full items-start gap-3">
                    <X
                        v-if="item.destructive"
                        class="mt-0.5 h-4 w-4 shrink-0"
                    />
                    <div class="grid min-w-0 flex-1 gap-1">
                        <ToastTitle class="break-words">{{
                            item.title
                        }}</ToastTitle>
                        <ToastDescription
                            v-if="item.description"
                            class="break-words"
                        >
                            {{ item.description }}
                        </ToastDescription>
                    </div>
                </div>
                <ToastClose>
                    <X class="h-3.5 w-3.5" />
                </ToastClose>
            </Toast>
        </template>
        <ToastViewport
            class="fixed bottom-0 right-0 z-[100] flex max-h-screen w-full flex-col gap-2 p-4"
        />
    </ToastProvider>
</template>
