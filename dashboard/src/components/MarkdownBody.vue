<script setup lang="ts">
import { computed, onMounted, onBeforeUnmount, ref } from "vue";
import { createMarkdownRenderer, renderMarkdown } from "@/lib/markdown";

const props = defineProps<{
    content: string;
    streaming?: boolean;
    // True when the answer cites a single chunk: the markers are then dropped
    // from the body (the sources footer keeps the citation).
    hideCitations?: boolean;
}>();

const emit = defineEmits<{
    citation: [number: number];
}>();

// Badge numbers come from the markers themselves (server-assigned), so no
// refs are needed at render time; click resolution happens in the store.
const md = createMarkdownRenderer();
const html = computed(() =>
    renderMarkdown(md, props.content, props.hideCitations),
);

const containerRef = ref<HTMLElement | null>(null);

// Badge clicks are handled by delegation on the container.
function handleClick(event: MouseEvent): void {
    const target =
        event.target instanceof HTMLElement
            ? event.target.closest<HTMLElement>(".citation-badge")
            : null;
    if (!target) return;
    const number = Number(target.dataset.cite);
    if (!Number.isNaN(number)) emit("citation", number);
}

onMounted(() => containerRef.value?.addEventListener("click", handleClick));
onBeforeUnmount(() =>
    containerRef.value?.removeEventListener("click", handleClick),
);
</script>

<template>
    <div
        ref="containerRef"
        class="markdown"
        :class="{ streaming }"
        data-testid="markdown-body"
        v-html="html"
    />
</template>
