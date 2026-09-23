<script setup lang="ts">
import { Upload } from "@lucide/vue";
import { ref } from "vue";
import { useAppStore } from "@/store/app";

const store = useAppStore();

const inputRef = ref<HTMLInputElement | null>(null);
const dragDepth = ref(0);
const dragOver = ref(false);

function pick(files: FileList | null): void {
    if (!files || files.length === 0) return;
    void store.upload(Array.from(files));
}

function onInput(event: Event): void {
    pick((event.target as HTMLInputElement).files);
    // Clear the input so the same file can be re-selected.
    if (inputRef.value) inputRef.value.value = "";
}

function onDrop(event: DragEvent): void {
    dragDepth.value = 0;
    dragOver.value = false;
    pick(event.dataTransfer?.files ?? null);
}
</script>

<template>
    <div
        data-testid="upload-zone"
        class="flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed p-6 text-center outline-none transition-colors focus-visible:ring-2 focus-visible:ring-ring"
        :class="dragOver ? 'border-primary bg-primary/5' : 'hover:bg-accent/40'"
        role="button"
        tabindex="0"
        @click="inputRef?.click()"
        @keydown.enter.prevent="inputRef?.click()"
        @dragenter.prevent="
            dragDepth++;
            dragOver = true;
        "
        @dragover.prevent
        @dragleave.prevent="
            dragDepth--;
            if (dragDepth <= 0) {
                dragDepth = 0;
                dragOver = false;
            }
        "
        @drop.prevent="onDrop"
    >
        <Upload class="h-5 w-5 text-muted-foreground" />
        <p class="text-sm text-muted-foreground">
            <span class="font-medium text-foreground">Click to upload</span> or
            drag and drop
        </p>
        <p class="text-xs text-muted-foreground">
            PDF, Office, text, slides, and spreadsheets
        </p>
        <input
            ref="inputRef"
            type="file"
            multiple
            class="hidden"
            accept=".pdf,.docx,.doc,.odt,.rtf,.epub,.rst,.txt,.md,.html,.htm,.pptx,.ppt,.csv,.xlsx,.xls"
            data-testid="upload-input"
            @change="onInput"
        />
    </div>
</template>
