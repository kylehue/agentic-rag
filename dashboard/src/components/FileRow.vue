<script setup lang="ts">
import { Eye, List, Trash2 } from "@lucide/vue";
import { computed } from "vue";
import type { StoredFile } from "@/api/types";
import { useAppStore } from "@/store/app";
import { FILE_KIND_LABELS, fileKind, shortId } from "@/lib/format";
import ChatFileIcon from "./ChatFileIcon.vue";
import Button from "./ui/button.vue";

const props = defineProps<{
    file: StoredFile;
}>();

const store = useAppStore();

const label = computed(
    () =>
        FILE_KIND_LABELS[
            fileKind(props.file.content_type, props.file.filename)
        ],
);

function requestDelete(): void {
    store.requestConfirm({
        title: "Delete file",
        description: `Delete ${props.file.filename} and all of its chunks? This cannot be undone.`,
        destructive: true,
        onConfirm: () => store.deleteFile(props.file.source_id),
    });
}
</script>

<template>
    <div
        class="flex items-center gap-2 rounded-md border p-2 text-xs"
        data-testid="file-row"
    >
        <ChatFileIcon
            :filename="file.filename"
            :content-type="file.content_type"
        />
        <button
            type="button"
            class="min-w-0 flex-1 truncate text-left font-medium hover:underline"
            :aria-label="`Preview ${file.filename}`"
            data-testid="file-name"
            @click="store.openFilePreview(file.source_id)"
        >
            {{ file.filename }}
        </button>
        <span
            class="hidden shrink-0 text-muted-foreground sm:inline"
            data-testid="file-kind"
        >
            {{ label }} · {{ shortId(file.source_id) }}
        </span>
        <div class="flex shrink-0 items-center gap-0.5">
            <Button
                variant="ghost"
                size="icon"
                class="h-6 w-6"
                :aria-label="`View ${file.filename}`"
                data-testid="file-view"
                @click="store.openFilePreview(file.source_id)"
            >
                <Eye class="h-3.5 w-3.5" />
            </Button>
            <Button
                variant="ghost"
                size="icon"
                class="h-6 w-6"
                :aria-label="`View chunks of ${file.filename}`"
                data-testid="file-chunks"
                @click="store.openFileChunks(file.source_id)"
            >
                <List class="h-3.5 w-3.5" />
            </Button>
            <Button
                variant="ghost"
                size="icon"
                class="h-6 w-6 hover:text-destructive"
                :aria-label="`Delete ${file.filename}`"
                data-testid="file-delete"
                @click="requestDelete"
            >
                <Trash2 class="h-3.5 w-3.5" />
            </Button>
        </div>
    </div>
</template>
