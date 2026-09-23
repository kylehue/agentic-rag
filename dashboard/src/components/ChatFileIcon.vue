<script setup lang="ts">
import { computed } from "vue";
import {
    File,
    FileImage,
    FileSpreadsheet,
    FileText,
    Presentation,
} from "@lucide/vue";
import { fileKind, type FileKind } from "@/lib/format";

const props = defineProps<{
    filename: string;
    contentType?: string | null;
}>();

const kind = computed<FileKind>(() =>
    fileKind(props.contentType ?? "", props.filename),
);

const icon = computed(() => {
    switch (kind.value) {
        case "spreadsheet":
            return FileSpreadsheet;
        case "image":
            return FileImage;
        case "slides":
            return Presentation;
        case "pdf":
        case "document":
        case "text":
            return FileText;
        default:
            return File;
    }
});
</script>

<template>
    <component :is="icon" class="h-4 w-4 shrink-0" :aria-hidden="true" />
</template>
