<script setup lang="ts">
import { FolderOpen } from "@lucide/vue";
import { computed } from "vue";
import { useAppStore } from "@/store/app";
import Badge from "./ui/badge.vue";
import FileRow from "./FileRow.vue";
import IngestRow from "./IngestRow.vue";
import Skeleton from "./ui/skeleton.vue";
import UploadZone from "./UploadZone.vue";

const store = useAppStore();

// Transient ingest rows, in upload order (object key order).
const ingestRows = computed(() => Object.values(store.ingests));
</script>

<template>
    <div class="flex h-full min-h-0 flex-col" data-testid="sources-panel">
        <div class="flex items-center gap-2 border-b p-4">
            <FolderOpen class="h-4 w-4" />
            <h2 class="text-sm font-semibold">Sources</h2>
            <Badge class="ml-auto" data-testid="sources-total">
                {{ store.totalFileCount }}
            </Badge>
        </div>

        <div
            class="scrollbar-thin min-h-0 flex-1 space-y-4 overflow-y-auto p-4"
        >
            <UploadZone />

            <section v-if="ingestRows.length > 0" class="space-y-2">
                <h3 class="text-xs font-medium text-muted-foreground">
                    Ingesting
                </h3>
                <IngestRow
                    v-for="row in ingestRows"
                    :key="row.filename"
                    :row="row"
                />
            </section>

            <section class="space-y-2">
                <h3 class="text-xs font-medium text-muted-foreground">Files</h3>
                <Skeleton v-if="store.filesLoading" class="h-12 w-full" />
                <p
                    v-else-if="
                        store.activeChatFiles.length === 0 &&
                        ingestRows.length === 0
                    "
                    class="rounded-md border border-dashed p-4 text-center text-xs text-muted-foreground"
                >
                    No files yet.
                </p>
                <FileRow
                    v-else
                    v-for="file in store.activeChatFiles"
                    :key="file.source_id"
                    :file="file"
                />
            </section>
        </div>
    </div>
</template>
