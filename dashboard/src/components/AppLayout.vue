<script setup lang="ts">
import { useAppStore } from "@/store/app";
import Sidebar from "./Sidebar.vue";
import ChatView from "./ChatView.vue";
import SourcesPanel from "./SourcesPanel.vue";
import CitedChunkDialog from "./CitedChunkDialog.vue";
import ConfirmDialog from "./ConfirmDialog.vue";
import FileChunksDialog from "./FileChunksDialog.vue";
import FilePreviewDialog from "./FilePreviewDialog.vue";
import { Sheet, SheetContent, SheetDescription, SheetTitle } from "./ui/sheet";

const store = useAppStore();
</script>

<template>
    <div class="flex h-dvh overflow-hidden">
        <Sheet
            :open="store.mobileMenuOpen"
            @update:open="store.mobileMenuOpen = $event"
        >
            <SheetContent side="left" class="w-72 max-w-[85vw]">
                <div class="flex items-center justify-between pb-3">
                    <SheetTitle class="text-lg">Agent</SheetTitle>
                </div>
                <div class="min-h-0 flex-1">
                    <Sidebar />
                </div>
            </SheetContent>
        </Sheet>

        <aside
            class="hidden w-64 shrink-0 border-r bg-card md:block"
            data-testid="sidebar"
        >
            <Sidebar />
        </aside>

        <main class="flex min-w-0 flex-1 flex-col">
            <ChatView />
        </main>

        <Sheet
            :modal="false"
            :open="store.sourcesOpen"
            @update:open="store.setSourcesOpen($event)"
        >
            <SheetContent side="right" class="w-[380px] max-w-[90vw]">
                <SheetTitle class="sr-only">Sources</SheetTitle>
                <SheetDescription class="sr-only">
                    Your ingested files and their progress
                </SheetDescription>
                <SourcesPanel />
            </SheetContent>
        </Sheet>

        <CitedChunkDialog />
        <FileChunksDialog />
        <FilePreviewDialog />
        <ConfirmDialog />
    </div>
</template>
