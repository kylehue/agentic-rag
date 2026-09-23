<script setup lang="ts">
import { FolderOpen, Menu, Moon, Sun } from "@lucide/vue";
import { computed } from "vue";
import { useAppStore } from "@/store/app";
import { shortId } from "@/lib/format";
import Badge from "./ui/badge.vue";
import Button from "./ui/button.vue";

const store = useAppStore();

const title = computed(() =>
    store.activeChatId ? `Chat ${shortId(store.activeChatId)}` : "New chat",
);
const subtitle = computed(() =>
    store.durableFileCount > 0
        ? `${store.durableFileCount} source${store.durableFileCount === 1 ? "" : "s"}`
        : "No sources yet",
);
</script>

<template>
    <header class="flex items-center gap-2 border-b px-3 py-2.5 md:px-4">
        <Button
            variant="ghost"
            size="icon"
            class="md:hidden"
            aria-label="Open menu"
            data-testid="menu-toggle"
            @click="store.mobileMenuOpen = true"
        >
            <Menu class="h-4 w-4" />
        </Button>
        <div class="min-w-0 flex-1">
            <h1 class="truncate text-sm font-semibold" data-testid="chat-title">
                {{ title }}
            </h1>
            <p
                class="truncate text-xs text-muted-foreground"
                data-testid="chat-subtitle"
            >
                {{ subtitle }}
            </p>
        </div>
        <Button
            variant="ghost"
            size="icon"
            class="h-8 w-8 text-muted-foreground hover:text-foreground"
            :aria-label="
                store.theme === 'dark'
                    ? 'Switch to light theme'
                    : 'Switch to dark theme'
            "
            data-testid="theme-toggle"
            @click="store.toggleTheme()"
        >
            <Sun v-if="store.theme === 'dark'" class="h-4 w-4" />
            <Moon v-else class="h-4 w-4" />
        </Button>
        <Button
            variant="outline"
            size="sm"
            class="relative gap-2"
            :aria-expanded="store.sourcesOpen ? 'true' : 'false'"
            data-testid="open-sources"
            @click="store.setSourcesOpen(!store.sourcesOpen)"
        >
            <FolderOpen class="h-4 w-4" />
            <span class="hidden sm:inline">Sources</span>
            <Badge
                v-if="store.totalFileCount > 0"
                class="absolute -right-2 -top-2 h-5 min-w-5 px-1 text-[10px]"
                data-testid="sources-count"
            >
                {{ store.totalFileCount }}
            </Badge>
        </Button>
    </header>
</template>
