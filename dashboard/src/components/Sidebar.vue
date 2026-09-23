<script setup lang="ts">
import { Bot, LogOut, Plus } from "@lucide/vue";
import { useAppStore } from "@/store/app";
import Button from "./ui/button.vue";
import { Avatar, AvatarFallback } from "./ui/avatar";
import ChatRow from "./ChatRow.vue";

const store = useAppStore();

function requestDeleteChat(chatId: string): void {
    store.requestConfirm({
        title: "Delete chat",
        description:
            "Delete this chat and everything in it: its files, chunks, vectors, and conversation history.",
        destructive: true,
        onConfirm: () => store.deleteChat(chatId),
    });
}
</script>

<template>
    <div class="flex h-full min-h-0 flex-col">
        <div class="flex items-center gap-2 px-4 py-4">
            <div
                class="flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-primary-foreground"
            >
                <Bot class="h-4 w-4" />
            </div>
            <span class="text-sm font-semibold">Agent</span>
        </div>

        <div class="px-3">
            <Button
                data-testid="new-chat"
                variant="outline"
                class="w-full justify-start"
                @click="store.newChat()"
            >
                <Plus class="h-4 w-4" />
                New chat
            </Button>
        </div>

        <nav
            aria-label="Chats"
            class="scrollbar-thin mt-3 min-h-0 flex-1 overflow-y-auto px-3 pb-3"
        >
            <p
                v-if="store.chats.length === 0"
                class="px-2 py-8 text-center text-xs text-muted-foreground"
            >
                No chats yet.
                <br />
                Ask a question to start one.
            </p>
            <ChatRow
                v-for="chat in store.chats"
                :key="chat.chat_id"
                :chat="chat"
                :active="chat.chat_id === store.activeChatId"
                @select="store.selectChat(chat.chat_id)"
                @delete="requestDeleteChat(chat.chat_id)"
            />
        </nav>

        <div class="flex items-center gap-2 border-t p-3">
            <Avatar class="h-8 w-8">
                <AvatarFallback class="text-xs font-semibold">
                    {{ store.username.slice(0, 1).toUpperCase() }}
                </AvatarFallback>
            </Avatar>
            <span class="min-w-0 flex-1 truncate text-sm">{{
                store.username
            }}</span>
            <Button
                data-testid="sign-out"
                variant="ghost"
                size="icon"
                class="h-8 w-8 text-muted-foreground hover:text-foreground"
                aria-label="Sign out"
                @click="store.logout()"
            >
                <LogOut class="h-4 w-4" />
            </Button>
        </div>
    </div>
</template>
