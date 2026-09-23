<script setup lang="ts">
import ChatHeader from "./ChatHeader.vue";
import MessageList from "./MessageList.vue";
import Composer from "./Composer.vue";
</script>

<template>
    <div class="flex min-h-0 flex-1 flex-col" data-testid="chat-view">
        <ChatHeader />
        <MessageList />
        <Composer />
    </div>
</template>
