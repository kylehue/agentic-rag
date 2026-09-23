<script setup lang="ts">
import { LoaderCircle } from "@lucide/vue";
import { onMounted } from "vue";
import AuthScreen from "@/components/AuthScreen.vue";
import AppLayout from "@/components/AppLayout.vue";
import Toaster from "@/components/ui/Toaster.vue";
import { useAppStore } from "@/store/app";

const store = useAppStore();

onMounted(() => {
    void store.init();
});
</script>

<template>
    <div class="min-h-screen bg-background text-foreground">
        <div
            v-if="store.authStatus === 'loading'"
            class="flex min-h-screen items-center justify-center"
        >
            <div class="flex flex-col items-center gap-3">
                <LoaderCircle class="spin-icon h-8 w-8 text-muted-foreground" />
                <span class="text-sm text-muted-foreground">loading</span>
            </div>
        </div>
        <AuthScreen v-else-if="store.authStatus === 'guest'" />
        <AppLayout v-else />
        <Toaster />
    </div>
</template>
