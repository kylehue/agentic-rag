<script setup lang="ts">
import { Bot, LoaderCircle } from "@lucide/vue";
import { computed, ref } from "vue";
import { useAppStore } from "@/store/app";
import Button from "./ui/button.vue";
import {
    Card,
    CardContent,
    CardDescription,
    CardHeader,
    CardTitle,
} from "./ui/card";
import Input from "./ui/input.vue";
import Label from "./ui/label.vue";

const store = useAppStore();

const mode = ref<"login" | "register">("login");
const username = ref("");
const password = ref("");
const busy = ref(false);
const error = ref("");

const usernameValid = computed(
    () =>
        username.value.trim().length >= 3 && username.value.trim().length <= 64,
);
const passwordValid = computed(() => password.value.length >= 4);
const canSubmit = computed(
    () => !busy.value && usernameValid.value && passwordValid.value,
);

async function submit(): Promise<void> {
    if (!canSubmit.value) return;
    busy.value = true;
    error.value = "";
    const detail =
        mode.value === "login"
            ? await store.login(username.value.trim(), password.value)
            : await store.register(username.value.trim(), password.value);
    busy.value = false;
    if (detail) error.value = detail;
}

function toggleMode(): void {
    mode.value = mode.value === "login" ? "register" : "login";
    error.value = "";
}
</script>

<template>
    <div class="flex min-h-screen items-center justify-center p-4">
        <Card class="animate-rise w-full max-w-sm">
            <CardHeader class="items-center text-center">
                <div
                    class="mb-2 flex h-10 w-10 items-center justify-center rounded-lg bg-primary text-primary-foreground"
                >
                    <Bot class="h-5 w-5" />
                </div>
                <CardTitle>Agent</CardTitle>
                <CardDescription>
                    {{
                        mode === "login"
                            ? "Sign in to your account"
                            : "Create an account"
                    }}
                </CardDescription>
            </CardHeader>
            <CardContent>
                <form class="flex flex-col gap-4" @submit.prevent="submit">
                    <div class="flex flex-col gap-2">
                        <Label for="auth-username">Username</Label>
                        <Input
                            id="auth-username"
                            v-model="username"
                            :disabled="busy"
                            autocomplete="username"
                            placeholder="your name"
                        />
                        <p
                            v-if="username.length > 0 && !usernameValid"
                            class="text-xs text-destructive"
                        >
                            3 to 64 characters.
                        </p>
                    </div>
                    <div class="flex flex-col gap-2">
                        <Label for="auth-password">Password</Label>
                        <Input
                            id="auth-password"
                            v-model="password"
                            type="password"
                            :disabled="busy"
                            :autocomplete="
                                mode === 'login'
                                    ? 'current-password'
                                    : 'new-password'
                            "
                            placeholder="secret"
                        />
                        <p
                            v-if="password.length > 0 && !passwordValid"
                            class="text-xs text-destructive"
                        >
                            At least 4 characters.
                        </p>
                    </div>
                    <p
                        v-if="error"
                        class="text-sm text-destructive"
                        data-testid="auth-error"
                    >
                        {{ error }}
                    </p>
                    <Button
                        data-testid="auth-submit"
                        type="submit"
                        :disabled="!canSubmit"
                        class="w-full"
                    >
                        <LoaderCircle v-if="busy" class="spin-icon h-4 w-4" />
                        {{ mode === "login" ? "Sign in" : "Create account" }}
                    </Button>
                    <button
                        type="button"
                        class="text-xs text-muted-foreground underline underline-offset-4 hover:text-foreground"
                        data-testid="auth-toggle"
                        @click="toggleMode"
                    >
                        {{
                            mode === "login"
                                ? "No account? Create one"
                                : "Have an account? Sign in"
                        }}
                    </button>
                </form>
            </CardContent>
        </Card>
    </div>
</template>
