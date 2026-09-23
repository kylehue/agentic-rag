<script setup lang="ts">
import { useAppStore } from "@/store/app";
import Button from "./ui/button.vue";
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
} from "./ui/dialog";

const store = useAppStore();
</script>

<template>
    <Dialog
        v-if="store.confirm"
        :open="true"
        @update:open="store.cancelConfirm()"
    >
        <DialogContent class="max-w-sm" data-testid="confirm-dialog">
            <DialogHeader>
                <DialogTitle data-testid="confirm-title">{{
                    store.confirm.title
                }}</DialogTitle>
                <DialogDescription data-testid="confirm-description">
                    {{ store.confirm.description }}
                </DialogDescription>
            </DialogHeader>
            <DialogFooter>
                <Button
                    variant="outline"
                    data-testid="confirm-cancel"
                    @click="store.cancelConfirm()"
                >
                    Cancel
                </Button>
                <Button
                    data-testid="confirm-button"
                    :variant="
                        store.confirm.destructive ? 'destructive' : 'default'
                    "
                    @click="store.executeConfirm()"
                >
                    {{ store.confirm.destructive ? "Delete" : "Confirm" }}
                </Button>
            </DialogFooter>
        </DialogContent>
    </Dialog>
</template>
