import { readonly, reactive } from "vue";

export type ToastVariant = "default" | "destructive";

export interface ToastItem {
    id: number;
    title: string;
    description?: string;
    variant: ToastVariant;
}

export interface ToastOptions {
    title: string;
    description?: string;
    variant?: ToastVariant;
}

const state = reactive<{ toasts: ToastItem[] }>({ toasts: [] });

let nextId = 1;

function dismiss(id: number): void {
    const index = state.toasts.findIndex((t) => t.id === id);
    if (index !== -1) state.toasts.splice(index, 1);
}

export function toast(options: ToastOptions): number {
    const item: ToastItem = {
        id: nextId++,
        title: options.title,
        description: options.description,
        variant: options.variant ?? "default",
    };
    state.toasts.push(item);
    return item.id;
}

export function dismissToast(id: number): void {
    dismiss(id);
}

// Error toasts from the API client.
export function toastError(options: { status: number; detail: string }): void {
    toast({
        title: "Error",
        description: options.detail,
        variant: "destructive",
    });
}

export function useToast() {
    return {
        toasts: readonly(state.toasts),
        toast,
        dismiss,
    };
}
