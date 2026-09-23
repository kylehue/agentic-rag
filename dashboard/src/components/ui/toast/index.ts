import {
    ToastClose as ToastClosePrimitive,
    ToastDescription as ToastDescriptionPrimitive,
    ToastPortal as ToastPortalPrimitive,
    ToastProvider as ToastProviderPrimitive,
    ToastRoot as ToastRootPrimitive,
    ToastTitle as ToastTitlePrimitive,
    ToastViewport as ToastViewportPrimitive,
} from "reka-ui";
import { h, type HTMLAttributes, type VNode } from "vue";
import { cn } from "@/lib/utils";

type ToastProps = { class?: HTMLAttributes["class"] };
type ToastCtx = {
    attrs: Record<string, unknown>;
    slots: { default?: () => VNode[] };
};

export const ToastProvider = ToastProviderPrimitive;
export const ToastPortal = ToastPortalPrimitive;
export const ToastRoot = ToastRootPrimitive;

export const Toast = (props: ToastProps, { attrs, slots }: ToastCtx) =>
    h(
        ToastRootPrimitive,
        {
            ...attrs,
            class: cn(
                "pointer-events-auto relative flex w-full items-center justify-start space-x-2 rounded-md border p-4 pr-8",
                props.class,
            ),
        },
        slots.default?.(),
    );

export const ToastViewport = (props: ToastProps, { attrs, slots }: ToastCtx) =>
    h(
        ToastViewportPrimitive,
        {
            ...attrs,
            as: "div",
            class: cn(
                "group/toast-viewport z-[100] flex max-h-screen w-full flex-col gap-2",
                props.class,
            ),
        },
        slots.default?.(),
    );

export const ToastTitle = (props: ToastProps, { attrs, slots }: ToastCtx) =>
    h(
        ToastTitlePrimitive,
        {
            ...attrs,
            as: "div",
            class: cn("text-sm font-semibold", props.class),
        },
        slots.default?.(),
    );

export const ToastDescription = (
    props: ToastProps,
    { attrs, slots }: ToastCtx,
) =>
    h(
        ToastDescriptionPrimitive,
        { ...attrs, as: "div", class: cn("text-sm opacity-90", props.class) },
        slots.default?.(),
    );

export const ToastClose = (props: ToastProps, { attrs, slots }: ToastCtx) =>
    h(
        ToastClosePrimitive,
        {
            ...attrs,
            as: "button",
            type: "button",
            class: cn(
                "absolute right-2 top-2 rounded-md p-1 text-foreground/50 opacity-0 transition-opacity hover:text-foreground focus:opacity-100 group-hover/toast:opacity-100 group-focus-within/toast:opacity-100",
                props.class,
            ),
        },
        slots.default?.(),
    );
