import {
    DialogClose as DialogClosePrimitive,
    DialogContent as DialogContentPrimitive,
    DialogDescription as DialogDescriptionPrimitive,
    DialogOverlay as DialogOverlayPrimitive,
    DialogPortal as DialogPortalPrimitive,
    DialogRoot,
    DialogTitle as DialogTitlePrimitive,
    DialogTrigger as DialogTriggerPrimitive,
} from "reka-ui";
import { h, type HTMLAttributes, type VNode } from "vue";
import { cn } from "@/lib/utils";

type DialogProps = { class?: HTMLAttributes["class"] };
type DialogCtx = {
    attrs: Record<string, unknown>;
    slots: { default?: () => VNode[] };
};

export const Dialog = DialogRoot;
export const DialogTrigger = DialogTriggerPrimitive;
export const DialogPortal = DialogPortalPrimitive;
export const DialogClose = DialogClosePrimitive;

export const DialogOverlay = (
    props: DialogProps,
    { attrs, slots }: DialogCtx,
) =>
    h(
        DialogOverlayPrimitive,
        {
            ...attrs,
            as: "div",
            class: cn(
                "fixed inset-0 z-50 bg-black/60 backdrop-blur-sm",
                props.class,
            ),
        },
        slots.default?.(),
    );

export const DialogContent = (
    props: DialogProps,
    { attrs, slots }: DialogCtx,
) =>
    h(
        DialogContentPrimitive,
        {
            ...attrs,
            as: "div",
            class: cn(
                "animate-fade-scale fixed left-1/2 top-1/2 z-50 grid w-[calc(100%-2rem)] max-w-lg -translate-x-1/2 -translate-y-1/2 gap-4 rounded-xl border bg-background p-6 shadow-lg",
                props.class,
            ),
        },
        slots.default?.(),
    );

export const DialogHeader = (props: DialogProps, { attrs, slots }: DialogCtx) =>
    h(
        "div",
        {
            ...attrs,
            class: cn(
                "flex flex-col space-y-1.5 text-center sm:text-left",
                props.class,
            ),
        },
        slots.default?.(),
    );

export const DialogFooter = (props: DialogProps, { attrs, slots }: DialogCtx) =>
    h(
        "div",
        {
            ...attrs,
            class: cn(
                "flex flex-col-reverse gap-2 sm:flex-row sm:justify-end sm:space-x-2",
                props.class,
            ),
        },
        slots.default?.(),
    );

export const DialogTitle = (props: DialogProps, { attrs, slots }: DialogCtx) =>
    h(
        DialogTitlePrimitive,
        { ...attrs, class: cn("text-lg font-semibold", props.class) },
        slots.default?.(),
    );

export const DialogDescription = (
    props: DialogProps,
    { attrs, slots }: DialogCtx,
) =>
    h(
        DialogDescriptionPrimitive,
        { ...attrs, class: cn("text-sm text-muted-foreground", props.class) },
        slots.default?.(),
    );
