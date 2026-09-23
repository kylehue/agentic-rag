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

type SheetProps = {
    side?: "top" | "right" | "bottom" | "left";
    class?: HTMLAttributes["class"];
};
type SheetCtx = {
    attrs: Record<string, unknown>;
    slots: { default?: () => VNode[] };
};

export const Sheet = DialogRoot;
export const SheetTrigger = DialogTriggerPrimitive;
export const SheetPortal = DialogPortalPrimitive;
export const SheetClose = DialogClosePrimitive;

export const SheetOverlay = (
    props: { class?: HTMLAttributes["class"] },
    { attrs, slots }: SheetCtx,
) =>
    h(
        DialogOverlayPrimitive,
        {
            ...attrs,
            as: "div",
            class: cn("fixed inset-0 z-50 bg-black/60", props.class),
        },
        slots.default?.(),
    );

const sideClasses: Record<NonNullable<SheetProps["side"]>, string> = {
    top: "inset-x-0 top-0 border-b",
    right: "inset-y-0 right-0 h-full w-3/4 max-w-[380px] border-l",
    bottom: "inset-x-0 bottom-0 border-t",
    left: "inset-y-0 left-0 h-full w-3/4 max-w-[380px] border-r",
};

export const SheetContent = (props: SheetProps, { attrs, slots }: SheetCtx) =>
    h(
        DialogContentPrimitive,
        {
            ...attrs,
            as: "div",
            "data-side": props.side,
            class: cn(
                "animate-fade-scale fixed z-50 flex w-full flex-col bg-background shadow-lg",
                sideClasses[props.side ?? "right"],
                props.class,
            ),
        },
        slots.default?.(),
    );

export const SheetHeader = (props: SheetProps, { attrs, slots }: SheetCtx) =>
    h(
        "div",
        { ...attrs, class: cn("flex flex-col space-y-2 p-4", props.class) },
        slots.default?.(),
    );

export const SheetFooter = (props: SheetProps, { attrs, slots }: SheetCtx) =>
    h(
        "div",
        { ...attrs, class: cn("mt-auto flex flex-col gap-2 p-4", props.class) },
        slots.default?.(),
    );

export const SheetTitle = (
    props: { class?: HTMLAttributes["class"] },
    { attrs, slots }: SheetCtx,
) =>
    h(
        DialogTitlePrimitive,
        { ...attrs, class: cn("text-base font-semibold", props.class) },
        slots.default?.(),
    );

export const SheetDescription = (
    props: { class?: HTMLAttributes["class"] },
    { attrs, slots }: SheetCtx,
) =>
    h(
        DialogDescriptionPrimitive,
        { ...attrs, class: cn("text-sm text-muted-foreground", props.class) },
        slots.default?.(),
    );
