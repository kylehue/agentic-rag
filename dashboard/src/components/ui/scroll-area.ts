import {
    ScrollAreaRoot,
    ScrollAreaScrollbar as ScrollAreaScrollbarPrimitive,
    ScrollAreaThumb as ScrollAreaThumbPrimitive,
    ScrollAreaViewport as ScrollAreaViewportPrimitive,
} from "reka-ui";
import { h, type HTMLAttributes, type VNode } from "vue";
import { cn } from "@/lib/utils";

type ScrollAreaProps = {
    orientation?: "vertical" | "horizontal";
    class?: HTMLAttributes["class"];
};
type ScrollAreaCtx = {
    attrs: Record<string, unknown>;
    slots: { default?: () => VNode[] };
};

export const ScrollArea = ScrollAreaRoot;
export const ScrollAreaViewport = ScrollAreaViewportPrimitive;

export const ScrollAreaScrollbar = (
    props: ScrollAreaProps,
    { attrs, slots }: ScrollAreaCtx,
) =>
    h(
        ScrollAreaScrollbarPrimitive,
        {
            ...attrs,
            class: cn(
                "flex touch-none select-none transition-colors",
                props.orientation === "horizontal"
                    ? "h-full flex-col border-t border-t-transparent p-[1px]"
                    : "h-full w-2.5 flex-col border-l border-l-transparent p-[1px]",
                props.class,
            ),
        },
        slots.default?.(),
    );

export const ScrollAreaThumb = (
    props: ScrollAreaProps,
    { attrs, slots }: ScrollAreaCtx,
) =>
    h(
        ScrollAreaThumbPrimitive,
        {
            ...attrs,
            class: cn("relative flex-1 rounded-full bg-border", props.class),
        },
        slots.default?.(),
    );
