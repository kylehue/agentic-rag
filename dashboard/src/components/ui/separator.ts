import { Separator as SeparatorPrimitive } from "reka-ui";
import { h, type HTMLAttributes, type VNode } from "vue";
import { cn } from "@/lib/utils";

type SeparatorProps = {
    orientation?: "horizontal" | "vertical";
    class?: HTMLAttributes["class"];
};
type SeparatorCtx = {
    attrs: Record<string, unknown>;
    slots: { default?: () => VNode[] };
};

export const Separator = (
    props: SeparatorProps,
    { attrs, slots }: SeparatorCtx,
) =>
    h(
        SeparatorPrimitive,
        {
            ...attrs,
            as: "div",
            class: cn(
                "shrink-0 bg-border",
                !props.orientation || props.orientation === "horizontal"
                    ? "h-px w-full"
                    : "h-full w-px",
                props.class,
            ),
        },
        slots.default?.(),
    );
