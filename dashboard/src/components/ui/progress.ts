import {
    ProgressIndicator as ProgressIndicatorPrimitive,
    ProgressRoot,
} from "reka-ui";
import { h, type HTMLAttributes, type StyleValue, type VNode } from "vue";
import { cn } from "@/lib/utils";

type ProgressProps = { class?: HTMLAttributes["class"]; style?: StyleValue };
type ProgressCtx = {
    attrs: Record<string, unknown>;
    slots: { default?: () => VNode[] };
};

// Reka's root provides the a11y state; the sizing base classes live here.
export const Progress = (props: ProgressProps, { attrs, slots }: ProgressCtx) =>
    h(
        ProgressRoot,
        {
            ...attrs,
            class: cn(
                "relative h-2 w-full overflow-hidden rounded-full bg-muted",
                props.class,
            ),
        },
        slots.default?.(),
    );

// Reka's indicator only sets data-value/data-max; the fill transform is the
// consumer's job (pass it via the style attribute).
export const ProgressIndicator = (
    props: ProgressProps,
    { attrs, slots }: ProgressCtx,
) =>
    h(
        ProgressIndicatorPrimitive,
        {
            ...attrs,
            class: cn(
                "h-full w-full flex-1 rounded-full bg-primary transition-all duration-300",
                props.class,
            ),
        },
        slots.default?.(),
    );
