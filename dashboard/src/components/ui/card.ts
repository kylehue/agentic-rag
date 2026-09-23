import { h, type HTMLAttributes, type VNode } from "vue";
import { cn } from "@/lib/utils";

type CardProps = { class?: HTMLAttributes["class"] };
type CardCtx = {
    attrs: Record<string, unknown>;
    slots: { default?: () => VNode[] };
};

export const Card = (props: CardProps, { attrs, slots }: CardCtx) =>
    h(
        "div",
        {
            ...attrs,
            class: cn(
                "rounded-xl border bg-card text-card-foreground shadow-sm",
                props.class,
            ),
        },
        slots.default?.(),
    );

export const CardHeader = (props: CardProps, { attrs, slots }: CardCtx) =>
    h(
        "div",
        { ...attrs, class: cn("flex flex-col space-y-1.5 p-4", props.class) },
        slots.default?.(),
    );

export const CardTitle = (props: CardProps, { attrs, slots }: CardCtx) =>
    h(
        "h3",
        {
            ...attrs,
            class: cn("font-semibold leading-none tracking-tight", props.class),
        },
        slots.default?.(),
    );

export const CardDescription = (props: CardProps, { attrs, slots }: CardCtx) =>
    h(
        "p",
        { ...attrs, class: cn("text-sm text-muted-foreground", props.class) },
        slots.default?.(),
    );

export const CardContent = (props: CardProps, { attrs, slots }: CardCtx) =>
    h(
        "div",
        { ...attrs, class: cn("p-4 pt-0", props.class) },
        slots.default?.(),
    );

export const CardFooter = (props: CardProps, { attrs, slots }: CardCtx) =>
    h(
        "div",
        { ...attrs, class: cn("flex items-center p-4 pt-0", props.class) },
        slots.default?.(),
    );
