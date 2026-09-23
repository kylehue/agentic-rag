import {
    AvatarFallback as AvatarFallbackPrimitive,
    AvatarImage as AvatarImagePrimitive,
    AvatarRoot,
} from "reka-ui";
import { h, type HTMLAttributes, type VNode } from "vue";
import { cn } from "@/lib/utils";

type AvatarProps = { class?: HTMLAttributes["class"] };
type AvatarCtx = {
    attrs: Record<string, unknown>;
    slots: { default?: () => VNode[] };
};

export const Avatar = AvatarRoot;
export const AvatarImage = AvatarImagePrimitive;

export const AvatarFallback = (
    props: AvatarProps,
    { attrs, slots }: AvatarCtx,
) =>
    h(
        AvatarFallbackPrimitive,
        {
            ...attrs,
            class: cn(
                "flex h-full w-full items-center justify-center rounded-full bg-muted text-muted-foreground",
                props.class,
            ),
        },
        slots.default?.(),
    );
