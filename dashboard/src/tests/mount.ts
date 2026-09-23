import {
    mount as vtuMount,
    type ComponentMountingOptions,
    type VueWrapper,
} from "@vue/test-utils";
import type { Component, ComponentPublicInstance } from "vue";
import { trackMount } from "./registry";

export function mount<T extends ComponentPublicInstance>(
    component: Component,
    options?: ComponentMountingOptions<T>,
): VueWrapper<T> {
    return trackMount(vtuMount(component, options) as unknown as VueWrapper<T>);
}
