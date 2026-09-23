import type { VueWrapper } from "@vue/test-utils";

const mounted: VueWrapper<unknown>[] = [];

export function trackMount<T extends VueWrapper<unknown>>(wrapper: T): T {
    mounted.push(wrapper);
    return wrapper;
}

// The test DOM is shared across specs in a file; unmount tracked wrappers
// between tests so effects and timers do not leak.
export function unmountAll(): void {
    while (mounted.length > 0) {
        const wrapper = mounted.pop();
        if (wrapper) {
            try {
                wrapper.unmount();
            } catch {
                // already gone
            }
        }
    }
}
