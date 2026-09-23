import { afterEach, vi } from "vitest";
import { config } from "@vue/test-utils";
import { unmountAll } from "./registry";

// The test DOM lacks matchMedia and ResizeObserver; both are required by
// the UI primitives.
if (typeof window !== "undefined") {
    if (!window.matchMedia) {
        Object.defineProperty(window, "matchMedia", {
            writable: true,
            value: (query: string) => ({
                matches: false,
                media: query,
                onchange: null,
                addListener: () => undefined,
                removeListener: () => undefined,
                addEventListener: () => undefined,
                removeEventListener: () => undefined,
                dispatchEvent: () => false,
            }),
        });
    }
    if (!window.ResizeObserver) {
        class ResizeObserverStub {
            observe() {}
            unobserve() {}
            disconnect() {}
        }
        window.ResizeObserver =
            ResizeObserverStub as unknown as typeof ResizeObserver;
    }
}

// CSS is not loaded in tests, so styling cannot be asserted there.
config.global.config.warnHandler = () => undefined;

afterEach(() => {
    unmountAll();
    // resetMocks does not reach implementations set on module-mocked vi.fn()s
    // (the api client), so reset everything explicitly.
    vi.resetAllMocks();
});
