import { afterEach, describe, expect, it, vi } from "vitest";
import {
    applyTheme,
    nextTheme,
    readThemePref,
    resetThemeWatcher,
    resolveDark,
    saveThemePref,
    THEME_KEY,
} from "./theme";

describe("theme preferences", () => {
    afterEach(() => {
        window.localStorage.clear();
        resetThemeWatcher();
    });

    it("defaults to system when nothing is stored", () => {
        expect(readThemePref()).toBe("system");
    });

    it("round-trips stored preferences", () => {
        for (const pref of ["light", "dark", "system"] as const) {
            saveThemePref(pref);
            expect(readThemePref()).toBe(pref);
        }
    });

    it("ignores invalid stored values", () => {
        window.localStorage.setItem(THEME_KEY, "neon");
        expect(readThemePref()).toBe("system");
    });
});

describe("resolveDark", () => {
    it("resolves light and dark directly", () => {
        expect(resolveDark("light")).toBe(false);
        expect(resolveDark("dark")).toBe(true);
    });

    it("follows the OS in system mode", () => {
        vi.stubGlobal(
            "matchMedia",
            vi.fn().mockReturnValue({
                matches: true,
                media: "",
                addEventListener: () => {},
                removeEventListener: () => {},
            }),
        );
        expect(resolveDark("system")).toBe(true);
        vi.unstubAllGlobals();
    });
});

describe("applyTheme", () => {
    afterEach(() => {
        resetThemeWatcher();
        document.documentElement.className = "";
    });

    it("toggles the dark class on the root element", () => {
        vi.stubGlobal(
            "matchMedia",
            vi.fn().mockReturnValue({
                matches: false,
                media: "",
                addEventListener: () => {},
                removeEventListener: () => {},
            }),
        );
        applyTheme("dark");
        expect(document.documentElement.classList.contains("dark")).toBe(true);
        applyTheme("light");
        expect(document.documentElement.classList.contains("dark")).toBe(false);
        vi.unstubAllGlobals();
    });

    it("tracks OS changes only in system mode", () => {
        const listeners = new Set<(event: { matches: boolean }) => void>();
        vi.stubGlobal(
            "matchMedia",
            vi.fn().mockImplementation(() => ({
                matches: false,
                media: "",
                addEventListener: (
                    _type: string,
                    fn: (event: { matches: boolean }) => void,
                ) => listeners.add(fn),
                removeEventListener: (
                    _type: string,
                    fn: (event: { matches: boolean }) => void,
                ) => listeners.delete(fn),
            })),
        );
        applyTheme("system");
        expect(listeners.size).toBe(1);
        for (const fn of listeners) fn({ matches: true });
        expect(document.documentElement.classList.contains("dark")).toBe(true);
        applyTheme("light");
        expect(listeners.size).toBe(0);
        vi.unstubAllGlobals();
    });
});

describe("nextTheme", () => {
    it("toggles between light and dark", () => {
        expect(nextTheme("light")).toBe("dark");
        expect(nextTheme("dark")).toBe("light");
        expect(nextTheme("system")).toBe("light");
    });
});
