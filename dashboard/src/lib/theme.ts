import { safeGet, safeSet } from "./storage";

export type ThemePref = "light" | "dark" | "system";

export const THEME_KEY = "rag:theme";

export function readThemePref(): ThemePref {
    const raw = safeGet(THEME_KEY);
    return raw === "light" || raw === "dark" || raw === "system"
        ? raw
        : "system";
}

export function saveThemePref(pref: ThemePref): void {
    safeSet(THEME_KEY, pref);
}

function systemDark(): boolean {
    if (typeof window === "undefined") return false;
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
}

export function resolveDark(pref: ThemePref): boolean {
    if (pref === "dark") return true;
    if (pref === "light") return false;
    return systemDark();
}

// The header button toggles between light and dark; it never selects system.
export function nextTheme(pref: ThemePref): ThemePref {
    return pref === "light" ? "dark" : "light";
}

let mediaListener: ((event: MediaQueryListEvent) => void) | null = null;

export function applyTheme(pref: ThemePref): void {
    if (typeof document === "undefined") return;
    document.documentElement.classList.toggle("dark", resolveDark(pref));
    watchSystem(pref);
}

// OS preference changes are tracked only while "system" is selected.
function watchSystem(pref: ThemePref): void {
    if (typeof window === "undefined") return;
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    if (mediaListener) {
        media.removeEventListener("change", mediaListener);
        mediaListener = null;
    }
    if (pref !== "system") return;
    mediaListener = (event) => {
        document.documentElement.classList.toggle("dark", event.matches);
    };
    media.addEventListener("change", mediaListener);
}

export function resetThemeWatcher(): void {
    if (typeof window !== "undefined" && mediaListener) {
        window
            .matchMedia("(prefers-color-scheme: dark)")
            .removeEventListener("change", mediaListener);
    }
    mediaListener = null;
}
