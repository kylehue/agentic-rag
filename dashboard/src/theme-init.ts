// Applies the theme before first paint. This module runs at load time,
// outside the app boot, so it reads the persisted preference directly.
// The storage key duplicates the one in lib/theme.ts: renaming it requires
// updating both places.
const THEME_KEY = "rag:theme";

const root = document.documentElement;

let pref: "light" | "dark" | "system" = "system";
try {
    const raw = window.localStorage.getItem(THEME_KEY);
    if (raw === "light" || raw === "dark" || raw === "system") pref = raw;
} catch {
    // storage unavailable; fall back to system
}

const dark =
    pref === "dark" ||
    (pref !== "light" &&
        window.matchMedia("(prefers-color-scheme: dark)").matches);

root.classList.toggle("dark", dark);
