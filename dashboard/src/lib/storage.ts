// localStorage can throw (private mode, quota exceeded). Every access goes
// through these so the app degrades instead of crashing.

export function safeGet(key: string): string | null {
    try {
        return window.localStorage.getItem(key);
    } catch {
        return null;
    }
}

export function safeSet(key: string, value: string): void {
    try {
        window.localStorage.setItem(key, value);
    } catch {
        // storage unavailable; persistence is best-effort
    }
}

export function safeRemove(key: string): void {
    try {
        window.localStorage.removeItem(key);
    } catch {
        // storage unavailable; nothing to clean
    }
}
