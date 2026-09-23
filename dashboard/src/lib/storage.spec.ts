import { describe, expect, it, vi } from "vitest";
import { safeGet, safeRemove, safeSet } from "./storage";

describe("storage helpers", () => {
    it("reads and writes normally", () => {
        safeSet("k", "v");
        expect(safeGet("k")).toBe("v");
        safeRemove("k");
        expect(safeGet("k")).toBeNull();
    });

    it("swallows storage errors", () => {
        const getItem = vi.fn(() => {
            throw new Error("denied");
        });
        const setItem = vi.fn(() => {
            throw new Error("quota");
        });
        const removeItem = vi.fn(() => {
            throw new Error("denied");
        });
        vi.stubGlobal("localStorage", {
            getItem,
            setItem,
            removeItem,
            clear: () => {},
        });
        expect(safeGet("k")).toBeNull();
        expect(() => safeSet("k", "v")).not.toThrow();
        expect(() => safeRemove("k")).not.toThrow();
        vi.unstubAllGlobals();
    });
});
