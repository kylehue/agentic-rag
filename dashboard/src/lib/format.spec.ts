import { describe, expect, it } from "vitest";
import {
    byteSize,
    fileKind,
    relativeDate,
    shortId,
    truncateArgs,
} from "./format";

describe("relativeDate", () => {
    const now = new Date(2026, 5, 15, 12, 0, 0); // 2026-06-15 12:00 local
    const at = (year: number, month: number, day: number, hour = 12) =>
        new Date(year, month, day, hour, 0, 0).getTime() / 1000;

    it("shows the time for today", () => {
        expect(relativeDate(at(2026, 5, 15, 3), now)).toBe("03:00");
    });

    it("shows Yesterday for the previous day", () => {
        expect(relativeDate(at(2026, 5, 14, 20), now)).toBe("Yesterday");
    });

    it('shows "N d ago" within a week', () => {
        expect(relativeDate(at(2026, 5, 11), now)).toBe("4 d ago");
    });

    it("shows a short date beyond a week", () => {
        expect(relativeDate(at(2026, 4, 20), now)).toBe("May 20");
    });

    it("includes the year for older dates", () => {
        expect(relativeDate(at(2024, 4, 20), now)).toContain("2024");
    });

    it("renders future timestamps (clock skew) as Yesterday", () => {
        expect(relativeDate(at(2026, 6, 25), now)).toBe("Yesterday");
    });
});

describe("byteSize", () => {
    it("renders zero bytes as an empty string on purpose", () => {
        expect(byteSize(0)).toBe("");
    });

    it("formats bytes without a unit prefix below 1KB", () => {
        expect(byteSize(500)).toBe("500 B");
    });

    it("uses one decimal below 10 units", () => {
        expect(byteSize(1536)).toBe("1.5 KB");
        expect(byteSize(2 * 1024 * 1024)).toBe("2.0 MB");
    });

    it("rounds values of 10 units or more", () => {
        expect(byteSize(15 * 1024 * 1024)).toBe("15 MB");
    });
});

describe("fileKind", () => {
    it("derives the category from the content type", () => {
        expect(fileKind("application/pdf", "x.pdf")).toBe("pdf");
        expect(fileKind("text/csv", "x.csv")).toBe("spreadsheet");
        expect(
            fileKind(
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "a",
            ),
        ).toBe("spreadsheet");
        expect(fileKind("image/png", "x.png")).toBe("image");
        expect(fileKind("application/msword", "x.doc")).toBe("document");
        expect(fileKind("application/vnd.ms-powerpoint", "x.ppt")).toBe(
            "slides",
        );
        expect(fileKind("text/plain", "x.txt")).toBe("text");
        expect(fileKind("text/markdown", "x.md")).toBe("text");
    });

    it("falls back to the filename extension", () => {
        expect(fileKind("", "report.xlsx")).toBe("spreadsheet");
        expect(fileKind("application/octet-stream", "doc.docx")).toBe(
            "document",
        );
        expect(fileKind("application/octet-stream", "deck.pptx")).toBe(
            "slides",
        );
        expect(fileKind("application/octet-stream", "shot.jpeg")).toBe("image");
        expect(fileKind("", "whatever.bin")).toBe("other");
    });
});

describe("shortId", () => {
    it("takes the first eight characters", () => {
        expect(shortId("abcdef12-3456-7890")).toBe("abcdef12");
    });
});

describe("truncateArgs", () => {
    it("serializes and collapses whitespace", () => {
        expect(truncateArgs({ a: 1, b: [2, 3] })).toBe('{"a":1,"b":[2,3]}');
    });

    it("truncates long arguments with an ellipsis", () => {
        expect(truncateArgs({ q: "x".repeat(100) }, 20)).toMatch(/^.{20}…$/);
    });

    it("handles non-serializable values", () => {
        const value = truncateArgs(undefined);
        expect(value).toBe("{}");
    });
});
