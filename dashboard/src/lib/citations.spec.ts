import { describe, expect, it } from "vitest";
import { buildCitations, citationCount, citationIndex } from "./citations";

const origin = "3ae82d8c-1b53-47a8-a723-9dbbd0ec520f";
const chunkA = "f97b9a7d-5328-4aec-86b3-6ee42d16db24";
const chunkB = "aa11aa11-2222-3333-4444-555566667777";

describe("citationIndex", () => {
    it("reads the marker number", () => {
        expect(citationIndex("#[1]")).toBe(1);
        expect(citationIndex("#[12]")).toBe(12);
    });

    it("rejects non-marker refs", () => {
        expect(citationIndex("#[o1:c1]")).toBeNull();
        expect(citationIndex("#abc")).toBeNull();
        expect(citationIndex("#[]")).toBeNull();
        expect(citationIndex("#[1]x")).toBeNull();
    });
});

describe("citationCount", () => {
    const ref = { origin_source_id: "o1", chunk_id: "c1" };

    it("counts resolved chunk_refs when present", () => {
        expect(
            citationCount("a #[1] b #[2]", { "#[1]": ref, "#[2]": ref }),
        ).toBe(2);
        expect(citationCount("a #[1]", { "#[1]": ref })).toBe(1);
        expect(citationCount("no markers", { "#[1]": ref })).toBe(1);
    });

    it("counts unique markers in the text while streaming", () => {
        expect(citationCount("a #[1] and again #[1]", null)).toBe(1);
        expect(citationCount("a #[1] b #[2] c #[1]", null)).toBe(2);
        expect(citationCount("partial #[1", null)).toBe(0);
    });

    it("is zero with no answer", () => {
        expect(citationCount(null, null)).toBe(0);
    });
});

describe("buildCitations", () => {
    const refs = {
        "#[1]": { origin_source_id: origin, chunk_id: chunkA },
        "#[2]": { origin_source_id: origin, chunk_id: chunkB },
    };

    it("numbers entries from their markers and resolves known chunks", () => {
        const citations = buildCitations(
            refs,
            (_originSourceId, chunkId) =>
                chunkId === chunkA
                    ? { text: "known text", page: 3, plugin: "text" }
                    : null,
            () => "file.txt",
        );
        expect(citations).toHaveLength(2);
        expect(citations[0]).toMatchObject({
            number: 1,
            refKey: "#[1]",
            chunkId: chunkA,
            filename: "file.txt",
            page: 3,
            plugin: "text",
            excerpt: "known text",
        });
        expect(citations[1]).toMatchObject({
            number: 2,
            refKey: "#[2]",
            chunkId: chunkB,
            excerpt: null,
            page: null,
            plugin: null,
        });
    });

    it("sorts rows by marker number regardless of key order", () => {
        const citations = buildCitations(
            { "#[2]": refs["#[2]"], "#[1]": refs["#[1]"] },
            () => null,
            () => null,
        );
        expect(citations.map((c) => c.number)).toEqual([1, 2]);
    });

    it("returns no citations when nothing is referenced", () => {
        expect(
            buildCitations(
                null,
                () => null,
                () => null,
            ),
        ).toEqual([]);
        expect(
            buildCitations(
                {},
                () => null,
                () => null,
            ),
        ).toEqual([]);
    });
});
