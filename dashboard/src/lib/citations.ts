// Citation helpers for the server's index-based markers: `#[1]`, `#[2]`, ...
//
// The service assigns the integer per answer and returns `chunk_refs` keyed
// by the marker string, each value holding the real
// `{chunk_id, origin_source_id}`. Markers that were never surfaced to the
// model are dropped server-side, so `chunk_refs` is authoritative for the
// sources list. Because markers carry their own number, the client does no
// numbering of its own.

import type { ChunkRefs } from "@/api/types";

/** The marker number behind a ref (`"#[3]"` -> 3), or null for a malformed ref. */
export function citationIndex(ref: string): number | null {
    const m = new RegExp(String.raw`^#\[(\d+)\]$`).exec(ref);
    return m ? Number(m[1]) : null;
}

// The unique-marker count while streaming (before the terminal frame).
const MARKER_RE = new RegExp(String.raw`#\[\d+\]`, "g");

/** How many distinct citations an answer carries. `chunk_refs` is
 * authoritative once the terminal frame has landed; before that the unique
 * markers in the streamed text are the best available count. */
export function citationCount(
    answer: string | null,
    chunkRefs: ChunkRefs | null,
): number {
    if (chunkRefs) return Object.keys(chunkRefs).length;
    if (!answer) return 0;
    return new Set(Array.from(answer.matchAll(MARKER_RE), (m) => m[0])).size;
}

export interface ResolvedChunk {
    text: string | null;
    page: number | null;
    plugin: string | null;
}

export interface CitationEntry {
    refKey: string;
    number: number;
    originSourceId: string;
    chunkId: string;
    filename: string | null;
    page: number | null;
    plugin: string | null;
    // null when the chunk text is unknown; the UI falls back to a
    // "cited chunk" label with a short id.
    excerpt: string | null;
}

// Builds the sources-list rows from a final answer's chunk_refs. `resolve`
// returns known chunk details (from the chunk cache); `filenameFor` maps an
// origin source id to the uploaded filename. Rows are sorted by marker number.
export function buildCitations(
    chunkRefs: ChunkRefs | null,
    resolve: (originSourceId: string, chunkId: string) => ResolvedChunk | null,
    filenameFor: (originSourceId: string) => string | null,
): CitationEntry[] {
    if (!chunkRefs) return [];
    return Object.entries(chunkRefs)
        .map(([refKey, value], position) => {
            const resolved = resolve(value.origin_source_id, value.chunk_id);
            return {
                refKey,
                // The number comes from the marker itself; the positional fallback
                // only guards a malformed ref key.
                number: citationIndex(refKey) ?? position + 1,
                originSourceId: value.origin_source_id,
                chunkId: value.chunk_id,
                filename: filenameFor(value.origin_source_id),
                page: resolved?.page ?? null,
                plugin: resolved?.plugin ?? null,
                excerpt: resolved?.text ?? null,
            };
        })
        .sort((a, b) => a.number - b.number);
}
