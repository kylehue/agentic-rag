import { describe, expect, it } from "vitest";
import { SseDecoder, parseSseData } from "./sse";

describe("SseDecoder", () => {
    it("decodes a single event with JSON data", () => {
        const decoder = new SseDecoder();
        const frames = decoder.push(
            'event: tool_call\ndata: {"name": "x", "arguments": {}}\n\n',
        );
        expect(frames).toEqual([
            { event: "tool_call", data: '{"name": "x", "arguments": {}}' },
        ]);
    });

    it("buffers partial data across chunk cuts", () => {
        const decoder = new SseDecoder();
        expect(decoder.push("event: a\nda")).toEqual([]);
        expect(decoder.push('ta: {"x":')).toEqual([]);
        const frames = decoder.push("1}\n\n");
        expect(frames).toEqual([{ event: "a", data: '{"x":1}' }]);
    });

    it("splits frames cut at any position", () => {
        const decoder = new SseDecoder();
        const stream =
            'event: one\ndata: {"n": 1}\n\nevent: two\ndata: {"n": 2}\n\n';
        const frames: unknown[] = [];
        for (let i = 0; i < stream.length; i++) {
            frames.push(...decoder.push(stream.slice(i, i + 1)));
        }
        expect(frames).toEqual([
            { event: "one", data: '{"n": 1}' },
            { event: "two", data: '{"n": 2}' },
        ]);
    });

    it("joins multi-line data payloads with newlines", () => {
        const decoder = new SseDecoder();
        const frames = decoder.push("event: x\ndata: line1\ndata: line2\n\n");
        expect(frames).toEqual([{ event: "x", data: "line1\nline2" }]);
    });

    it("ignores comments and metadata lines (id, retry)", () => {
        const decoder = new SseDecoder();
        const frames = decoder.push(
            ': keepalive\nid: 42\nretry: 3000\nevent: x\ndata: {"ok": true}\n\n',
        );
        expect(frames).toEqual([{ event: "x", data: '{"ok": true}' }]);
    });

    it("decodes empty frames as no events", () => {
        const decoder = new SseDecoder();
        expect(decoder.push("\n\n\n")).toEqual([]);
    });

    it("defaults the event name to message when omitted", () => {
        const decoder = new SseDecoder();
        const frames = decoder.push('data: {"a": 1}\n\n');
        expect(frames).toEqual([{ event: "message", data: '{"a": 1}' }]);
    });

    it("accepts CRLF line endings", () => {
        const decoder = new SseDecoder();
        const frames = decoder.push('event: x\r\ndata: {"y": 2}\r\n\r\n');
        expect(frames).toEqual([{ event: "x", data: '{"y": 2}' }]);
    });

    it("flushes the remainder of the buffer once the stream closes", () => {
        const decoder = new SseDecoder();
        expect(decoder.push('event: last\ndata: {"tail": true}\n')).toEqual([]);
        expect(decoder.flush()).toEqual([
            { event: "last", data: '{"tail": true}' },
        ]);
        expect(decoder.flush()).toEqual([]);
    });
});

describe("parseSseData", () => {
    it("parses JSON payloads", () => {
        expect(parseSseData('{"a": 1}')).toEqual({ a: 1 });
    });

    it("keeps non-JSON payloads as raw strings", () => {
        expect(parseSseData("plain text")).toBe("plain text");
    });
});
