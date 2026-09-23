// Server-sent events: incremental frame decoding plus a fetch-based stream
// wrapper. Frames are separated by blank lines, and network chunks can cut a
// frame anywhere, so partial data is buffered.

export interface RawSseFrame {
    event: string;
    // Multi-line data payloads are joined with newlines. Non-JSON payloads
    // stay raw strings; consumers cast them.
    data: string;
}

export interface SseFrame {
    event: string;
    data: unknown;
}

// Sentinel status for network-level failures, kept out of the 0-999 HTTP
// range so callers can tell "no server" from a real status code.
export const NETWORK_STATUS = 0;

export class SseDecoder {
    private pending = "";
    private lines: string[] = [];

    // Consume one network chunk; return every complete frame it contains.
    push(chunk: string): RawSseFrame[] {
        const frames: RawSseFrame[] = [];
        const text = this.pending + chunk;
        this.pending = "";
        let start = 0;
        let index = 0;
        while (index < text.length) {
            const char = text[index];
            if (char !== "\n" && char !== "\r") {
                index++;
                continue;
            }
            const line = text.slice(start, index);
            index += char === "\r" && text[index + 1] === "\n" ? 2 : 1;
            start = index;
            if (line === "") {
                if (this.lines.length > 0) frames.push(this.consume());
            } else {
                this.lines.push(line);
            }
        }
        this.pending = text.slice(start);
        return frames;
    }

    // Decode whatever remains in the buffer once the stream closes.
    flush(): RawSseFrame[] {
        if (this.pending !== "") {
            const line = this.pending;
            this.pending = "";
            if (line !== "") this.lines.push(line);
        }
        if (this.lines.length > 0) return [this.consume()];
        return [];
    }

    private consume(): RawSseFrame {
        const lines = this.lines;
        this.lines = [];
        let event = "";
        const dataLines: string[] = [];
        for (const raw of lines) {
            // Comments (lines starting with ':') are ignored.
            if (raw.startsWith(":")) continue;
            const colon = raw.indexOf(":");
            if (colon === -1) continue;
            const field = raw.slice(0, colon);
            let value = raw.slice(colon + 1);
            if (value.startsWith(" ")) value = value.slice(1);
            if (field === "data") dataLines.push(value);
            else if (field === "event") event = value;
            // `id`, `retry`, and unknown fields are ignored.
        }
        return { event: event || "message", data: dataLines.join("\n") };
    }
}

// JSON when possible; anything else stays a raw string for the consumer to
// cast.
export function parseSseData<T>(data: string): T {
    if (data === "") return data as T;
    try {
        return JSON.parse(data) as T;
    } catch {
        return data as T;
    }
}
