const encoder = new TextEncoder();

export function fakeSseBody(chunks: string[]): ReadableStream<Uint8Array> {
    return new ReadableStream<Uint8Array>({
        start(controller) {
            for (const chunk of chunks) {
                controller.enqueue(encoder.encode(chunk));
            }
            controller.close();
        },
    });
}

export function fakeSseResponse(
    chunks: string[],
    init?: { status?: number; body?: string },
): Response {
    if (init?.body !== undefined) {
        return new Response(init.body, {
            status: init.status ?? 200,
            headers: { "Content-Type": "application/json" },
        });
    }
    return new Response(fakeSseBody(chunks), {
        status: init?.status ?? 200,
        headers: { "Content-Type": "text/event-stream" },
    });
}

export type FakeFrame = { event: string; data: unknown };

// Fake streams in tests are async generators yielding typed frames.
export function framesOf(
    frames: Array<[string, unknown]>,
): AsyncGenerator<FakeFrame> {
    return (async function* () {
        for (const [event, data] of frames) {
            yield { event, data };
        }
    })();
}

// Yields its frames, then waits until the signal aborts (simulating a live
// stream the user can stop).
export function framesUntilAbort(
    signal: AbortSignal,
    frames: Array<[string, unknown]>,
): AsyncGenerator<FakeFrame> {
    return (async function* () {
        for (const [event, data] of frames) {
            yield { event, data };
        }
        await new Promise<void>((resolve) => {
            if (signal.aborted) resolve();
            else
                signal.addEventListener("abort", () => resolve(), {
                    once: true,
                });
        });
    })();
}
