# Agentic RAG Dashboard

Web client for the RAG (FastAPI) backend. Sign in, ingest documents,
ask questions about them, and inspect the cited chunks behind each answer.

The backend runs separately; the default API base is `http://localhost:8000`.
Point `VITE_RAG_BASE_URL` at another instance if needed.

## Setup

```sh
npm install
```

## Commands

```sh
npm run dev        # dev server on http://localhost:5173
npm test           # Vitest unit and component tests (single run)
npm run test:watch # Vitest watch mode
npm run e2e        # Playwright e2e against the server
npm run build      # typecheck (vue-tsc) then production build
npm run preview    # serve the production build
```

The e2e suite starts the Vite dev server and the server itself when they
are not already running, and reuses them when they are.

## Testing

Unit tests (Vitest + jsdom):

- `src/api/sse.spec.ts` frame decoding, `src/api/client.spec.ts` client and stream wrapper behavior against mocked fetch.
- `src/lib/*.spec.ts` formatting, persistence, theme, citations, markdown (including the citation badge rule and XSS escaping).
- `src/store/*.spec.ts` store behavior with the API client fully mocked (every method, and the real `ApiError` class for instanceof checks).
- `src/components/*.spec.ts` key component behavior (composer, tool steps, assistant message, sources panel, button).

E2E (Playwright, real backend):

- `e2e/journey.spec.ts` covers the full user journey for a text document (register, ingest, ask, stream, citation click, chunk inspector, preview, file and chat deletion), a table document journey (chunk browser, citation), a mid-stream reload (draft re-offer), and a mid-ingest reload (progress recovery).
