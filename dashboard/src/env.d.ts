/// <reference types="vite/client" />

interface ImportMetaEnv {
    /** Base URL of the rag-server API. Defaults to the local dev server. */
    readonly VITE_RAG_BASE_URL?: string;
}

interface ImportMeta {
    readonly env: ImportMetaEnv;
}
