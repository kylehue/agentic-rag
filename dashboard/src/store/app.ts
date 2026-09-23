import { computed, ref } from "vue";
import { defineStore } from "pinia";
import { ApiError, api } from "@/api/client";
import type {
    AnswerStreamPayload,
    ChatInfo,
    ChatMessage,
    ChunkRefs,
    IngestChatPayload,
    ListFilesResponse,
    StoredChunk,
    ToolCallPayload,
    ToolResultPayload,
} from "@/api/types";
import {
    buildCitations,
    type CitationEntry,
    type ResolvedChunk,
} from "@/lib/citations";
import {
    applyTheme,
    nextTheme,
    readThemePref,
    saveThemePref,
    type ThemePref,
} from "@/lib/theme";
import { safeGet, safeRemove, safeSet } from "@/lib/storage";
import { toast } from "@/components/ui/use-toast";

export const PERSISTED = {
    activeChat: "rag:active-chat",
    sourcesOpen: "rag:sources-open",
    draft: "rag:draft",
} as const;

export interface ToolStep {
    name: string;
    arguments: Record<string, unknown> | null;
    result: string | null;
}

export interface AssistantTurn {
    kind: "assistant";
    id: number;
    toolSteps: ToolStep[];
    // null until the terminal frame (or history load)
    answer: string | null;
    chunkRefs: ChunkRefs | null;
    citations: CitationEntry[] | null;
    streaming: boolean;
    failed: boolean;
    error: string | null;
}

export interface UserTurn {
    kind: "user";
    id: number;
    content: string;
}

export type Turn = UserTurn | AssistantTurn;

export interface IngestRow {
    filename: string;
    jobId: string;
    status: "queued" | "running" | "done" | "error";
    stage: "queued" | "processing" | "saving" | "embedding" | "done" | "failed";
    pluginState: string | null;
    chunkCount: number | null;
    error: string | null;
    progress: number;
}

export interface ChunkCacheEntry {
    chunks: StoredChunk[] | null;
    loading: boolean;
    loaded: boolean;
}

export interface ConfirmState {
    title: string;
    description: string;
    destructive: boolean;
}

let nextTurnId = 1;

function newId(): number {
    return nextTurnId++;
}

// Group heterogeneous history items into turns. Consecutive non-user items
// collapse into one assistant turn; each user item closes the group.
export function groupHistory(items: ChatMessage[]): Turn[] {
    const turns: Turn[] = [];
    let current: AssistantTurn | null = null;
    for (const item of items) {
        if (item.type === "user") {
            current = null;
            turns.push({ kind: "user", id: newId(), content: item.content });
            continue;
        }
        if (!current) {
            current = {
                kind: "assistant",
                id: newId(),
                toolSteps: [],
                answer: null,
                chunkRefs: null,
                citations: null,
                streaming: false,
                failed: false,
                error: null,
            };
            turns.push(current);
        }
        if (item.type === "tool_call") {
            current.toolSteps.push({
                name: item.name,
                arguments: item.arguments ?? null,
                result: null,
            });
        } else if (item.type === "tool_result") {
            const step = [...current.toolSteps]
                .reverse()
                .find((s) => s.result === null);
            if (step) step.result = item.content;
            else
                current.toolSteps.push({
                    name: item.name,
                    arguments: null,
                    result: item.content,
                });
        } else if (item.type === "answer") {
            current.answer = item.answer;
            current.chunkRefs = item.chunk_refs ?? null;
            current.citations = buildCitations(
                current.chunkRefs,
                () => null,
                () => null,
            );
        }
    }
    return turns;
}

const STAGE_PROGRESS: Record<string, number> = {
    queued: 5,
    processing: 40,
    saving: 70,
    embedding: 85,
    done: 100,
};

export const useAppStore = defineStore("app", () => {
    // --- auth ---
    const authStatus = ref<"loading" | "guest" | "authed">("loading");
    const username = ref("");

    // --- navigation / theme ---
    const mobileMenuOpen = ref(false);
    const sourcesOpen = ref(safeGet(PERSISTED.sourcesOpen) === "1");
    const theme = ref<ThemePref>(readThemePref());

    // --- chats ---
    const chats = ref<ChatInfo[]>([]);
    const chatsLoading = ref(false);
    const activeChatId = ref<string | null>(null);
    const messages = ref<Turn[]>([]);
    const historyLoading = ref(false);
    const activeChatFiles = ref<ListFilesResponse["files"]>([]);
    const filesLoading = ref(false);

    // --- streaming ask ---
    const streaming = ref(false);
    const stopping = ref(false);
    let answerController: AbortController | null = null;

    // recovered in-flight question, pre-filled into the composer
    const recoveredDraft = ref<string | null>(null);

    // --- chunk cache (per origin source id) ---
    const chunkCache = ref<Record<string, ChunkCacheEntry>>({});

    // --- transient ingest progress (per filename) ---
    const ingests = ref<Record<string, IngestRow>>({});
    const jobStreams = new Map<string, AbortController>();
    let filesRefreshQueued = false;

    // --- modals (app level) ---
    const citedChunk = ref<{
        originSourceId: string;
        chunkId: string;
    } | null>(null);
    const fileChunks = ref<{
        originSourceId: string;
        filename: string;
    } | null>(null);
    const filePreview = ref<{ sourceId: string } | null>(null);
    const confirm = ref<ConfirmState | null>(null);
    let confirmAction: (() => void | Promise<void>) | null = null;

    // --- derived ---
    const activeChat = computed(
        () => chats.value.find((c) => c.chat_id === activeChatId.value) ?? null,
    );
    const durableFileCount = computed(() => activeChatFiles.value.length);
    const inFlightFileCount = computed(() => Object.keys(ingests.value).length);
    const totalFileCount = computed(
        () => durableFileCount.value + inFlightFileCount.value,
    );
    const isStreaming = computed(() => streaming.value);

    // Monotonic scroll counter: MessageList force-scrolls on each bump
    // (new question posted, history finished loading).
    const scrollForce = ref(0);

    // --- helpers ---

    function setSourcesOpen(open: boolean): void {
        sourcesOpen.value = open;
        safeSet(PERSISTED.sourcesOpen, open ? "1" : "0");
    }

    function setTheme(pref: ThemePref): void {
        theme.value = pref;
        saveThemePref(pref);
        applyTheme(pref);
    }

    function toggleTheme(): void {
        setTheme(nextTheme(theme.value));
    }

    function persistActiveChat(chatId: string | null): void {
        if (chatId) safeSet(PERSISTED.activeChat, chatId);
        else safeRemove(PERSISTED.activeChat);
    }

    // The server may mint a chat during a stream (first frame) or on ingest.
    function adoptChatId(chatId: string): void {
        if (chatId !== activeChatId.value) {
            activeChatId.value = chatId;
            persistActiveChat(chatId);
        }
        if (!chats.value.some((c) => c.chat_id === chatId)) {
            chats.value.push({
                chat_id: chatId,
                created_at: Date.now() / 1000,
                sources: [],
            });
        }
        // The draft was recorded before the chat id existed: update it so
        // recovery can check the real chat's history.
        const raw = safeGet(PERSISTED.draft);
        if (raw) {
            try {
                const draft = JSON.parse(raw) as {
                    question: string;
                    chatId: string | null;
                };
                if (draft && !draft.chatId && draft.question) {
                    safeSet(
                        PERSISTED.draft,
                        JSON.stringify({ question: draft.question, chatId }),
                    );
                }
            } catch {
                // unreadable draft; leave it
            }
        }
    }

    function filenameFor(sourceId: string): string | null {
        return (
            activeChatFiles.value.find((f) => f.source_id === sourceId)
                ?.filename ?? null
        );
    }

    function resolveChunk(
        originSourceId: string,
        chunkId: string,
    ): ResolvedChunk | null {
        const entry = chunkCache.value[originSourceId];
        const chunk = entry?.chunks?.find((c) => c.chunk_id === chunkId);
        if (!chunk) return null;
        const page = chunk.metadata?.source_page_number;
        return {
            text: chunk.text,
            page: typeof page === "number" ? page : null,
            plugin: chunk.plugin,
        };
    }

    function refreshCitations(turn: AssistantTurn): void {
        if (turn.chunkRefs === null) {
            turn.citations = null;
            return;
        }
        turn.citations = buildCitations(
            turn.chunkRefs,
            (originSourceId, chunkId) => resolveChunk(originSourceId, chunkId),
            (originSourceId) => filenameFor(originSourceId),
        );
    }

    async function ensureChunks(
        originSourceId: string,
        chatId: string | null,
    ): Promise<StoredChunk[] | null> {
        const entry = chunkCache.value[originSourceId];
        if (entry?.loaded) return entry.chunks;
        if (entry?.loading) return null;
        chunkCache.value[originSourceId] = {
            chunks: null,
            loading: true,
            loaded: false,
        };
        try {
            const result = await api.listFileChunks(originSourceId, chatId);
            chunkCache.value[originSourceId] = {
                chunks: result.chunks,
                loading: false,
                loaded: true,
            };
            return result.chunks;
        } catch (error) {
            chunkCache.value[originSourceId] = {
                chunks: null,
                loading: false,
                loaded: false,
            };
            if (!isAbortLike(error)) surfaceError(error);
            return null;
        }
    }

    // After the terminal frame, fetch chunk details so citations show real
    // text, pages, and plugins (answer-carried details win when present).
    async function resolveCitationsAsync(turn: AssistantTurn): Promise<void> {
        if (turn.chunkRefs === null) {
            refreshCitations(turn);
            return;
        }
        const origins = [
            ...new Set(
                Object.values(turn.chunkRefs).map((r) => r.origin_source_id),
            ),
        ];
        for (const origin of origins) {
            await ensureChunks(origin, activeChatId.value);
        }
        // Rebuild the rows now that the chunk cache holds the details.
        refreshCitations(turn);
    }

    function isAbortLike(error: unknown): boolean {
        return error instanceof Error && error.name === "AbortError";
    }

    function surfaceError(error: unknown): void {
        const detail =
            error instanceof ApiError
                ? error.message
                : error instanceof Error
                  ? error.message
                  : "Something went wrong";
        toast({ title: "Error", description: detail, variant: "destructive" });
    }

    // --- auth ---

    async function init(): Promise<void> {
        setTheme(theme.value);
        try {
            const me = await api.me();
            username.value = me.username;
            // The boot is atomic: the session restore (history, files, ingest
            // recovery, draft recovery) completes before the layout mounts, so the
            // composer sees a recovered draft when it initializes.
            await restoreSession();
            authStatus.value = "authed";
        } catch (error) {
            if (error instanceof ApiError && error.status === 401) {
                authStatus.value = "guest";
                return;
            }
            // network failure or other error: stay signed out rather than spin
            authStatus.value = "guest";
            surfaceError(error);
        }
    }

    async function restoreSession(): Promise<void> {
        try {
            chatsLoading.value = true;
            chats.value = await api.listChats();
        } catch (error) {
            surfaceError(error);
        } finally {
            chatsLoading.value = false;
        }

        const persisted = safeGet(PERSISTED.activeChat);
        const stillExists =
            persisted !== null &&
            chats.value.some((c) => c.chat_id === persisted);
        if (stillExists && persisted) {
            activeChatId.value = persisted;
        } else {
            safeRemove(PERSISTED.activeChat);
        }

        if (activeChatId.value) {
            await loadChatContext(activeChatId.value);
        } else {
            activeChatFiles.value = [];
            messages.value = [];
        }
        await recoverDraft();
    }

    function errorDetail(error: unknown): string {
        if (error instanceof ApiError) return error.message;
        if (error instanceof Error) return error.message;
        return "Something went wrong";
    }

    // Returns null on success, or the error detail for inline display.
    async function login(
        loginUsername: string,
        password: string,
    ): Promise<string | null> {
        try {
            const me = await api.login(loginUsername, password);
            await onAuthed(me.username);
            return null;
        } catch (error) {
            return errorDetail(error);
        }
    }

    // Registration does not establish a session; login follows it.
    async function register(
        registerUsername: string,
        password: string,
    ): Promise<string | null> {
        try {
            await api.register(registerUsername, password);
            const me = await api.login(registerUsername, password);
            await onAuthed(me.username);
            return null;
        } catch (error) {
            return errorDetail(error);
        }
    }

    async function onAuthed(name: string): Promise<void> {
        username.value = name;
        // Same atomicity as the boot: restore before flipping to authenticated.
        await restoreSession();
        authStatus.value = "authed";
    }

    async function logout(): Promise<void> {
        try {
            await api.logout();
        } catch {
            // best effort; clear local state regardless
        }
        clearAllLocalState();
    }

    function clearAllLocalState(): void {
        abortAnswerStream();
        abortJobStreams();
        authStatus.value = "guest";
        username.value = "";
        chats.value = [];
        activeChatId.value = null;
        persistActiveChat(null);
        messages.value = [];
        activeChatFiles.value = [];
        ingestsByDelete();
        clearChunkCache();
        recoveredDraft.value = null;
        safeRemove(PERSISTED.draft);
        mobileMenuOpen.value = false;
        setSourcesOpen(false);
        citedChunk.value = null;
        fileChunks.value = null;
        filePreview.value = null;
        confirm.value = null;
        confirmAction = null;
    }

    function ingestsByDelete(): void {
        for (const key of Object.keys(ingests.value)) delete ingests.value[key];
    }

    function clearChunkCache(): void {
        for (const key of Object.keys(chunkCache.value))
            delete chunkCache.value[key];
    }

    // --- chat management ---

    async function refreshChats(): Promise<void> {
        try {
            chats.value = await api.listChats();
            if (
                activeChatId.value &&
                !chats.value.some((c) => c.chat_id === activeChatId.value)
            ) {
                // the active chat was deleted (e.g. by another surface)
                activeChatId.value = null;
                persistActiveChat(null);
                messages.value = [];
                activeChatFiles.value = [];
            }
        } catch (error) {
            surfaceError(error);
        }
    }

    function newChat(): void {
        if (streaming.value) return;
        activeChatId.value = null;
        persistActiveChat(null);
        messages.value = [];
        activeChatFiles.value = [];
        abortJobStreams();
        ingestsByDelete();
        clearChunkCache();
        mobileMenuOpen.value = false;
    }

    async function selectChat(chatId: string): Promise<void> {
        if (chatId === activeChatId.value) return;
        // Switching chats mid-stream drops the partial assistant message
        // (intended). The navigation abort must not mark it failed.
        abortAnswerStream();
        abortJobStreams();
        ingestsByDelete();
        clearChunkCache();
        activeChatId.value = chatId;
        persistActiveChat(chatId);
        mobileMenuOpen.value = false;
        await loadChatContext(chatId);
    }

    async function loadChatContext(chatId: string): Promise<void> {
        messages.value = [];
        activeChatFiles.value = [];
        recoveredDraft.value = null;
        historyLoading.value = true;
        filesLoading.value = true;
        try {
            const [history, files] = await Promise.all([
                api.chatMessages(chatId),
                api.listFiles(chatId),
            ]);
            if (activeChatId.value !== chatId) return;
            messages.value = groupHistory(history);
            activeChatFiles.value = files.files;
        } catch (error) {
            if (activeChatId.value === chatId) {
                messages.value = [];
                surfaceError(error);
            }
        } finally {
            if (activeChatId.value === chatId) {
                historyLoading.value = false;
                filesLoading.value = false;
                scrollForce.value++;
            }
        }
        // Citation details (chunk text, page, plugin) resolve in the background:
        // a slow chunks fetch must not keep history and files in loading state.
        void resolveHistoryCitations(chatId);
        await recoverIngests();
    }

    async function resolveHistoryCitations(chatId: string): Promise<void> {
        const turns = messages.value;
        for (const turn of turns) {
            if (activeChatId.value !== chatId) return;
            if (turn.kind === "assistant" && turn.chunkRefs) {
                await resolveCitationsAsync(turn);
            }
        }
    }

    function requestConfirm(options: {
        title: string;
        description: string;
        destructive?: boolean;
        onConfirm: () => void | Promise<void>;
    }): void {
        confirm.value = {
            title: options.title,
            description: options.description,
            destructive: options.destructive ?? false,
        };
        confirmAction = options.onConfirm;
    }

    function cancelConfirm(): void {
        confirm.value = null;
        confirmAction = null;
    }

    // The close event clears the pending item; the confirm handler clears it
    // itself before acting.
    async function executeConfirm(): Promise<void> {
        const action = confirmAction;
        confirm.value = null;
        confirmAction = null;
        if (action) await action();
    }

    async function deleteChat(chatId: string): Promise<void> {
        try {
            await api.deleteChat(chatId);
        } catch (error) {
            surfaceError(error);
            return;
        }
        chats.value = chats.value.filter((c) => c.chat_id !== chatId);
        if (activeChatId.value === chatId) {
            const first = chats.value[0];
            if (first) {
                activeChatId.value = first.chat_id;
                persistActiveChat(first.chat_id);
                await loadChatContext(first.chat_id);
            } else {
                activeChatId.value = null;
                persistActiveChat(null);
                messages.value = [];
                activeChatFiles.value = [];
            }
        }
    }

    // --- ask flow ---

    function saveDraft(question: string, chatId: string | null): void {
        safeSet(PERSISTED.draft, JSON.stringify({ question, chatId }));
    }

    function clearDraft(): void {
        safeRemove(PERSISTED.draft);
    }

    async function recoverDraft(): Promise<void> {
        const raw = safeGet(PERSISTED.draft);
        if (!raw) return;
        let draft: { question: string; chatId: string | null } | null = null;
        try {
            draft = JSON.parse(raw) as {
                question: string;
                chatId: string | null;
            };
            if (!draft || typeof draft.question !== "string" || !draft.question)
                draft = null;
        } catch {
            draft = null;
        }
        if (!draft) {
            safeRemove(PERSISTED.draft);
            return;
        }
        // The draft is only offered while the question is still unanswered on
        // the server (see below). A null chat id (recorded before the server
        // minted the chat) falls back to the newest chat, which is where that
        // question lives.
        let target = draft.chatId;
        if (!target) {
            const newest = [...chats.value].sort(
                (a, b) => b.created_at - a.created_at,
            )[0];
            target = newest?.chat_id ?? null;
        }
        if (!target) {
            safeRemove(PERSISTED.draft);
            return;
        }
        try {
            const history = await api.chatMessages(target);
            // The run is unrecovered only if the question's user turn exists and
            // no answer item follows it. (Checkpoints land after every node, so a
            // mid-run history can end in a tool_call/tool_result, not the user
            // turn.)
            let questionIndex = -1;
            for (let i = history.length - 1; i >= 0; i--) {
                const item = history[i];
                if (item.type === "user" && item.content === draft.question) {
                    questionIndex = i;
                    break;
                }
            }
            const answered =
                questionIndex !== -1 &&
                history
                    .slice(questionIndex + 1)
                    .some((m) => m.type === "answer");
            if (questionIndex !== -1 && !answered) {
                safeSet(PERSISTED.activeChat, target);
                if (activeChatId.value !== target) {
                    activeChatId.value = target;
                    await loadChatContext(target);
                }
                // Set after the context load: switching chats clears the recovered
                // draft, so the order matters.
                recoveredDraft.value = draft.question;
                toast({
                    title: "Recovered question",
                    description:
                        "The page reloaded before this answer finished. Ask it again to continue.",
                });
            } else {
                clearDraft();
            }
        } catch {
            // cannot verify; keep the draft for the next load
        }
    }

    function consumeRecoveredDraft(): void {
        recoveredDraft.value = null;
    }

    async function ask(question: string): Promise<void> {
        const text = question.trim();
        if (!text || streaming.value) return;

        // The stream has no replay: persist the in-flight question before
        // starting so a reload can re-offer it.
        saveDraft(text, activeChatId.value);

        // Raw conversation log (browser console): the question, then each
        // discrete frame exactly as the server sent it, no shaping. Token
        // deltas are skipped; the terminal answer frame carries the full
        // raw answer.
        console.log("[conversation] user", text);

        const userTurn: UserTurn = { kind: "user", id: newId(), content: text };
        const assistantTurn: AssistantTurn = {
            kind: "assistant",
            id: newId(),
            toolSteps: [],
            answer: null,
            chunkRefs: null,
            citations: null,
            streaming: true,
            failed: false,
            error: null,
        };
        messages.value.push(userTurn, assistantTurn);
        scrollForce.value++;
        streaming.value = true;
        answerController = new AbortController();
        const controller = answerController;
        // Read the turn back as its reactive proxy so identity checks and
        // mutations go through Vue (pushing a raw object yields a proxy on read).
        const live = messages.value[messages.value.length - 1] as AssistantTurn;

        try {
            for await (const frame of api.answerStream(
                text,
                activeChatId.value,
                controller.signal,
            )) {
                // A chat switch dropped the partial message; stop feeding it.
                if (messages.value[messages.value.length - 1] !== live) break;
                if (frame.event !== "answer_delta") {
                    console.log(`[conversation] ${frame.event}`, frame.data);
                }
                switch (frame.event) {
                    case "chat": {
                        const data = frame.data as IngestChatPayload;
                        if (data?.chat_id) adoptChatId(data.chat_id);
                        break;
                    }
                    case "answer_delta": {
                        const data = frame.data as { content: string };
                        live.answer =
                            (live.answer ?? "") + (data.content ?? "");
                        refreshCitations(live);
                        break;
                    }
                    case "tool_call": {
                        const data = frame.data as ToolCallPayload;
                        live.toolSteps.push({
                            name: data.name,
                            arguments: data.arguments ?? null,
                            result: null,
                        });
                        break;
                    }
                    case "tool_result": {
                        const data = frame.data as ToolResultPayload;
                        const step = [...live.toolSteps]
                            .reverse()
                            .find((s) => s.result === null);
                        if (step) step.result = data.content;
                        else
                            live.toolSteps.push({
                                name: data.name,
                                arguments: null,
                                result: data.content,
                            });
                        break;
                    }
                    case "answer": {
                        const data = frame.data as AnswerStreamPayload;
                        // The terminal frame overrides whatever was streamed.
                        live.answer = data.answer;
                        live.chunkRefs = data.chunk_refs ?? null;
                        if (data.chat_id) adoptChatId(data.chat_id);
                        await resolveCitationsAsync(live);
                        break;
                    }
                    default:
                        break;
                }
            }
            live.streaming = false;
            clearDraft();
            await refreshChats();
        } catch (error) {
            live.streaming = false;
            if (controller.signal.aborted) {
                // A user stop or chat switch: not a failure.
                clearDraft();
            } else {
                live.failed = true;
                live.error =
                    error instanceof ApiError
                        ? error.message
                        : "Something went wrong";
                surfaceError(error);
            }
        } finally {
            streaming.value = false;
            stopping.value = false;
            answerController = null;
        }
    }

    function abortAnswerStream(): void {
        if (answerController) {
            answerController.abort();
            answerController = null;
        }
    }

    async function stop(): Promise<void> {
        if (!streaming.value || stopping.value) return;
        const chatId = activeChatId.value;
        if (!chatId) return;
        stopping.value = true;
        try {
            await api.stopAnswer(chatId);
        } catch {
            // the run may have finished between the click and the call
        }
        abortAnswerStream();
    }

    // --- sources / ingest ---

    async function upload(files: File[]): Promise<void> {
        if (files.length === 0) return;
        try {
            const response = await api.uploadFiles(files, activeChatId.value);
            const previousChat = activeChatId.value;
            if (response.chat_id) adoptChatId(response.chat_id);
            if (previousChat !== activeChatId.value) {
                // a new chat was minted: load its durable context
                activeChatFiles.value = [];
                const loaded = await api.listFiles(activeChatId.value);
                activeChatFiles.value = loaded.files;
            }
            for (const job of response.jobs) {
                ingests.value[job.file] = {
                    filename: job.file,
                    jobId: job.job_id,
                    status: "queued",
                    stage: "queued",
                    pluginState: null,
                    chunkCount: null,
                    error: null,
                    progress: STAGE_PROGRESS.queued,
                };
                // Attach every job's stream in parallel: awaiting them serially
                // would hold each later file's row back until the previous job
                // finished.
                void attachJobStream(job.job_id, job.file);
            }
            await refreshChats();
        } catch (error) {
            surfaceError(error);
        }
    }

    async function attachJobStream(
        jobId: string,
        filename: string,
        fromReconcile = false,
    ): Promise<void> {
        const controller = new AbortController();
        jobStreams.set(jobId, controller);
        try {
            for await (const frame of api.ingestStream(
                jobId,
                controller.signal,
            )) {
                // The row is keyed by the job's file. Frames may name emitted
                // sub-files (e.g. extracted tables), but only the job's file ever
                // receives the terminal frame, so keying rows by frame file would
                // leave sub-file rows stuck forever.
                applyIngestEvent(
                    jobId,
                    filename,
                    frame.event,
                    frame.data as Record<string, unknown>,
                );
            }
        } catch (error) {
            // A navigation abort must not settle state; only surface real errors.
            if (!isAbortLike(error) && !controller.signal.aborted) {
                surfaceError(error);
            }
        } finally {
            jobStreams.delete(jobId);
            // If the stream ended or dropped while the job's row is still
            // unsettled, reconcile against the jobs snapshot (authoritative),
            // re-attaching when the job is still running. One level deep to
            // prevent a re-attach loop.
            if (!controller.signal.aborted && !fromReconcile) {
                void reconcileJob(jobId, filename);
            }
        }
    }

    async function reconcileJob(
        jobId: string,
        filename: string,
    ): Promise<void> {
        if (!ingests.value[filename]) return;
        const chatId = activeChatId.value;
        if (!chatId) return;
        let response: Awaited<ReturnType<typeof api.ingestJobs>>;
        try {
            response = await api.ingestJobs(chatId);
        } catch {
            // The row settles on the next recovery instead.
            return;
        }
        if (activeChatId.value !== chatId) return;
        const job = response.jobs.find((j) => j.job_id === jobId);
        if (!job) return;
        for (const event of job.events) {
            applyIngestEvent(job.job_id, job.file, event.name, event.payload);
        }
        if (
            (job.status === "queued" || job.status === "running") &&
            ingests.value[filename]
        ) {
            void attachJobStream(job.job_id, job.file, true);
        }
    }

    function abortJobStreams(): void {
        for (const controller of jobStreams.values()) controller.abort();
        jobStreams.clear();
    }

    function applyIngestEvent(
        jobId: string,
        filename: string,
        event: string,
        payload: Record<string, unknown> | unknown,
    ): void {
        const data = (payload ?? {}) as Record<string, unknown>;
        switch (event) {
            case "chat": {
                const chatId = data.chat_id;
                if (typeof chatId === "string" && chatId) adoptChatId(chatId);
                break;
            }
            case "queued": {
                upsertIngestRow(jobId, filename, {
                    status: "queued",
                    stage: "queued",
                    progress: STAGE_PROGRESS.queued,
                });
                break;
            }
            case "started": {
                upsertIngestRow(jobId, filename, {
                    status: "running",
                    stage: "processing",
                    progress: STAGE_PROGRESS.queued + 5,
                });
                break;
            }
            case "stage": {
                const stage = data.stage as string | undefined;
                if (!stage) break;
                const row = upsertIngestRow(jobId, filename, {
                    status: "running",
                    stage:
                        stage === "processing" ||
                        stage === "saving" ||
                        stage === "embedding"
                            ? (stage as IngestRow["stage"])
                            : "processing",
                    progress:
                        STAGE_PROGRESS[stage] ?? STAGE_PROGRESS.processing,
                });
                if (typeof data.chunk_count === "number")
                    row.chunkCount = data.chunk_count;
                break;
            }
            case "plugin_state": {
                const row = upsertIngestRow(jobId, filename, {});
                const plugin =
                    typeof data.plugin === "string" ? data.plugin : null;
                const state = typeof data.state === "string" ? data.state : "";
                // Emitted sub-files do their work under the parent job; name them
                // so the row still says what is happening.
                const subFile =
                    typeof data.file === "string" && data.file !== filename
                        ? ` (${data.file})`
                        : "";
                row.pluginState = state
                    ? `${plugin ? `${plugin}${subFile}: ` : ""}${state}`
                    : null;
                break;
            }
            case "file_done": {
                const row = upsertIngestRow(jobId, filename, {
                    status: "done",
                    stage: "done",
                    progress: 95,
                });
                if (typeof data.chunk_count === "number")
                    row.chunkCount = data.chunk_count;
                break;
            }
            case "done": {
                const row = ingests.value[filename];
                if (row) row.status = "done";
                delete ingests.value[filename];
                void promoteFile();
                break;
            }
            case "error": {
                const row = upsertIngestRow(jobId, filename, {
                    status: "error",
                    stage: "failed",
                    progress: 100,
                });
                row.error =
                    typeof data.error === "string"
                        ? data.error
                        : "Ingest failed";
                break;
            }
            default:
                break;
        }
    }

    function upsertIngestRow(
        jobId: string,
        filename: string,
        patch: Partial<IngestRow>,
    ): IngestRow {
        const existing = ingests.value[filename];
        if (existing) {
            Object.assign(existing, patch);
            return existing;
        }
        const row: IngestRow = {
            filename,
            jobId,
            status: "queued",
            stage: "queued",
            pluginState: null,
            chunkCount: null,
            error: null,
            progress: 0,
            ...patch,
        };
        ingests.value[filename] = row;
        return row;
    }

    // Completion promotes the file into the durable list; re-fetch the chat's
    // files so the two lists never show the same filename at once.
    async function promoteFile(): Promise<void> {
        if (filesRefreshQueued) return;
        const chatId = activeChatId.value;
        if (!chatId) return;
        filesRefreshQueued = true;
        try {
            const files = await api.listFiles(chatId);
            if (activeChatId.value === chatId)
                activeChatFiles.value = files.files;
        } catch (error) {
            surfaceError(error);
        } finally {
            filesRefreshQueued = false;
        }
    }

    // Recovery after reload: read the snapshot, replay buffered events, then
    // re-attach to every still-running job's live stream.
    async function recoverIngests(): Promise<void> {
        const chatId = activeChatId.value;
        if (!chatId) return;
        let response: {
            chat_id: string;
            jobs: {
                job_id: string;
                status: string;
                file: string;
                created_at: number;
                events: { name: string; payload: Record<string, unknown> }[];
            }[];
        };
        try {
            response = await api.ingestJobs(chatId);
        } catch (error) {
            surfaceError(error);
            return;
        }
        if (activeChatId.value !== chatId) return;
        for (const job of response.jobs) {
            for (const event of job.events) {
                applyIngestEvent(
                    job.job_id,
                    job.file,
                    event.name,
                    event.payload,
                );
            }
            if (job.status === "queued" || job.status === "running") {
                void attachJobStream(job.job_id, job.file);
            }
        }
    }

    // --- files ---

    async function deleteFile(sourceId: string): Promise<void> {
        const chatId = activeChatId.value;
        try {
            await api.deleteFile(sourceId, chatId);
        } catch (error) {
            surfaceError(error);
            return;
        }
        activeChatFiles.value = activeChatFiles.value.filter(
            (f) => f.source_id !== sourceId,
        );
        delete chunkCache.value[sourceId];
        await refreshChats();
    }

    // --- modals ---

    async function openCitedChunk(
        originSourceId: string,
        chunkId: string,
    ): Promise<void> {
        citedChunk.value = { originSourceId, chunkId };
        await ensureChunks(originSourceId, activeChatId.value);
    }

    // The badge number addresses the citations list of the turn it renders in.
    async function openCitation(
        turn: AssistantTurn,
        number: number,
    ): Promise<void> {
        const entry = turn.citations?.find((c) => c.number === number);
        if (entry) await openCitedChunk(entry.originSourceId, entry.chunkId);
    }

    function openFileChunks(sourceId: string): void {
        fileChunks.value = {
            originSourceId: sourceId,
            filename: filenameFor(sourceId) ?? "",
        };
        void ensureChunks(sourceId, activeChatId.value);
    }

    function openFilePreview(sourceId: string): void {
        filePreview.value = { sourceId };
    }

    function closeFilePreview(): void {
        filePreview.value = null;
    }

    return {
        // state
        authStatus,
        username,
        mobileMenuOpen,
        sourcesOpen,
        theme,
        chats,
        chatsLoading,
        activeChatId,
        messages,
        historyLoading,
        activeChatFiles,
        filesLoading,
        streaming,
        stopping,
        recoveredDraft,
        ingests,
        chunkCache,
        citedChunk,
        fileChunks,
        filePreview,
        confirm,
        // derived
        activeChat,
        durableFileCount,
        inFlightFileCount,
        totalFileCount,
        isStreaming,
        scrollForce,
        // actions
        init,
        login,
        register,
        logout,
        setSourcesOpen,
        setTheme,
        toggleTheme,
        refreshChats,
        newChat,
        selectChat,
        loadChatContext,
        requestConfirm,
        cancelConfirm,
        executeConfirm,
        deleteChat,
        ask,
        stop,
        recoverDraft,
        consumeRecoveredDraft,
        upload,
        recoverIngests,
        deleteFile,
        ensureChunks,
        openCitedChunk,
        openCitation,
        openFileChunks,
        openFilePreview,
        closeFilePreview,
    };
});
