import { vi, type Mock } from "vitest";

// Tests mock the whole client object (every method plus the error class
// used in instanceof checks). Adding an endpoint means updating this list.
export interface MockApi {
    register: Mock;
    login: Mock;
    logout: Mock;
    me: Mock;
    listChats: Mock;
    createChat: Mock;
    chatMessages: Mock;
    deleteChat: Mock;
    uploadFiles: Mock;
    ingestJobs: Mock;
    ingestStream: Mock;
    listFiles: Mock;
    getFile: Mock;
    listFileChunks: Mock;
    deleteFile: Mock;
    fileLink: Mock;
    fileBlob: Mock;
    answerStream: Mock;
    stopAnswer: Mock;
}

export function createMockApi(): MockApi {
    return {
        register: vi.fn(),
        login: vi.fn(),
        logout: vi.fn(),
        me: vi.fn(),
        listChats: vi.fn(),
        createChat: vi.fn(),
        chatMessages: vi.fn(),
        deleteChat: vi.fn(),
        uploadFiles: vi.fn(),
        ingestJobs: vi.fn(),
        ingestStream: vi.fn(),
        listFiles: vi.fn(),
        getFile: vi.fn(),
        listFileChunks: vi.fn(),
        deleteFile: vi.fn(),
        fileLink: vi.fn(),
        fileBlob: vi.fn(),
        answerStream: vi.fn(),
        stopAnswer: vi.fn(),
    };
}

// Specs mock the client at the top level of the file. The factory body runs
// lazily (after module load), so it may reference this imported helper.
export async function mockClientFactory(): Promise<Record<string, unknown>> {
    const actual =
        await vi.importActual<typeof import("@/api/client")>("@/api/client");
    return {
        ...actual,
        api: createMockApi(),
    };
}
