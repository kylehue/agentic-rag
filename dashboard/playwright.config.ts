import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, devices } from "@playwright/test";

// E2E runs against the server and the Vite dev server.
// Both reuse an already-running instance when present.
const here = path.dirname(fileURLToPath(import.meta.url));
const backendDir = path.join(here, "..");
const python = path.join(backendDir, "venv", "bin", "python");

export default defineConfig({
    testDir: "./e2e",
    timeout: 300_000,
    expect: { timeout: 30_000 },
    fullyParallel: false,
    workers: 1,
    reporter: [["list"]],
    use: {
        baseURL: "http://localhost:5173",
        trace: "retain-on-failure",
        ...devices["desktop Chrome"],
    },
    webServer: [
        {
            command: "npm run dev",
            url: "http://localhost:5173",
            reuseExistingServer: true,
            timeout: 90_000,
        },
        {
            command: `${python} -m uvicorn app.main:app --host 127.0.0.1 --port 8000`,
            cwd: backendDir,
            url: "http://127.0.0.1:8000/health/",
            reuseExistingServer: true,
            timeout: 90_000,
        },
    ],
});
