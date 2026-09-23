import { expect, test } from "@playwright/test";
import { randomUUID } from "node:crypto";

const EXAMPLE_DOC = "./example_docs/doc.pdf";
const SCANNED_DOC = "./example_docs/scanned_doc.pdf";
const TABLE_DOC = "./example_docs/customers.csv";

async function signUp(page: import("@playwright/test").Page): Promise<string> {
    const username = `e2e_${randomUUID().slice(0, 8)}`;
    await page.goto("/");
    await page.getByTestId("auth-toggle").click();
    await page.getByLabel("Username").fill(username);
    await page.getByLabel("Password").fill("e2e-pass-1");
    await page.getByTestId("auth-submit").click();
    await expect(page.getByTestId("sidebar")).toBeVisible();
    return username;
}

test("full journey: register, ingest, ask, cite, delete file, delete chat", async ({
    page,
}) => {
    await signUp(page);

    // Upload through the sources panel and watch the background ingest finish.
    await page.getByTestId("open-sources").click();
    await expect(page.getByTestId("sources-panel")).toBeVisible();
    await page.getByTestId("upload-input").setInputFiles(EXAMPLE_DOC);
    await expect(
        page.getByTestId("ingest-row").filter({ hasText: "doc.pdf" }),
    ).toBeVisible();
    // The hosted partitioning API is slow for PDFs; give it room.
    await expect(page.getByTestId("file-row")).toContainText("doc.pdf", {
        timeout: 180_000,
    });
    // Close the sources panel so the chat is fully interactive.
    await page.keyboard.press("Escape");
    await expect(page.getByTestId("sources-panel")).toBeHidden();

    // Ask a question and watch the answer stream in. The model occasionally
    // cites nothing (or an index it was never shown); the server then records
    // no chunk_refs, so retry the question in that case.
    let answer: import("@playwright/test").Locator;
    for (let attempt = 0; attempt < 4; attempt++) {
        await page
            .getByTestId("composer-input")
            .fill("When did the Riverside Community Garden begin?");
        await page.getByTestId("composer-input").press("Enter");
        answer = page
            .getByTestId("assistant-message")
            .last()
            .getByTestId("markdown-body");
        await expect(answer).toContainText(/April 2026/i, { timeout: 180_000 });
        // The terminal frame resolves citations; the sources block is the
        // settled signal (the streaming caret disappears after the first delta).
        const cited = await page
            .getByTestId("sources-block")
            .last()
            .waitFor({ state: "visible", timeout: 180_000 })
            .then(() => true)
            .catch(() => false);
        if (cited) break;
    }
    await expect(page.getByTestId("sources-block").last()).toBeVisible();

    // The answer cites the document: open the chunk inspector from the
    // sources block (a single-citation answer renders no inline badge).
    await page.getByTestId("source-item").first().click();
    await expect(page.getByTestId("cited-chunk-dialog")).toBeVisible();
    await expect(page.getByTestId("cited-chunk-filename")).toContainText(
        "doc.pdf",
    );
    await expect(page.getByTestId("cited-chunk-text")).not.toBeEmpty();
    await page.getByTestId("cited-chunk-view-file").click();
    await expect(page.getByTestId("preview-dialog")).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(page.getByTestId("preview-dialog")).toBeHidden();

    // Delete the file (from the sources panel) with confirmation.
    await page.getByTestId("open-sources").click();
    const fileDelete = page.getByTestId("file-delete");
    await fileDelete.click();
    await expect(page.getByTestId("confirm-dialog")).toBeVisible();
    await page.getByTestId("confirm-button").click();
    await expect(page.getByTestId("file-row")).toBeHidden();

    // Delete the chat with confirmation; the app resets to a fresh state.
    const row = page.getByTestId("chat-row").first();
    await row.hover();
    await row.getByTestId("chat-delete").click();
    await expect(page.getByTestId("confirm-dialog")).toBeVisible();
    await page.getByTestId("confirm-button").click();
    await expect(page.getByTestId("chat-row")).toHaveCount(0);
    await expect(page.getByTestId("chat-title")).toHaveText("New chat");
});

test("table document journey: ingest, browse chunks, cite", async ({
    page,
}) => {
    await signUp(page);

    // A table document ingests through the csv/table plugins.
    await page.getByTestId("open-sources").click();
    await page.getByTestId("upload-input").setInputFiles(TABLE_DOC);
    await expect(
        page.getByTestId("file-row").filter({ hasText: "customers.csv" }),
    )
        .toContainText("customers.csv", { timeout: 180_000 })
        .catch(async () => {
            console.log(
                "DIAG ingest rows:",
                await page.evaluate(() =>
                    [
                        ...document.querySelectorAll(
                            '[data-testid="ingest-row"]',
                        ),
                    ].map((r) => r.textContent?.slice(0, 160)),
                ),
            );
            console.log(
                "DIAG panel:",
                (await page.getByTestId("sources-panel").textContent())?.slice(
                    0,
                    400,
                ),
            );
            throw new Error("customers.csv never settled into the file list");
        });

    // Browse the file's chunks from the sources panel.
    await page
        .getByTestId("file-row")
        .filter({ hasText: "customers.csv" })
        .getByTestId("file-chunks")
        .click();
    await expect(page.getByTestId("chunks-dialog")).toBeVisible();
    await expect(page.getByTestId("chunks-filename")).toContainText(
        "customers.csv",
    );
    const chunkRows = page.getByTestId("chunk-row");
    await expect(chunkRows.first()).toBeVisible();
    await chunkRows.first().getByTestId("chunk-preview").click();
    await expect(page.getByTestId("chunk-full").first()).not.toBeEmpty();
    await expect(page.getByTestId("chunk-plugin").first()).toContainText(
        "table",
    );
    await page.keyboard.press("Escape");
    await expect(page.getByTestId("chunks-dialog")).toBeHidden();

    // Ask a question about the table and open the cited chunk via its badge.
    await page
        .getByTestId("composer-input")
        .fill("How many customers are in the data?");
    await page.getByTestId("composer-input").press("Enter");
    const answer = page
        .getByTestId("assistant-message")
        .last()
        .getByTestId("markdown-body");
    await expect(answer).toContainText(/customer/i, { timeout: 180_000 });
    // Retries: the model occasionally cites nothing (or an index it was never shown).
    let cited = false;
    for (let attempt = 0; attempt < 4 && !cited; attempt++) {
        cited = await page
            .getByTestId("sources-block")
            .last()
            .waitFor({ state: "visible", timeout: 180_000 })
            .then(() => true)
            .catch(() => false);
        if (!cited) {
            await page
                .getByTestId("composer-input")
                .fill("How many customers are in the data?");
            await page.getByTestId("composer-input").press("Enter");
        }
    }
    await expect(page.getByTestId("sources-block").last()).toBeVisible();
    await expect(
        page.getByTestId("source-item").first().getByTestId("source-filename"),
    ).toContainText("customers.csv");
    await page.getByTestId("source-item").first().click();
    await expect(page.getByTestId("cited-chunk-dialog")).toBeVisible();
    await expect(page.getByTestId("cited-chunk-filename")).toContainText(
        "customers.csv",
    );
    await expect(page.getByTestId("cited-chunk-text")).not.toBeEmpty();
    await page.getByTestId("cited-chunk-view-file").click();
    await expect(page.getByTestId("preview-dialog")).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(page.getByTestId("preview-dialog")).toBeHidden();

    // Delete the table file with confirmation.
    await page.getByTestId("open-sources").click();
    await page
        .getByTestId("file-row")
        .filter({ hasText: "customers.csv" })
        .getByTestId("file-delete")
        .click();
    await expect(page.getByTestId("confirm-dialog")).toBeVisible();
    await page.getByTestId("confirm-button").click();
    await expect(page.getByTestId("file-row")).toBeHidden();
});

test("mid-stream reload re-offers the in-flight question", async ({ page }) => {
    await signUp(page);
    await page.getByTestId("open-sources").click();
    await page.getByTestId("upload-input").setInputFiles(EXAMPLE_DOC);
    await expect(page.getByTestId("file-row")).toContainText("doc.pdf", {
        timeout: 180_000,
    });
    await page.keyboard.press("Escape");
    await expect(page.getByTestId("sources-panel")).toBeHidden();

    const question = "What was the garden originally used for?";
    await page.getByTestId("composer-input").fill(question);
    await page.getByTestId("composer-input").press("Enter");
    // Reload while the run is in flight, before the answer begins. Tool
    // activity is the reliable signal: by then the user turn is checkpointed
    // and no answer exists yet, so the draft must still be offered. Waiting
    // for the first answer delta instead is racy, since a short answer can
    // finish before the reload and (correctly) clear the draft.
    await page
        .getByTestId("tool-steps")
        .last()
        .waitFor({ state: "visible", timeout: 90_000 })
        .catch(() =>
            page
                .getByTestId("markdown-body")
                .last()
                .waitFor({ timeout: 90_000 }),
        );
    await page.reload();
    await expect(page.getByTestId("sidebar")).toBeVisible();

    // The composer is pre-filled with the lost question and a toast explains.
    await expect(page.getByTestId("composer-input")).toHaveValue(question);
    await expect(
        page.getByText("Recovered question", { exact: true }),
    ).toBeVisible();
});

test("mid-ingest reload recovers the background progress", async ({ page }) => {
    await signUp(page);
    await page.getByTestId("open-sources").click();
    // The scanned document takes longer to ingest than a plain text file.
    await page.getByTestId("upload-input").setInputFiles(SCANNED_DOC);
    const ingestRow = page
        .getByTestId("ingest-row")
        .filter({ hasText: "scanned_doc.pdf" });
    await expect(ingestRow).toBeVisible();

    await page.reload();
    await expect(page.getByTestId("sidebar")).toBeVisible();
    // Recovery re-opens the persisted sources panel and re-attaches the job.
    await expect(
        page.getByTestId("ingest-row").filter({ hasText: "scanned_doc.pdf" }),
    ).toBeVisible({ timeout: 30_000 });
    // The job settles into the durable file list.
    await expect(page.getByTestId("file-row")).toContainText(
        "scanned_doc.pdf",
        {
            timeout: 240_000,
        },
    );
});
