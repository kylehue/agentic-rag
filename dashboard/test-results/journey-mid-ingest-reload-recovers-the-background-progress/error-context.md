# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: journey.spec.ts >> mid-ingest reload recovers the background progress
- Location: e2e/journey.spec.ts:240:1

# Error details

```
Error: expect(locator).toBeVisible() failed

Locator: getByTestId('ingest-row').filter({ hasText: 'scanned_doc.pdf' })
Expected: visible
Timeout: 30000ms
Error: element(s) not found

Call log:
  - Expect "toBeVisible" getByTestId('ingest-row').filter({ hasText: 'scanned_doc.pdf' }) with timeout 30000ms
  - waiting for getByTestId('ingest-row').filter({ hasText: 'scanned_doc.pdf' })

```

```yaml
- complementary:
  - text: Agent
  - button "New chat"
  - navigation "Chats":
    - button "Chat 63f6a872 1 source · 23:43 Delete Chat 63f6a872":
      - paragraph: Chat 63f6a872
      - paragraph: 1 source · 23:43
      - button "Delete Chat 63f6a872"
  - text: E e2e_52019e5c
  - button "Sign out"
- main:
  - heading "Chat 63f6a872" [level=1]
  - paragraph: 1 source
  - button "Switch to dark theme"
  - button "Sources 1" [expanded]
  - heading "Ask your documents" [level=2]
  - paragraph: Upload files in the sources panel, then ask a question about them.
  - button "Add files"
  - textbox "Ask a question about your documents"
  - button "Send" [disabled]
  - paragraph: Enter to send · Shift+Enter for a new line
- dialog "Sources":
  - heading "Sources" [level=2]
  - paragraph: Your ingested files and their progress
  - heading "Sources" [level=2]
  - text: "1"
  - button "Click to upload or drag and drop PDF, Office, text, slides, and spreadsheets":
    - paragraph: Click to upload or drag and drop
    - paragraph: PDF, Office, text, slides, and spreadsheets
  - heading "Files" [level=3]
  - button "Preview scanned_doc.pdf": scanned_doc.pdf
  - text: PDF · c7bee22f
  - button "View scanned_doc.pdf"
  - button "View chunks of scanned_doc.pdf"
  - button "Delete scanned_doc.pdf"
- region "Notifications (F8)"
```

# Test source

```ts
  155 |         .getByTestId("assistant-message")
  156 |         .last()
  157 |         .getByTestId("markdown-body");
  158 |     await expect(answer).toContainText(/customer/i, { timeout: 180_000 });
  159 |     // Retries: the model occasionally cites nothing (or an index it was never shown).
  160 |     let cited = false;
  161 |     for (let attempt = 0; attempt < 4 && !cited; attempt++) {
  162 |         cited = await page
  163 |             .getByTestId("sources-block")
  164 |             .last()
  165 |             .waitFor({ state: "visible", timeout: 180_000 })
  166 |             .then(() => true)
  167 |             .catch(() => false);
  168 |         if (!cited) {
  169 |             await page
  170 |                 .getByTestId("composer-input")
  171 |                 .fill("How many customers are in the data?");
  172 |             await page.getByTestId("composer-input").press("Enter");
  173 |         }
  174 |     }
  175 |     await expect(page.getByTestId("sources-block").last()).toBeVisible();
  176 |     await expect(
  177 |         page.getByTestId("source-item").first().getByTestId("source-filename"),
  178 |     ).toContainText("customers.csv");
  179 |     await page.getByTestId("source-item").first().click();
  180 |     await expect(page.getByTestId("cited-chunk-dialog")).toBeVisible();
  181 |     await expect(page.getByTestId("cited-chunk-filename")).toContainText(
  182 |         "customers.csv",
  183 |     );
  184 |     await expect(page.getByTestId("cited-chunk-text")).not.toBeEmpty();
  185 |     await page.getByTestId("cited-chunk-view-file").click();
  186 |     await expect(page.getByTestId("preview-dialog")).toBeVisible();
  187 |     await page.keyboard.press("Escape");
  188 |     await expect(page.getByTestId("preview-dialog")).toBeHidden();
  189 | 
  190 |     // Delete the table file with confirmation.
  191 |     await page.getByTestId("open-sources").click();
  192 |     await page
  193 |         .getByTestId("file-row")
  194 |         .filter({ hasText: "customers.csv" })
  195 |         .getByTestId("file-delete")
  196 |         .click();
  197 |     await expect(page.getByTestId("confirm-dialog")).toBeVisible();
  198 |     await page.getByTestId("confirm-button").click();
  199 |     await expect(page.getByTestId("file-row")).toBeHidden();
  200 | });
  201 | 
  202 | test("mid-stream reload re-offers the in-flight question", async ({ page }) => {
  203 |     await signUp(page);
  204 |     await page.getByTestId("open-sources").click();
  205 |     await page.getByTestId("upload-input").setInputFiles(EXAMPLE_DOC);
  206 |     await expect(page.getByTestId("file-row")).toContainText("doc.pdf", {
  207 |         timeout: 180_000,
  208 |     });
  209 |     await page.keyboard.press("Escape");
  210 |     await expect(page.getByTestId("sources-panel")).toBeHidden();
  211 | 
  212 |     const question = "What was the garden originally used for?";
  213 |     await page.getByTestId("composer-input").fill(question);
  214 |     await page.getByTestId("composer-input").press("Enter");
  215 |     // Reload while the run is in flight, before the answer begins. Tool
  216 |     // activity is the reliable signal: by then the user turn is checkpointed
  217 |     // and no answer exists yet, so the draft must still be offered. Waiting
  218 |     // for the first answer delta instead is racy, since a short answer can
  219 |     // finish before the reload and (correctly) clear the draft.
  220 |     await page
  221 |         .getByTestId("tool-steps")
  222 |         .last()
  223 |         .waitFor({ state: "visible", timeout: 90_000 })
  224 |         .catch(() =>
  225 |             page
  226 |                 .getByTestId("markdown-body")
  227 |                 .last()
  228 |                 .waitFor({ timeout: 90_000 }),
  229 |         );
  230 |     await page.reload();
  231 |     await expect(page.getByTestId("sidebar")).toBeVisible();
  232 | 
  233 |     // The composer is pre-filled with the lost question and a toast explains.
  234 |     await expect(page.getByTestId("composer-input")).toHaveValue(question);
  235 |     await expect(
  236 |         page.getByText("Recovered question", { exact: true }),
  237 |     ).toBeVisible();
  238 | });
  239 | 
  240 | test("mid-ingest reload recovers the background progress", async ({ page }) => {
  241 |     await signUp(page);
  242 |     await page.getByTestId("open-sources").click();
  243 |     // The scanned document takes longer to ingest than a plain text file.
  244 |     await page.getByTestId("upload-input").setInputFiles(SCANNED_DOC);
  245 |     const ingestRow = page
  246 |         .getByTestId("ingest-row")
  247 |         .filter({ hasText: "scanned_doc.pdf" });
  248 |     await expect(ingestRow).toBeVisible();
  249 | 
  250 |     await page.reload();
  251 |     await expect(page.getByTestId("sidebar")).toBeVisible();
  252 |     // Recovery re-opens the persisted sources panel and re-attaches the job.
  253 |     await expect(
  254 |         page.getByTestId("ingest-row").filter({ hasText: "scanned_doc.pdf" }),
> 255 |     ).toBeVisible({ timeout: 30_000 });
      |       ^ Error: expect(locator).toBeVisible() failed
  256 |     // The job settles into the durable file list.
  257 |     await expect(page.getByTestId("file-row")).toContainText(
  258 |         "scanned_doc.pdf",
  259 |         {
  260 |             timeout: 240_000,
  261 |         },
  262 |     );
  263 | });
  264 | 
```