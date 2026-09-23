import { describe, expect, it } from "vitest";
import { createMarkdownRenderer, renderMarkdown } from "./markdown";

const md = createMarkdownRenderer();

function render(content: string, hideCitations = false): string {
    return renderMarkdown(md, content, hideCitations);
}

describe("markdown rendering", () => {
    it("renders headings, lists, and blockquotes", () => {
        const html = render("# Title\n\n- one\n- two\n\n> quoted");
        expect(html).toContain("<h1>Title</h1>");
        expect(html).toContain("<li>one</li>");
        expect(html).toContain("<blockquote>");
    });

    it("renders tables", () => {
        const html = render("| a | b |\n| - | - |\n| 1 | 2 |");
        expect(html).toContain("<table>");
        expect(html).toContain("<th>a</th>");
        expect(html).toContain("<td>1</td>");
    });

    it("escapes raw html", () => {
        const html = render("<script>alert(1)</script>");
        expect(html).not.toContain("<script>");
        expect(html).toContain("&lt;script&gt;");
    });

    it("turns bare urls into links", () => {
        const html = render("see https://example.com/x");
        expect(html).toContain('<a href="https://example.com/x"');
    });

    it("keeps single newlines visible", () => {
        const html = render("line one\nline two");
        expect(html).toContain("<br>");
    });

    it("highlights fenced code blocks", () => {
        const html = render("```js\nconst a = 1\n```");
        expect(html).toContain("<pre><code");
        expect(html).toContain("hljs-");
    });

    it("renders code spans with escaped content", () => {
        const html = render("use `foo()`");
        expect(html).toContain("<code>foo()</code>");
    });
});

describe("citation badges", () => {
    it("renders markers as numbered buttons carrying their own number", () => {
        const html = render("first #[1] and #[2] again #[1]");
        expect(html).toContain('data-cite="1" data-ref="#[1]"');
        expect(html).toContain('data-cite="2" data-ref="#[2]"');
        expect(html).not.toContain("<p>first #[1]");
    });

    it("supports multi-digit markers", () => {
        const html = render("see #[12] for details");
        expect(html).toContain('data-cite="12" data-ref="#[12]"');
    });

    it("repeats the same number for repeated markers", () => {
        const html = render("#[1] ... #[1]");
        const count = html.match(/data-cite="1"/g)?.length ?? 0;
        expect(count).toBe(2);
    });

    it("never badge-ifies markers inside code spans", () => {
        const html = render("literal `#[1]` stays");
        expect(html).not.toContain("citation-badge");
        expect(html).toContain("<code>#[1]</code>");
    });

    it("leaves unfinished markers as text while streaming", () => {
        const html = render("partial #[1");
        expect(html).not.toContain("citation-badge");
        expect(html).toContain("#[1");
    });

    it("does not badge-ify old-format id markers", () => {
        const html = render("legacy #[origin1:chunk1] stays literal");
        expect(html).not.toContain("citation-badge");
        expect(html).toContain("#[origin1:chunk1]");
    });

    it("drops the markers entirely when citations are hidden", () => {
        const html = render("the fact #[1] is here", true);
        expect(html).not.toContain("citation-badge");
        expect(html).not.toContain("#[1]");
        expect(html).toContain("the fact");
    });
});
