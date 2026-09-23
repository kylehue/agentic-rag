import hljs from "highlight.js";
import MarkdownItCallable, {
    type MarkdownIt,
    type StateInline,
    type Token,
} from "markdown-it";
import { citationIndex } from "./citations";

// Index-based markers (#[1], #[2], ...): the integer is assigned by the
// server per answer, so the marker carries its own badge number.
const CITE_RULE_RE = new RegExp(String.raw`^#\[\d+\]`);

// Inline rule: a completed citation marker becomes a real button. The code
// rule runs earlier in the ruler, so markers inside code spans are consumed
// by it and never become badges.
function citeRule(state: StateInline, silent: boolean): boolean {
    if (state.src[state.pos] !== "#") return false;
    const match = CITE_RULE_RE.exec(state.src.slice(state.pos));
    if (!match) return false;
    if (!silent) {
        const token = state.push("cite", "", 0);
        token.meta = { ref: match[0] };
    }
    state.pos += match[0].length;
    return true;
}

function escapeHtml(text: string): string {
    return text
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
}

interface CiteEnv {
    // An answer that cites a single chunk shows no inline badge: the sources
    // footer already carries the citation, and the marker is pure noise.
    hideCitations?: boolean;
}

function citeRenderer(
    tokens: Token[],
    index: number,
    _options: unknown,
    env: unknown,
): string {
    if ((env as CiteEnv | undefined)?.hideCitations) return "";
    const ref = (tokens[index].meta?.ref as string | undefined) ?? "";
    const number = citationIndex(ref);
    const label = number == null ? "" : String(number);
    return (
        `<button type="button" class="citation-badge" ` +
        `data-cite="${label}" data-ref="${escapeHtml(ref)}">${label}</button>`
    );
}

// html mode off: raw HTML in answers stays escaped, or answers become an XSS
// vector. linkify on: bare URLs become links. breaks on: single newlines
// stay visible.
export function createMarkdownRenderer(): MarkdownIt {
    const md = new MarkdownItCallable({
        html: false,
        linkify: true,
        breaks: true,
        highlight(code, lang) {
            if (lang && hljs.getLanguage(lang)) {
                try {
                    return hljs.highlight(code, {
                        language: lang,
                        ignoreIllegals: true,
                    }).value;
                } catch {
                    // fall through to the default escaping
                }
            }
            return "";
        },
    });
    md.inline.ruler.before("link", "cite", citeRule);
    md.renderer.rules.cite = citeRenderer;
    return md;
}

export function renderMarkdown(
    md: MarkdownIt,
    content: string,
    hideCitations = false,
): string {
    return md.render(content, { hideCitations });
}
