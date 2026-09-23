export type FileKind =
    "pdf" | "spreadsheet" | "image" | "document" | "text" | "slides" | "other";

const EXT_MAP: Record<string, FileKind> = {
    pdf: "pdf",
    csv: "spreadsheet",
    xlsx: "spreadsheet",
    xls: "spreadsheet",
    png: "image",
    jpg: "image",
    jpeg: "image",
    gif: "image",
    webp: "image",
    svg: "image",
    bmp: "image",
    ico: "image",
    doc: "document",
    docx: "document",
    odt: "document",
    rtf: "document",
    epub: "document",
    txt: "text",
    md: "text",
    markdown: "text",
    html: "text",
    htm: "text",
    rst: "text",
    json: "text",
    xml: "text",
    ppt: "slides",
    pptx: "slides",
};

export const FILE_KIND_LABELS: Record<FileKind, string> = {
    pdf: "PDF",
    spreadsheet: "Spreadsheet",
    image: "Image",
    document: "Document",
    text: "Text",
    slides: "Slides",
    other: "File",
};

function extOf(filename: string): string {
    const dot = filename.lastIndexOf(".");
    if (dot <= 0) return "";
    return filename.slice(dot + 1).toLowerCase();
}

// Category from the content type, with a filename extension fallback.
export function fileKind(contentType: string, filename: string): FileKind {
    const ct = (contentType || "").toLowerCase();
    if (ct.includes("pdf")) return "pdf";
    if (ct.startsWith("image/")) return "image";
    if (ct.includes("spreadsheet") || ct.includes("excel") || ct === "text/csv")
        return "spreadsheet";
    if (ct.includes("presentation") || ct.includes("powerpoint"))
        return "slides";
    if (
        ct.includes("msword") ||
        ct.includes("opendocument.text") ||
        ct === "application/rtf" ||
        ct === "application/epub+zip"
    )
        return "document";
    if (ct.startsWith("text/")) return "text";
    if (ct && ct !== "application/octet-stream") return "other";
    return EXT_MAP[extOf(filename)] ?? "other";
}

// Today: the time; yesterday: "Yesterday"; under a week: "N d ago";
// beyond that a short date. A future timestamp (clock skew) renders as
// "Yesterday".
export function relativeDate(
    unixSeconds: number,
    now: Date = new Date(),
): string {
    const then = new Date(unixSeconds * 1000);
    if (Number.isNaN(then.getTime())) return "";
    if (then.getTime() > now.getTime()) return "Yesterday";

    const startOf = (d: Date) =>
        new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
    const dayDiff = Math.round((startOf(now) - startOf(then)) / 86_400_000);
    if (dayDiff <= 0)
        return then.toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
            hour12: false,
        });
    if (dayDiff === 1) return "Yesterday";
    if (dayDiff < 7) return `${dayDiff} d ago`;

    const sameYear = then.getFullYear() === now.getFullYear();
    return then.toLocaleDateString([], {
        month: "short",
        day: "numeric",
        ...(sameYear ? {} : { year: "numeric" }),
    });
}

// Zero-byte sizes render as an empty string on purpose; values under 10
// units get one decimal.
export function byteSize(bytes: number): string {
    if (!Number.isFinite(bytes) || bytes <= 0) return "";
    const units = ["B", "KB", "MB", "GB", "TB"];
    let value = bytes;
    let unit = 0;
    while (value >= 1024 && unit < units.length - 1) {
        value /= 1024;
        unit++;
    }
    const text = value < 10 ? value.toFixed(1) : String(Math.round(value));
    return `${text} ${units[unit]}`;
}

export function shortId(id: string): string {
    return id.slice(0, 8);
}

// One-line JSON for tool arguments, capped for display (full text lives in
// the tooltip).
export function truncateArgs(args: unknown, max = 60): string {
    let text: string;
    try {
        text = JSON.stringify(args ?? {});
    } catch {
        text = String(args);
    }
    text = text.replace(/\s+/g, " ");
    if (text.length <= max) return text;
    return `${text.slice(0, max)}…`;
}
