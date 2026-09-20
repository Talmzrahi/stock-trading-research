# ═══════════════════════════════════════════════════════════════════════
#  Earnings press releases as structure, not one flattened line.
#
#  Design: research/design_sentiment_v3.md. The v2 downloader kept only
#  flattened text: paragraphs merged, tables smeared into number strings,
#  quote marks deleted. v3 compares a release with the company's earlier
#  releases piece by piece, so it needs the pieces. Two steps:
#
#    compact_html(raw)  what is stored: the exhibit's HTML with styling
#                       stripped but every structural tag kept, so any
#                       later layer can re-parse without re-downloading
#    blocks(html)       what layers read: an ordered list of paragraphs
#                       ("p") and table rows ("row", with cells)
#
#  EDGAR exhibits come from many generators — <p> with <font> (2010s),
#  thousands of nested <div>s (Workiva), whole releases laid out inside
#  tables — so a table is only treated as a table when its cells hold
#  numbers. Otherwise it is page layout and its contents are paragraphs.
# ═══════════════════════════════════════════════════════════════════════

import re

import lxml.html
from lxml import etree

BLOCK_TAGS = {"p", "div", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "center",
              "blockquote", "pre", "dt", "dd", "dl", "section", "article", "body", "td", "th",
              "tr", "table", "hr", "title"}
DROP_TAGS = ["script", "style", "head", "img", "object", "embed"]
KEEP_ATTRS = {"colspan", "rowspan"}

NUMERIC_CELL = re.compile(r"^[\s$€£(]*[-–—]?[\d.,]*\d[\d.,]*\s*[)%]*\s*$")
DASH_CELL = re.compile(r"^[\s$(]*[-–—]+[\s)%]*$")          # accounting zero
BULLET = re.compile(r"^[\s•●▪◦·\-–—*]+$")
ROW, CELL = "\ue000", "\ue001"      # private-use: XML-legal, never in real text


def exhibit(submission):
    """The first EX-99 document in a combined EDGAR submission:
    (type, filename, raw content between <TEXT> tags). None if absent."""
    m = re.search(r"<DOCUMENT>\s*<TYPE>(EX-99[.\d]*)(.*?)</DOCUMENT>", submission, re.S | re.I)
    if not m:
        return None
    body = m.group(2)
    name = re.search(r"<FILENAME>([^\n<]+)", body)
    text = re.search(r"<TEXT>(.*?)</TEXT>", body, re.S | re.I)
    return m.group(1).upper(), (name.group(1).strip() if name else ""), (text.group(1) if text else body)


def _is_html(raw):
    return re.search(r"<(p|div|table|br|font|tr|span)\b", raw, re.I) is not None


def compact_html(raw):
    """Strip everything but structure and text: attributes other than
    colspan/rowspan, scripts, styles, images, comments. Plain-text exhibits
    are kept as they are inside <pre>. Typically 5-10x smaller than raw."""
    if not _is_html(raw):
        pre = etree.Element("pre")
        pre.text = raw
        return etree.tostring(pre, encoding="unicode")
    raw = re.sub(r"^\s*<\?xml[^>]*\?>", "", raw)     # lxml refuses str with a declaration
    root = lxml.html.fromstring(raw)
    for el in root.xpath("|".join(f"//{t}" for t in DROP_TAGS)):
        el.drop_tree()
    for el in root.xpath("//comment()"):
        el.drop_tree()
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        for a in [a for a in el.attrib if a not in KEEP_ATTRS]:
            del el.attrib[a]
    # <font>/<span>/<a> carry no structure; keep their text only. Bold and
    # superscripts stay: they mark headings and footnote references.
    for el in root.xpath("//font|//span|//a|//small|//big"):
        el.drop_tag()
    html = lxml.html.tostring(root, encoding="unicode")
    return re.sub(r">\s+<", "><", html)


C1 = re.compile(r"[\x80-\x9f]")


def _fix_chars(text):
    """Old filings write curly quotes, dashes and bullets as Windows-1252
    codes (&#146; &#151; &#149;), which decode to invisible control
    characters: "Company's" came out as "Companys". Map them back."""
    return C1.sub(lambda m: bytes([ord(m.group())]).decode("cp1252", errors="ignore") or " ", text)


def _norm(text):
    return re.sub(r"\s+", " ", _fix_chars(text).replace("\xa0", " ")).strip()


def _flatten_whitespace(root):
    """Source files hard-wrap long lines, so newlines inside text are just
    spaces. Collapse them before structural line breaks are added, or a
    paragraph splits wherever the generator happened to wrap it."""
    for el in root.iter():
        if not isinstance(el.tag, str) or el.tag == "pre":
            continue
        if el.text:
            el.text = re.sub(r"\s+", " ", el.text)
        if el.tail:
            el.tail = re.sub(r"\s+", " ", el.tail)


def _clean_cells(cells):
    """EDGAR splits "$ 6,582" and "(12 )" and "13 %" across cells; rejoin
    them so each cell is one value, and drop the empty spacer cells."""
    out = []
    for c in (c for c in cells if c):
        if out and out[-1] in {"$", "€", "£", "("}:
            out[-1] = out[-1] + c
        elif out and c in {")", "%", ")%", "%)", "pts", "bps"}:
            out[-1] = out[-1] + c
        else:
            out.append(c)
    return out


def _row_cells(tr):
    return _clean_cells([_norm(td.text_content()) for td in tr.xpath("./td|./th")])


def _is_data_table(table):
    """A data table has numbers in its cells; a layout table holds prose."""
    rows = [_row_cells(tr) for tr in table.xpath(".//tr")]
    rows = [r for r in rows if r]
    if not rows:
        return False
    numeric = sum(any(NUMERIC_CELL.match(c) or DASH_CELL.match(c) for c in r) for r in rows)
    longest = max(len(c) for r in rows for c in r)
    return numeric / len(rows) >= 0.3 and not (len(rows) <= 2 and longest > 300)


def _replace(old, new):
    new.tail = old.tail
    old.getparent().replace(old, new)


def blocks(html):
    """Ordered paragraphs and table rows.

    Returns a list of {"k": "p", "t": text} and {"k": "row", "c": [cells]}.
    Tables are resolved innermost first: a data table becomes one line per
    row; a layout table is unwrapped so its contents read as paragraphs.
    """
    root = lxml.html.fromstring(html)
    if root.tag == "pre" or (len(root) == 0 and not _is_html(html)):
        return _plain_blocks(_fix_chars(root.text_content()))
    _flatten_whitespace(root)

    while True:
        leaves = root.xpath("//table[not(.//table)]")
        if not leaves:
            break
        for table in leaves:
            holder = etree.Element("div")
            if _is_data_table(table):
                lines = [ROW + CELL.join(cells) for cells in map(_row_cells, table.xpath(".//tr")) if cells]
                holder.text = "\n" + "\n".join(lines) + "\n"
            else:
                for cell in table.xpath(".//td|.//th"):
                    part = etree.SubElement(holder, "div")
                    part.text = cell.text
                    for child in list(cell):
                        part.append(child)
            _replace(table, holder)

    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        if el.tag in BLOCK_TAGS:
            el.text = "\n" + (el.text or "")
            el.tail = "\n" + (el.tail or "")
        elif el.tag == "br":
            el.tail = "\n" + (el.tail or "")

    out = []
    for line in root.text_content().split("\n"):
        if line.startswith(ROW):
            cells = [c for c in (_norm(c) for c in line[1:].split(CELL)) if c]
            if len(cells) == 1 and len(cells[0]) > 120:
                out.append({"k": "p", "t": cells[0]})      # a footnote inside a table
            elif cells:
                out.append({"k": "row", "c": cells})
            continue
        text = _norm(line)
        if text and not BULLET.match(text):
            out.append({"k": "p", "t": text})
    return out


def _plain_blocks(text):
    """Plain-text exhibits: blank lines separate paragraphs; lines with
    wide gaps and numbers are table rows."""
    out = []
    for chunk in re.split(r"\n\s*\n", text):
        lines = [l for l in chunk.split("\n") if l.strip()]
        rowish = [l for l in lines if re.search(r"\S {2,}\S", l) and re.search(r"\d", l)]
        if lines and len(rowish) / len(lines) >= 0.5:
            out += [{"k": "row", "c": _clean_cells([_norm(c) for c in re.split(r" {2,}", l.strip())])}
                    for l in lines]
        elif lines:
            out.append({"k": "p", "t": _norm(" ".join(lines))})
    return out
