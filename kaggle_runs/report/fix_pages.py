"""Page column of contents / list of tables / list of figures: body page = PDF page - (page of 'CHAPTER 1') + 1."""
import re, sys, docx, pymupdf
from docx.oxml.ns import qn
pdf = sys.argv[1]; src = sys.argv[2] if len(sys.argv) > 2 else None; dst = sys.argv[3] if len(sys.argv) > 3 else None
doc = pymupdf.open(pdf)
norm = lambda s: re.sub(r"[^a-z0-9.:()\- ]", "", re.sub(r"\s+", " ", s.lower())).strip()
pages_lines = [[re.sub(r"\s+", " ", l).strip() for l in pg.get_text().splitlines() if l.strip()] for pg in doc]
texts = [norm(" ".join(ls)) for ls in pages_lines]
ch1 = next(i for i, t in enumerate(texts) if i > 12 and t.startswith("chapter 1"))
NUM = r"(\d+(\.\d+)*\.?\s+)?"
def find_line(title, prefix="", start=0):
    n = norm(title)
    for i in range(max(ch1, start), len(texts)):
        for l in pages_lines[i]:
            ln = norm(l)
            if prefix:
                if ln.startswith(norm(prefix)):
                    return i
            elif re.fullmatch(NUM + re.escape(n), ln):
                return i
    return None
cursor = [ch1]
def heading_page(title, prefix=""):
    i = find_line(title, prefix, cursor[0] if not prefix else ch1)
    if i is None:
        return None
    if not prefix:
        cursor[0] = i
    return str(i - ch1 + 1)
if src is None:
    for k in ["chapter 2", "chapter 3", "chapter 4", "chapter 5", "chapter 6", "chapter 7", "references", "appendix 2"]:
        print(k, next((i - ch1 + 1 for i in range(ch1, len(texts)) if texts[i].startswith(k) or (" " + k) in texts[i][:40]), None))
    sys.exit()
D = docx.Document(src)
T = [docx.table.Table(e, D) for e in D.element.body.iterchildren() if e.tag == qn("w:tbl")]
changes, missing = [], []
def setc(cell, v):
    rs = [x for p in cell.paragraphs for x in p.runs]
    if rs and cell.text.strip() != v:
        changes.append(f"{cell.text.strip()}→{v}"); rs[0].text = v
        for x in rs[1:]: x.text = ""
for r in T[2].rows[1:]:
    title, pg = r.cells[1].text.strip(), r.cells[2].text.strip()
    if not re.fullmatch(r"\d+", pg): continue
    p = heading_page(title)
    (setc(r.cells[2], p) if p else missing.append(title))
for t, prefix in ((3, "table"), (4, "figure")):
    for r in T[t].rows[1:]:
        no, title = r.cells[0].text.strip(), r.cells[1].text.strip()
        p = heading_page(title, f"{prefix} {no}:")
        if p is None:
            i = find_line(title, "", ch1); p = None if i is None else str(i - ch1 + 1)
        (setc(r.cells[2], p) if p else missing.append(f"{prefix} {no}"))
D.save(dst); print(len(changes), "page cells changed:", changes); print("not found:", missing)
