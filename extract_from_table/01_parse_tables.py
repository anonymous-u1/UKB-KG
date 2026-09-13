"""Stage 1 (table): render every <table-wrap> of an article as markdown.

Multi-row headers are flattened with " / ", rowspan/colspan are expanded into
a dense grid, section rows (a label with an otherwise empty row) are bolded,
and table footnotes are appended in italics -- all so that the LLM sees a
table whose cells still line up with their headers.

One .txt per article, containing all of its tables separated by "---".
"""

import argparse
import io
import os
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

from utils.pmc_xml import build_xml_index, find_xml_file, read_paper_list

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stdout.reconfigure(line_buffering=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xml-dir", required=True)
    p.add_argument("--output-dir", required=True, help="Where the .txt files go.")
    p.add_argument("--paper-list", required=True)
    return p.parse_args()


# ========== UTILS ==========
def clean_text(el):
    if el is None:
        return ""
    for sup in el.find_all("sup"):
        label = sup.get_text(strip=True)
        sup.replace_with(f"<sup>{label}</sup>")
    txt = el.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", txt)


def emphasize_section_rows(matrix):
    """Bold the label of rows that carry no data (they head a table section)."""
    for row in matrix[1:]:
        if row[0] and all(not c.strip() for c in row[1:]):
            row[0] = f"**{row[0]}**"
    return matrix


# ========== Table Parse ==========
def rows_to_grid(tr_list):
    """Expand rowspan/colspan into a dense rectangular grid."""
    grid, span = [], {}
    for tr in tr_list:
        row, c = [], 0

        # Carry down cells still spanning from earlier rows.
        while span.get(c, {}).get("down", 0) > 0:
            row.append(span[c]["text"])
            span[c]["down"] -= 1
            c += 1

        for cell in tr.find_all(["th", "td"], recursive=False):
            rs, cs = int(cell.get("rowspan", 1)), int(cell.get("colspan", 1))
            txt = clean_text(cell)

            for _ in range(cs):
                row.append(txt)
                if rs > 1:
                    span[c] = {"down": rs - 1, "text": txt}
                c += 1
                # Skip columns occupied by a cell spanning down into this row.
                while span.get(c, {}).get("down", 0) > 0:
                    row.append(span[c]["text"])
                    span[c]["down"] -= 1
                    c += 1

        grid.append(row)

    width = max((len(r) for r in grid), default=0)
    for r in grid:
        r.extend([""] * (width - len(r)))
    return grid


def extract_table(table_wrap):
    """Extract the table matrix from a <table-wrap>."""
    table = table_wrap.find("table")
    if not table:
        return None

    thead, tbody = table.find("thead"), table.find("tbody")
    if thead:
        head_rows = thead.find_all("tr", recursive=False)
        body_rows = (
            tbody.find_all("tr", recursive=False)
            if tbody
            else table.find_all("tr", recursive=False)[len(head_rows):]
        )
    else:
        all_tr = table.find_all("tr", recursive=False)
        # No <thead>: the header is everything before the first row with a <td>.
        for i, tr in enumerate(all_tr):
            if tr.find("td"):
                head_rows, body_rows = all_tr[:i], all_tr[i:]
                break
        else:
            head_rows, body_rows = all_tr[:1], all_tr[1:]

    header_grid, body_grid = rows_to_grid(head_rows), rows_to_grid(body_rows)

    # Flatten multi-level headers into one row.
    header = []
    if header_grid:
        for col in range(len(header_grid[0])):
            parts = [
                header_grid[row][col].strip()
                for row in range(len(header_grid))
                if header_grid[row][col].strip()
            ]
            header.append(" / ".join(dict.fromkeys(parts)))

    width = max(len(header), max((len(r) for r in body_grid), default=0))
    header += [""] * (width - len(header))
    for r in body_grid:
        r += [""] * (width - len(r))

    return emphasize_section_rows([header] + body_grid)


# ========== Extract footnote ==========
def extract_footnotes(t):
    """Extract footnotes from <table-wrap-foot>."""
    notes, foot = [], t.find("table-wrap-foot")
    if not foot:
        return notes

    for p_tag in foot.find_all("p", recursive=False):
        text_html = "".join(str(x) for x in p_tag.contents).strip()
        text_html = re.sub(r"\s+", " ", text_html)
        if text_html:
            notes.append(text_html)

    for fn in foot.find_all("fn", recursive=False):
        label = (
            "".join(str(x) for x in fn.find("label").contents).strip()
            if fn.find("label")
            else ""
        )
        p_tag = fn.find("p")
        text = (
            "".join(str(x) for x in p_tag.contents).strip()
            if p_tag
            else fn.get_text(" ", strip=True)
        )
        text = re.sub(r"\s+", " ", text)
        if text:
            notes.append(f"{label} {text}" if label else text)
    return notes


# ========== Markdown conversion ==========
def to_markdown(matrix):
    head, body = matrix[0], matrix[1:]
    lines = [
        "| " + " | ".join(head) + " |",
        "| " + " | ".join(["---"] * len(head)) + " |",
    ]
    lines.extend("| " + " | ".join(r) + " |" for r in body)
    return "\n".join(lines)


# ========== Main parse function ==========
def parse_tables(xml_path, table_path):
    try:
        with open(xml_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "lxml")
    except Exception as e:
        print(f"❌ Failed to parse {xml_path}: {e}")
        return 0

    tables = soup.find_all("table-wrap")
    if not tables:
        print(f"⚠️ No tables found in {os.path.basename(xml_path)}")
        return 0

    markdown_tables = []
    for i, t in enumerate(tables, 1):
        caption = t.find("caption")
        title = caption.get_text(strip=True) if caption else f"Table {i}"
        matrix = extract_table(t)
        if not matrix:
            continue
        md_text = f"### Table {i}: \n\n**{title}**\n\n{to_markdown(matrix)}\n\n"

        footnotes = extract_footnotes(t)
        if footnotes:
            md_text += "\n\n".join(f"*{n}*" for n in footnotes)

        markdown_tables.append(md_text)

    if not markdown_tables:
        print(f"⚠️ No valid tables extracted from {os.path.basename(xml_path)}")
        return 0

    os.makedirs(os.path.dirname(table_path), exist_ok=True)
    with open(table_path, "w", encoding="utf-8") as out_f:
        out_f.write("\n\n---\n\n".join(markdown_tables))
    print(f"✅ Extracted {len(markdown_tables)} tables from {os.path.basename(xml_path)}")
    return len(markdown_tables)


def main():
    args = parse_args()
    table_dir = Path(args.output_dir)
    file_names = read_paper_list(args.paper_list)

    print(f"Building XML index under {args.xml_dir} ...")
    xml_index = build_xml_index(args.xml_dir)
    print(f"✅ Indexed {len(xml_index)} XML files")

    total_tables = 0
    valid_papers = 0

    for file_name in file_names:
        xml_path = find_xml_file(file_name, xml_index)
        if xml_path is None:
            print(f"⚠️ No XML file found for {file_name}")
            continue
        num_tables = parse_tables(xml_path, table_dir / file_name.replace(".xml", ".txt"))
        if num_tables > 0:
            valid_papers += 1
            total_tables += num_tables

    print("\n====================== Summary ======================")
    print(f"✅ Total papers processed: {len(file_names)}")
    print(f"✅ Papers with at least one valid table: {valid_papers}")
    print(f"✅ Total tables extracted: {total_tables}")
    print(f"✅ Tables written to: {table_dir}")
    print("=====================================================")


if __name__ == "__main__":
    main()
