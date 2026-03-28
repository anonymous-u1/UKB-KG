import os
import re
from bs4 import BeautifulSoup
from typing import Optional, List
from pathlib import Path
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdout.reconfigure(line_buffering=True)

# ========== UTILS ==========
def build_xml_index(directory: Path) -> dict:
    """
    Build an index {filename: full_path} for fast lookup of XML files under a directory.
    """
    xml_index = {}
    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.endswith(".xml"):
                xml_index[fname] = os.path.join(root, fname)
    return xml_index


def find_xml_file(pmcid: str, xml_index: dict) -> Optional[str]:
    """ Find XML path by PMCID using pre-built index. """
    if not pmcid.endswith(".xml"):
        pmcid += ".xml"
    return xml_index.get(pmcid)


def clean_text(el):
    if el is None:
        return ""
    for sup in el.find_all("sup"):
        label = sup.get_text(strip=True)
        sup.replace_with(f"<sup>{label}</sup>")
    txt = el.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", txt)


def emphasize_section_rows(matrix):
    for i, row in enumerate(matrix[1:], start=1):
        if row[0] and all(not c.strip() for c in row[1:]):
            row[0] = f"**{row[0]}**"
    return matrix


# ========== Table Parse ==========
def rows_to_grid(tr_list):
    grid, span = [], {}
    for tr in tr_list:
        row, c = [], 0

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
                # 跳过被合并占位的列
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
    """Extracte table matrix from <table-wrap>"""
    table = table_wrap.find("table")
    if not table:
        return None

    thead, tbody = table.find("thead"), table.find("tbody")
    if thead:
        head_rows = thead.find_all("tr", recursive=False)
        body_rows = tbody.find_all("tr", recursive=False) if tbody else table.find_all("tr", recursive=False)[len(head_rows):]
    else:
        all_tr = table.find_all("tr", recursive=False)

        for i, tr in enumerate(all_tr):
            if tr.find("td"):
                head_rows, body_rows = all_tr[:i], all_tr[i:]
                break
        else:
            head_rows, body_rows = all_tr[:1], all_tr[1:]

    header_grid, body_grid = rows_to_grid(head_rows), rows_to_grid(body_rows)

    # Merging multi-level headers
    header = []
    if header_grid:
        for col in range(len(header_grid[0])):
            parts = [header_grid[row][col].strip() for row in range(len(header_grid)) if header_grid[row][col].strip()]
            header.append(" / ".join(dict.fromkeys(parts)))

    width = max(len(header), max((len(r) for r in body_grid), default=0))
    header += [""] * (width - len(header))
    for r in body_grid:
        r += [""] * (width - len(r))

    matrix = [header] + body_grid
    return emphasize_section_rows(matrix)


# ========== Extract footnote ==========
def extract_footnotes(t):
    """Extract footnote from <table-wrap-foot>"""
    notes, foot = [], t.find("table-wrap-foot")
    if not foot:
        return notes

    for p_tag in foot.find_all("p", recursive=False):
        text_html = "".join(str(x) for x in p_tag.contents).strip()
        text_html = re.sub(r"\s+", " ", text_html)
        if text_html:
            notes.append(text_html)

    for fn in foot.find_all("fn", recursive=False):
        label = "".join(str(x) for x in fn.find("label").contents).strip() if fn.find("label") else ""
        p_tag = fn.find("p")
        text = "".join(str(x) for x in p_tag.contents).strip() if p_tag else fn.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text)
        if text:
            notes.append(f"{label} {text}" if label else text)
    return notes


# ========== Markdown conversion ==========
def to_markdown(matrix):
    lines = []
    head, body = matrix[0], matrix[1:]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("| " + " | ".join(["---"] * len(head)) + " |")
    for r in body:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


# ========== Main parse function ==========
def parse_table(xml_path, table_path):
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
        md_table = to_markdown(matrix)
        md_text = f"### Table {i}: \n\n**{title}**\n\n{md_table}\n\n"

        footnotes = extract_footnotes(t)
        if footnotes:
            md_text += "\n\n".join(f"*{n}*" for n in footnotes)

        markdown_tables.append(md_text)

    # 保存结果
    if markdown_tables:
        output = "\n\n---\n\n".join(markdown_tables)
        os.makedirs(os.path.dirname(table_path), exist_ok=True)
        with open(table_path, "w", encoding="utf-8") as out_f:
            out_f.write(output)
        print(f"✅ Extracted {len(markdown_tables)} tables from {os.path.basename(xml_path)}")
        return len(markdown_tables)
    else:
        print(f"⚠️ No valid tables extracted from {os.path.basename(xml_path)}")
        return 0


def main():
    xml_dir = Path("PMC_Data/xml")
    table_dir = Path("PMC_Data/table")
    paper_list = Path("PMC_Data/pmcid-ukb-xml-sample.txt")

    with open(paper_list, "r", encoding="utf-8") as f:
        file_names = [line.strip() for line in f if line.strip()]

    print("Building XML index...")
    xml_index = build_xml_index(Path(xml_dir))
    print(f"✅ Indexed {len(xml_index)} XML files under {xml_dir}")

    total_tables = 0
    valid_papers = 0

    for file_name in file_names:
        xml_path = find_xml_file(file_name, xml_index)
        table_path = table_dir / (file_name.replace(".xml", ".txt"))
        num_tables = parse_table(xml_path, table_path)
        if num_tables > 0:
            valid_papers += 1
            total_tables += num_tables

    print("\n====================== Summary ======================")
    print(f"✅ Total papers processed: {len(file_names)}")
    print(f"✅ Papers with at least one valid table: {valid_papers}")
    print(f"✅ Total tables extracted: {total_tables}")
    print("=====================================================")

if __name__ == "__main__":
    main()
