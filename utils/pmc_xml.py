"""Section extraction from PMC OA JATS XML.

An article is reduced to Title + Abstract + (Discussion or Results) +
Conclusion.  Discussion and Results are mutually exclusive: Results is only
used as a fallback for articles that have no Discussion section.
"""

import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import pubmed_parser as pp
import tiktoken
from lxml import etree

# Articles are truncated to this many tokens before being sent to the LLM.
MAX_CONTEXT_TOKENS = 20000

_encoder = tiktoken.encoding_for_model("gpt-4o")

_ROMAN = r"(?:[ivxlcdm]+)"
_NUM = r"(?:\d+(?:\.\d+)*)"

# Leading section numbering, e.g. "4 ", "4.1.2 ", "IV) ", "a. "
_PREFIX_RE = re.compile(
    rf"^\s*(?:"
    rf"{_NUM}"
    rf"|{_ROMAN}"
    rf"|[a-zA-Z]"
    rf")\s*[\)\].:\-–—]*\s+",
    flags=re.IGNORECASE,
)

DISCUSSION_ALIASES = ["discussion", "discussions"]
RESULTS_ALIASES = ["result", "results"]
CONCLUSION_ALIASES = ["conclusion", "conclusions"]


def build_xml_index(directory) -> dict:
    """Build an index {filename: full_path} for fast lookup of XML files."""
    xml_index = {}
    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.endswith(".xml"):
                xml_index[fname] = os.path.join(root, fname)
    return xml_index


def find_xml_file(pmcid: str, xml_index: dict) -> Optional[str]:
    """Find XML path by PMCID using a pre-built index."""
    if not pmcid.endswith(".xml"):
        pmcid += ".xml"
    return xml_index.get(pmcid)


def get_title(file_name: str, title_file: str) -> Optional[str]:
    """Look up an article title in the metadata CSV.

    Only used when pubmed_parser fails to parse the XML.
    """
    if not title_file or not os.path.isfile(title_file):
        return None
    papers = pd.read_csv(title_file, encoding="ISO-8859-1")
    matched = papers[papers["PMCID"] == file_name[:-4]]["Title"]
    if len(matched) == 0:
        return None
    return matched.tolist()[0]


def extract_abstract(tree) -> Optional[str]:
    abstract_texts = tree.xpath("//abstract//text()")
    full_abstract = " ".join(t.strip() for t in abstract_texts if t.strip())
    if full_abstract != "":
        return full_abstract
    return None


def _norm_title(s: str) -> str:
    """Normalize section titles for robust matching."""
    if not s:
        return ""
    s = s.strip()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    s = _PREFIX_RE.sub("", s)
    return s.strip().lower()


def _sec_title_text(sec_node) -> str:
    """Get the visible title text of a <sec> (join all title text)."""
    title_parts = sec_node.xpath("./title//text()")
    return " ".join(t.strip() for t in title_parts if t and t.strip()).strip()


def _find_all_major_secs(tree, aliases):
    alias_norm = {_norm_title(a) for a in aliases}
    return [
        sec
        for sec in tree.xpath("//sec[title]")
        if _norm_title(_sec_title_text(sec)) in alias_norm
    ]


def _major_subsection_titles(tree, major_aliases) -> set:
    """Normalized titles of every <sec> under any section matching an alias.

    Returns an empty set when the article has no such section, which is how
    callers detect "this article has no Discussion".
    """
    major_secs = _find_all_major_secs(tree, major_aliases)
    if not major_secs:
        return set()

    titles = []
    for major_sec in major_secs:
        for s in [major_sec] + major_sec.xpath(".//sec[title]"):
            title = _sec_title_text(s)
            if title:
                titles.append(title)

    norm_titles = {_norm_title(t) for t in titles if _norm_title(t)}
    if not norm_titles:
        # Section exists but every title is blank; fall back to the alias.
        norm_titles = {_norm_title(major_aliases[0])}
    return norm_titles


def get_section(xml_path: str, file_name: str, title_file: str = "") -> Optional[dict]:
    """Return {section name: text} for one article, or None if nothing usable."""
    chapter_dict = {}
    tree = etree.parse(xml_path)

    try:
        xml_parse_data = pp.parse_pubmed_xml(xml_path)
        title = xml_parse_data["full_title"]
        abstract = xml_parse_data["abstract"]
    except Exception:
        print("XML parsing error")
        title = get_title(file_name, title_file)
        abstract = extract_abstract(tree)

    if title is not None and title.strip() != "":
        chapter_dict["Title"] = title.strip()
    if abstract is not None and abstract.strip() != "":
        chapter_dict["Abstract"] = abstract.strip()

    discussion_titles = _major_subsection_titles(tree, DISCUSSION_ALIASES)
    results_titles = set()
    if not discussion_titles:
        results_titles = _major_subsection_titles(tree, RESULTS_ALIASES)
    conclusion_titles = _major_subsection_titles(tree, CONCLUSION_ALIASES)

    paragraphs = pp.parse_pubmed_paragraph(xml_path, all_paragraph=True)

    def _collect(norm_titles):
        buf = []
        for p in paragraphs:
            sec = p.get("section")
            if not sec:
                continue
            if _norm_title(sec) in norm_titles:
                txt = p.get("text", "")
                if txt and txt.strip():
                    buf.append(txt.strip())
        return "\n".join(buf).strip() if buf else None

    if discussion_titles:
        text = _collect(discussion_titles)
        if text:
            chapter_dict["Discussion"] = text
    elif results_titles:
        text = _collect(results_titles)
        if text:
            chapter_dict["Results"] = text

    if conclusion_titles:
        text = _collect(conclusion_titles)
        if text:
            chapter_dict["Conclusion"] = text

    return chapter_dict if chapter_dict else None


def get_article_text(xml_path: str, file_name: str, title_file: str = "") -> str:
    """Render an article as markdown-ish sections, truncated to the token cap.

    Returns '' for articles with fewer than two usable sections.
    """
    chapter_dict = get_section(xml_path, file_name, title_file)

    if chapter_dict is None or len(chapter_dict) < 2:
        return ""

    text = ""
    for k, v in chapter_dict.items():
        text += f"## {k}:\n{v}\n"

    tokens = _encoder.encode(text)
    print(f"Abstract and Conclusion has {len(tokens)} tokens")
    if len(tokens) > MAX_CONTEXT_TOKENS:
        print(
            f"⚠️ Text too long: {len(tokens)} tokens, "
            f"truncating to {MAX_CONTEXT_TOKENS} tokens."
        )
        text = _encoder.decode(tokens[:MAX_CONTEXT_TOKENS])

    return text


def count_tokens(text: str) -> int:
    return len(_encoder.encode(text))


def read_paper_list(paper_list: str) -> list:
    """Read a newline-separated list of PMCID .xml filenames, sorted."""
    with open(paper_list, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    names.sort()
    return names


def iter_xml_dir(xml_dir) -> list:
    """Sorted list of every .xml filename under a directory tree."""
    return sorted(build_xml_index(Path(xml_dir)).keys())
