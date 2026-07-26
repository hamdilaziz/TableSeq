from __future__ import annotations

import html
import re
import unicodedata

COORD_RE = re.compile(r"<x_\d+>|<y_\d+>")
SPECIAL_RE = re.compile(r"<s>|</s>|<pad>|<unk>|<html>")
FORMAT_TAGS_RE = re.compile(
    r"</?(?:b|strong|i|em|u|sup|sub|small|font|span)(?:\s+[^>]*)?>",
    re.IGNORECASE,
)


def truncate_to_first_table(value: str) -> str:
    """Keep the first complete table when one is present."""
    start = value.find("<table")
    if start >= 0:
        value = value[start:]
    end = value.find("</table>")
    if end >= 0:
        value = value[: end + len("</table>")]
    return value


def clean_tableseq_html(value: str, remove_formatting_tags: bool = False) -> str:
    """Remove TableSeq control tokens without changing table structure."""
    value = html.unescape(str(value))
    value = unicodedata.normalize("NFKC", value)
    value = truncate_to_first_table(value)
    value = COORD_RE.sub("", value)
    value = SPECIAL_RE.sub("", value)
    if remove_formatting_tags:
        value = FORMAT_TAGS_RE.sub("", value)
    value = re.sub(r"\s+>", ">", value)
    value = re.sub(r">\s+<", "><", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()
