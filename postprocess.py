# postprocess_safe.py

import re


SPECIAL_RE = re.compile(r"<s>|</s>|<pad>|<unk>|<html>")
COORD_RE = re.compile(r"<x_\d+>|<y_\d+>")

# Your dataset uses td, not th.
CORE_TAGS = {"table", "thead", "tbody", "tfoot", "tr", "td"}
TAG_RE = re.compile(r"</?(table|thead|tbody|tfoot|tr|td)\b(?:\s[^>]*)?>", re.IGNORECASE)


def remove_special_tokens(s: str) -> str:
    s = str(s)
    s = SPECIAL_RE.sub("", s)
    s = COORD_RE.sub("", s)
    return s


def truncate_after_first_table(s: str) -> str:
    end = s.lower().find("</table>")
    if end >= 0:
        return s[: end + len("</table>")]
    return s


def count_open_close(s: str, tag: str) -> tuple[int, int]:
    open_count = len(re.findall(fr"<{tag}\b(?:\s[^>]*)?>", s, flags=re.IGNORECASE))
    close_count = len(re.findall(fr"</{tag}\s*>", s, flags=re.IGNORECASE))
    return open_count, close_count


def is_core_balanced(s: str) -> bool:
    for tag in CORE_TAGS:
        open_count, close_count = count_open_close(s, tag)
        if open_count != close_count:
            return False
    return True


def is_complete_table(s: str) -> bool:
    low = s.lower()
    return "<table" in low and "</table>" in low and is_core_balanced(s)


def repair_unclosed_core_tags(s: str) -> str:
    """
    Streaming repair.

    This only inserts missing closing tags for core table tags.
    It does not change colspan/rowspan and does not invent rows/cells.
    """
    out = []
    stack = []
    last = 0

    def close_tag(tag: str) -> None:
        out.append(f"</{tag}>")

    def close_until_allowed(allowed_top_tags: set[str]) -> None:
        while stack and stack[-1] not in allowed_top_tags:
            close_tag(stack.pop())

    for match in TAG_RE.finditer(s):
        out.append(s[last:match.start()])

        full_tag = match.group(0)
        tag = match.group(1).lower()
        is_closing = full_tag.startswith("</")

        if not is_closing:
            if tag == "td":
                # A new cell starts while the previous cell is still open.
                if stack and stack[-1] == "td":
                    close_tag(stack.pop())

            elif tag == "tr":
                # A new row starts; close unfinished cell and previous row.
                close_until_allowed({"table", "thead", "tbody", "tfoot"})
                if stack and stack[-1] == "tr":
                    close_tag(stack.pop())

            elif tag in {"thead", "tbody", "tfoot"}:
                # A new section starts; close unfinished row/cell and previous section.
                close_until_allowed({"table"})
                if stack and stack[-1] in {"thead", "tbody", "tfoot"}:
                    close_tag(stack.pop())

            elif tag == "table":
                # If a second table starts before the first is closed, close the first.
                if "table" in stack:
                    while stack:
                        close_tag(stack.pop())

            out.append(full_tag)
            stack.append(tag)

        else:
            if tag in stack:
                # Close any tags opened after the target tag.
                while stack and stack[-1] != tag:
                    close_tag(stack.pop())

                if stack and stack[-1] == tag:
                    out.append(full_tag)
                    stack.pop()
            else:
                # Ignore unmatched closing tags.
                pass

        last = match.end()

    out.append(s[last:])

    while stack:
        close_tag(stack.pop())

    return "".join(out)


def postprocess_tableseq_malformed_only(
    s: str,
    gen_len: int | None = None,
    max_length: int = 2048,
) -> str:
    """
    Conservative post-processing.

    It returns the original prediction unchanged unless the table is clearly
    incomplete or unbalanced.
    """
    s = remove_special_tokens(s)
    s = truncate_after_first_table(s)

    needs_repair = (
        not is_complete_table(s)
        or not is_core_balanced(s)
        or (gen_len is not None and gen_len >= max_length and "</table>" not in s.lower())
    )

    if not needs_repair:
        return s

    repaired = repair_unclosed_core_tags(s)
    repaired = truncate_after_first_table(repaired)

    return repaired