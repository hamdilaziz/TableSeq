from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

try:
    from apted import APTED, Config
except ImportError:  # Keep CLI/help and non-TEDS utilities usable in partial environments.
    APTED = None

    class Config:  # type: ignore[no-redef]
        pass

from lxml import html


try:
    from rapidfuzz.distance import Levenshtein as RFLevenshtein

    def levenshtein_distance(a, b) -> int:
        return RFLevenshtein.distance(a, b)

except Exception:
    def levenshtein_distance(a, b) -> int:
        if a == b:
            return 0

        if len(a) < len(b):
            a, b = b, a

        previous = list(range(len(b) + 1))

        for i, ca in enumerate(a, start=1):
            current = [i]
            for j, cb in enumerate(b, start=1):
                insert = current[j - 1] + 1
                delete = previous[j] + 1
                replace = previous[j - 1] + (ca != cb)
                current.append(min(insert, delete, replace))
            previous = current

        return previous[-1]


COORD_RE = re.compile(r"<x_\d+>|<y_\d+>")
SPECIAL_RE = re.compile(r"<s>|</s>|<pad>|<unk>|<html>")
SPACE_RE = re.compile(r"\s+")
TAG_RE = re.compile(r"</?([a-zA-Z0-9]+)(?:\s[^>]*)?>")

MAIN_TAGS = {"table", "thead", "tbody", "tfoot", "tr", "td", "th"}
CELL_TAGS = {"td", "th"}


@dataclass
class TableTree:
    tag: str
    colspan: int = 1
    rowspan: int = 1
    content: Optional[str] = None
    children: Optional[list["TableTree"]] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


class FastTEDSConfig(Config):
    def __init__(self, structure_only: bool = False):
        super().__init__()
        self.structure_only = structure_only

    def rename(self, node1: TableTree, node2: TableTree) -> float:
        if node1.tag != node2.tag:
            return 1.0

        if node1.tag in CELL_TAGS:
            if node1.colspan != node2.colspan:
                return 1.0
            if node1.rowspan != node2.rowspan:
                return 1.0

            if not self.structure_only:
                c1 = node1.content or ""
                c2 = node2.content or ""

                if c1 or c2:
                    max_len = max(len(c1), len(c2), 1)
                    return levenshtein_distance(c1, c2) / max_len

        return 0.0


def clean_raw_html(raw: str) -> str:
    if raw is None:
        return ""

    raw = COORD_RE.sub("", raw)
    raw = SPECIAL_RE.sub("", raw)
    raw = SPACE_RE.sub(" ", raw)
    raw = raw.strip()

    return raw


def ensure_table(raw: str) -> str:
    raw = clean_raw_html(raw)

    if "<table" in raw:
        return raw

    return f"<table>{raw}</table>"


def parse_table(raw: str):
    raw = ensure_table(raw)

    try:
        root = html.fromstring(raw)
    except Exception:
        try:
            root = html.fromstring(f"<table>{raw}</table>")
        except Exception:
            return None

    if root.tag == "table":
        return root

    tables = root.xpath(".//table")
    if tables:
        return tables[0]

    return None


def normalize_text(text: str) -> str:
    text = text or ""
    text = SPACE_RE.sub(" ", text).strip()
    text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)
    text = re.sub(r"(?<=\d)\s+([,.;:%])", r"\1", text)
    text = re.sub(r"([,.;:])\s+(?=\d)", r"\1", text)
    return text


def safe_int(value, default: int = 1) -> int:
    try:
        value = int(value)
        return value if value > 0 else default
    except Exception:
        return default


def convert_to_table_tree(node, structure_only: bool = False) -> Optional[TableTree]:
    tag = str(node.tag).lower()

    if tag not in MAIN_TAGS:
        children = []
        for child in node:
            converted = convert_to_table_tree(child, structure_only=structure_only)
            if converted is not None:
                children.append(converted)

        if len(children) == 1:
            return children[0]

        if len(children) > 1:
            wrapper = TableTree("table")
            wrapper.children.extend(children)
            return wrapper

        return None

    if tag in CELL_TAGS:
        colspan = safe_int(node.attrib.get("colspan", "1"))
        rowspan = safe_int(node.attrib.get("rowspan", "1"))

        content = None
        if not structure_only:
            content = normalize_text(node.text_content())

        return TableTree(
            tag=tag,
            colspan=colspan,
            rowspan=rowspan,
            content=content,
            children=[],
        )

    tree = TableTree(tag=tag)

    for child in node:
        converted = convert_to_table_tree(child, structure_only=structure_only)
        if converted is not None:
            tree.children.append(converted)

    return tree


def count_nodes(tree: Optional[TableTree]) -> int:
    if tree is None:
        return 0

    return 1 + sum(count_nodes(child) for child in tree.children)


def tree_similarity(
    gt_tree: Optional[TableTree],
    pred_tree: Optional[TableTree],
    gt_nodes: int,
    pred_nodes: int,
    structure_only: bool,
) -> float:
    if gt_tree is None or pred_tree is None:
        return 0.0

    if APTED is None:
        raise ImportError(
            "The 'apted' package is required for TEDS/S-TEDS metrics. "
            "Install the project dependencies with `pip install -e .`."
        )

    n_nodes = max(gt_nodes, pred_nodes, 1)
    dist = APTED(pred_tree, gt_tree, FastTEDSConfig(structure_only=structure_only)).compute_edit_distance()
    score = 1.0 - float(dist) / float(n_nodes)

    if score < 0.0:
        return 0.0

    if score > 1.0:
        return 1.0

    return score


def structure_tokens(raw: str) -> list[str]:
    raw = ensure_table(raw)
    tokens = []

    for match in TAG_RE.finditer(raw):
        full = match.group(0)
        tag = match.group(1).lower()

        if tag not in MAIN_TAGS:
            continue

        closing = full.startswith("</")

        if closing:
            tokens.append(f"</{tag}>")
            continue

        if tag in CELL_TAGS:
            colspan_match = re.search(r'colspan=["\']?(\d+)', full)
            rowspan_match = re.search(r'rowspan=["\']?(\d+)', full)

            colspan = colspan_match.group(1) if colspan_match else "1"
            rowspan = rowspan_match.group(1) if rowspan_match else "1"

            tokens.append(f"<{tag}:c{colspan}:r{rowspan}>")
        else:
            tokens.append(f"<{tag}>")

    return tokens


def sequence_similarity(gt_tokens: list[str], pred_tokens: list[str]) -> float:
    if not gt_tokens and not pred_tokens:
        return 1.0

    if not gt_tokens or not pred_tokens:
        return 0.0

    max_len = max(len(gt_tokens), len(pred_tokens), 1)
    dist = levenshtein_distance(gt_tokens, pred_tokens)
    score = 1.0 - float(dist) / float(max_len)

    return max(0.0, min(1.0, score))


class FastTableMetrics:
    """
    Fast TEDS/S-TEDS evaluator with ground-truth caching.

    Use `key=name` during validation so the ground-truth tree is parsed only once.
    """

    def __init__(self):
        self._gt_teds_cache = {}
        self._gt_steds_cache = {}
        self._gt_seq_cache = {}

    def _get_gt_tree(self, gt_raw: str, key: Optional[str], structure_only: bool):
        cache = self._gt_steds_cache if structure_only else self._gt_teds_cache
        cache_key = key if key is not None else gt_raw

        if cache_key in cache:
            return cache[cache_key]

        table = parse_table(gt_raw)
        tree = convert_to_table_tree(table, structure_only=structure_only) if table is not None else None
        n_nodes = count_nodes(tree)

        cache[cache_key] = (tree, n_nodes)
        return tree, n_nodes

    def score_steds(self, gt_raw: str, pred_raw: str, key: Optional[str] = None) -> float:
        gt_tree, gt_nodes = self._get_gt_tree(gt_raw, key=key, structure_only=True)

        pred_table = parse_table(pred_raw)
        pred_tree = convert_to_table_tree(pred_table, structure_only=True) if pred_table is not None else None
        pred_nodes = count_nodes(pred_tree)

        return tree_similarity(
            gt_tree=gt_tree,
            pred_tree=pred_tree,
            gt_nodes=gt_nodes,
            pred_nodes=pred_nodes,
            structure_only=True,
        )

    def score_teds(self, gt_raw: str, pred_raw: str, key: Optional[str] = None) -> float:
        gt_tree, gt_nodes = self._get_gt_tree(gt_raw, key=key, structure_only=False)

        pred_table = parse_table(pred_raw)
        pred_tree = convert_to_table_tree(pred_table, structure_only=False) if pred_table is not None else None
        pred_nodes = count_nodes(pred_tree)

        return tree_similarity(
            gt_tree=gt_tree,
            pred_tree=pred_tree,
            gt_nodes=gt_nodes,
            pred_nodes=pred_nodes,
            structure_only=False,
        )

    def score_structure_sequence(
        self,
        gt_raw: str,
        pred_raw: str,
        key: Optional[str] = None,
    ) -> float:
        cache_key = key if key is not None else gt_raw

        if cache_key in self._gt_seq_cache:
            gt_tokens = self._gt_seq_cache[cache_key]
        else:
            gt_tokens = structure_tokens(gt_raw)
            self._gt_seq_cache[cache_key] = gt_tokens

        pred_tokens = structure_tokens(pred_raw)
        return sequence_similarity(gt_tokens, pred_tokens)

    def clear_cache(self) -> None:
        self._gt_teds_cache.clear()
        self._gt_steds_cache.clear()
        self._gt_seq_cache.clear()
