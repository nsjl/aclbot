from abc import ABC, abstractmethod
from multiprocessing import Pool
from pathlib import Path
import pkg_resources
import re
from typing import Any, Dict, Iterator, List

from tqdm import tqdm

from intertext_graph import Edge, Etype, Node, SpanNode, IntertextDocument
from intertext_graph.parsers.utils import chunksize, num_processes

NTYPE_DOCUMENT_TITLE = "document-title"
NTYPE_SECTION_TITLE = "section-title"
NTYPE_PARAGRAPH = "paragraph"
NTYPE_ABSTRACT = "section-title"
NTYPE_CAPTION = "caption"
NTYPE_LIST = "list"
NTYPE_LIST_ITEM = "list_item"
NYTPE_ELEMENT_REFERENCE = "elem_reference"
NTYPE_BIB_REFERENCE = "bib_reference"
NTYPE_HEADNOTE = "headnote"
NTYPE_FOOTNOTE = "footnote"
NTYPE_FIGURE = "figure"
NTYPE_TABLE = "table"
NTYPE_EQUATION = "equation"
NTYPE_MEDIA = "media"
NTYPE_REFERENCE = "reference"
NTYPE_MISC = "misc"

NTYPES = []
NTYPES += [NTYPE_DOCUMENT_TITLE]
NTYPES += [NTYPE_SECTION_TITLE]
NTYPES += [NTYPE_PARAGRAPH]
NTYPES += [NTYPE_ABSTRACT]
NTYPES += [NTYPE_CAPTION]
NTYPES += [NTYPE_LIST]
NTYPES += [NTYPE_LIST_ITEM]
NTYPES += [NYTPE_ELEMENT_REFERENCE]
NTYPES += [NTYPE_BIB_REFERENCE]
NTYPES += [NTYPE_HEADNOTE]
NTYPES += [NTYPE_FOOTNOTE]
NTYPES += [NTYPE_FIGURE]
NTYPES += [NTYPE_TABLE]
NTYPES += [NTYPE_EQUATION]
NTYPES += [NTYPE_MEDIA]
NTYPES += [NTYPE_REFERENCE]
NTYPES += [NTYPE_MISC]


class IntertextParser(ABC):

    def __init__(self, data: Path | str | Any) -> None:
        if isinstance(data, Path):
            self._data = str(data.resolve())
        elif isinstance(data, str):
            self._data = data if data.startswith('http') else str(Path(data).resolve())
        else:
            self._data = str(data)
        try:
            version = pkg_resources.require('intertext-graph')[0].version
        except pkg_resources.DistributionNotFound:
            version = 'N/A'
        self._meta = {
            'parser': type(self).__name__,
            'intertext-graph': version
        }

    @classmethod
    def _make_node(cls, content: str, ntype: str, meta: Dict[str, Any] = None) -> Node:
        """Adds a created by meta attribute and passes the arguments to the Node constructor."""
        if not meta:
            meta = {}
        meta['created_by'] = cls.__name__
        return Node(content, ntype, meta)

    @classmethod
    def _make_edge(cls, src_node: Node, tgt_node: Node, etype: Etype, meta: Dict[str, Any] = None) -> Edge:
        """Adds a created by meta attribute and passes the arguments to the Edge constructor."""
        if not meta:
            meta = {}
        meta['created_by'] = cls.__name__
        return Edge(src_node, tgt_node, etype, meta)

    @classmethod
    def _make_span_node(
            cls,
            ntype: str,
            src_node: Node,
            start: int = 0,
            end: int = None,
            meta: Dict[str, Any] = None,
            label: Dict = None) -> Node:
        """Adds a created by meta attribute and passes the arguments to the SpanNode constructor."""
        if not meta:
            meta = {}
        meta['created_by'] = cls.__name__
        return SpanNode(ntype, src_node, start, end, meta, label)

    @classmethod
    def _parse_whitespace(cls, text: str) -> str:
        # This replaces line breaks, tabs, spaces, and non-breaking spaces (both as a string and unicode character)
        text = re.sub(r'[\n\t   ]+', ' ', text)
        return text.strip()
    
    @abstractmethod
    def parse(self) -> IntertextDocument:
        raise NotImplementedError

    @classmethod
    def _batch_func(cls, path: Any) -> Any:
        raise NotImplementedError

    @classmethod
    def batch_parse(cls, files: List[Any]) -> Iterator[IntertextDocument]:
        """Parse a list of files using multiprocessing."""
        total = len(files)
        with Pool(processes=num_processes()) as pool:
            for parsed in tqdm(
                    pool.imap_unordered(cls._batch_func, files, chunksize(total)),
                    total=total,
                    desc='parsing documents'):
                if parsed:
                    yield parsed
            pool.close()
            pool.join()
