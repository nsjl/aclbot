# import copy
import logging
import xml.etree.ElementTree
# from copy import deepcopy
from typing import Any # , Optional, Tuple, List, Dict
from xml.etree import ElementTree
from pathlib import Path

import os.path

from intertext_graph.itgraph import IntertextDocument, Node, Edge, Etype, SpanNode
# from intertext_graph.parsers.f1000_xml_parser import F1000XMLParser
from intertext_graph.parsers.itparser import IntertextParser
# from lxml import etree

# Types used to be taken from nlpeer library, but we want to decouple from it
# from nlpeer import NTYPES, NTYPE_TITLE, NTYPE_HEADING, NTYPE_PARAGRAPH, NTYPE_ABSTRACT, NTYPE_LIST, NTYPE_LIST_ITEM, \
#     NYTPE_ELEMENT_REFERENCE, NTYPE_BIB_REFERENCE, NTYPE_HEADNOTE, NTYPE_FOOTNOTE, NTYPE_FIGURE, NTYPE_TABLE, \
#     NTYPE_FORMULA, NTYPE_MEDIA, NTYPE_BIB_ITEM

from intertext_graph.parsers.itparser import NTYPES, NTYPE_DOCUMENT_TITLE, NTYPE_SECTION_TITLE, NTYPE_PARAGRAPH, NTYPE_ABSTRACT, NTYPE_CAPTION, \
    NTYPE_LIST, NTYPE_LIST_ITEM, NYTPE_ELEMENT_REFERENCE, NTYPE_BIB_REFERENCE, NTYPE_HEADNOTE, NTYPE_FOOTNOTE, NTYPE_FIGURE, NTYPE_TABLE, \
    NTYPE_EQUATION, NTYPE_MEDIA, NTYPE_REFERENCE, NTYPE_MISC


# FIXME: This only works for ACL right now, probably need to do this using
#  semantic scholar API
VENUE_NAME_MAP = {
    'acl-long': 'Proceedings of the {index}th Annual Meeting of the Association '
                'for Computational Linguistics (Volume 1: Long Papers)',
    'acl-short': 'Proceedings of the {index}th Annual Meeting of the Association '
                    'for Computational Linguistics (Volume 2: Short Papers)',
    'acl-demo': 'Proceedings of the {index}th Annual Meeting of the Association '
                    'for Computational Linguistics: System Demonstrations',
    'acl-srw': 'Proceedings of the {index}th Annual Meeting of the Association '
                    'for Computational Linguistics: Student Research Workshop',
    'acl-tutorials': 'Proceedings of the {index}th Annual Meeting of the Association '
                    'for Computational Linguistics: Tutorial Abstracts'
}

VENUE_SHORT_NAME_MAP = {
    'acl-long': 'ACL {year}',
    'acl-short': 'ACL {year}',
    'acl-demo': 'ACL {year}',
    'acl-srw': 'ACL {year}',
    'acl-tutorials': 'ACL {year}'
}

VENUE_INDEX_MAP = {
    'acl-long': lambda year: str(int(year) - 1962),
    'acl-short': lambda year: str(int(year) - 1962),
    'acl-demo': lambda year: str(int(year) - 1962),
    'acl-srw': lambda year: str(int(year) - 1962),
    'acl-tutorials': lambda year: str(int(year) - 1962)
}


class TEIXMLParser(IntertextParser):
    """
    Parser to transform a TEI XML document into an IntertextDocument.

    Author: Nils Dycke
    Co-author: Jan-Micha Bodensohn (scaffolding and base paragraph parser)
    """

    def __init__(self, xml_file_path: str | Path):
        """
        Initialize the TEIXMLParser for a particular paper.

        :param xml_file_path: filepath of the TEI XML file (GROBID output)
        """
        super(TEIXMLParser, self).__init__(xml_file_path)
        self._xml_file_path = self._data

    def parse(self) -> IntertextDocument:
        """
        Parse the TEI XML document into an IntertextDocument.

        :return: the IntertextDocument
        """
        itg = self._parse_document()
        itg.meta.update(self._meta)
        
        return itg

    @classmethod
    def _batch_func(cls, path: Any) -> Any:
        parser = cls(path)
        try:
            intertext_document = parser.parse()
            return intertext_document
        except xml.etree.ElementTree.ParseError:
            return None


    @staticmethod
    def _parse_section_content(section, prefix):
        sub_graph = IntertextDocument([], [], prefix)
        graph_refs = []
        figures = {}

        # add artificial root node
        root = Node("root", ntype=NTYPE_DOCUMENT_TITLE)
        sub_graph.add_node(root)

        predecessor = root
        list_parent = None
        for child in section:
            # add paragraphs
            if child.tag == f"{prefix}p":
                paragraph = child

                # get text
                paragraph_text, children_ix = TEIXMLParser._flatten_xml_element_with_child_ix(paragraph)
                TEIXMLParser._check_text(paragraph_text, "paragraph")

                # get references in text
                reference_ixs = [(t, pre_ix, pre_ix + contl) for (t, pre_ix, contl) in children_ix if
                                 t.tag == f"{prefix}ref"]

                # ACL layout specific --> lists start with bulletpoints
                if paragraph_text[0] == "\u2022":
                    if list_parent is None:
                        list_parent = TEIXMLParser._add_node(sub_graph, "", NTYPE_LIST, predecessor=predecessor,
                                                             parent=root)
                        predecessor = list_parent
                    li = TEIXMLParser._add_node(sub_graph, paragraph_text, NTYPE_LIST_ITEM, predecessor=predecessor,
                                                parent=list_parent)
                    predecessor = li
                else:
                    if list_parent is not None:
                        # no list type paragraph any more: erase existing list parent
                        list_parent = None

                    p = TEIXMLParser._add_node(sub_graph, paragraph_text, NTYPE_PARAGRAPH, predecessor=predecessor,
                                               parent=root)
                    predecessor = p

                if len(reference_ixs) > 0:
                    graph_refs += [(predecessor, reference_ixs)]

            elif child.tag == f"{prefix}figure":
                fxid, figure = TEIXMLParser._parse_figure(sub_graph, child, prefix, predecessor, root)
                predecessor = figure
                figures[fxid] = figure
            elif child.tag == f"{prefix}head":
                # head -- ignoring them
                pass
            elif child.tag == f"{prefix}formula":
                # get text
                formula_text = TEIXMLParser._flatten_xml_element(child)
                TEIXMLParser._check_text(formula_text, "formula")

                p = TEIXMLParser._add_node(sub_graph, formula_text, NTYPE_EQUATION, predecessor=predecessor, parent=root)
                predecessor = p
            else:
                # add the rest
                logging.info(f"UNKNOWN TAG {section.tag} within section -- adding as flattend version.")
                paragraph_text, children_ix = TEIXMLParser._flatten_xml_element_with_child_ix(child)
                TEIXMLParser._check_text(paragraph_text, "paragraph")

                # get references in text
                reference_ixs = [(t, pre_ix, pre_ix + contl) for (t, pre_ix, contl) in children_ix if
                                 t.tag == f"{prefix}ref"]

                p = TEIXMLParser._add_node(sub_graph, paragraph_text, NTYPE_PARAGRAPH, predecessor=predecessor,
                                           parent=root)
                predecessor = p

                if len(reference_ixs) > 0:
                    graph_refs += [(predecessor, reference_ixs)]

        return sub_graph, predecessor, graph_refs, figures

    def _parse_abstract(self, doc, abstract, prefix, predecessor, article_title):
        abstract_title = self._add_node(doc, "Abstract", NTYPE_ABSTRACT, predecessor=predecessor, parent=article_title)
        predecessor = abstract_title

        if abstract is None:
            return abstract_title, predecessor

        content = ""
        merge = False
        for child in abstract:
            # get text (including any possible children)
            text = self._flatten_xml_element(child)
            self._check_text(text, "abstract paragraph")

            content += text

            if child.tag != f"{prefix}p":
                # if non-paragraph, treat as hick-up by GROBID and append to content of next node and force merge
                logging.info(f"Unexpected paragraph tag '{child.tag}' in the abstract! Treating as in-line text.")

                merge = True
                content += " "
            elif merge and predecessor.ntype == NTYPE_PARAGRAPH:
                # if a previous element was erroneously inserted, merge this paragraph with the previous one
                predecessor.content += " " + text

                merge = False
                content = ""
            else:
                # no merging required or cannot merge with previous node: simply add a new one with all content
                predecessor = self._add_node(doc, content, NTYPE_PARAGRAPH, predecessor=predecessor,
                                             parent=abstract_title)

                merge = False
                content = ""

        # if there is still content left (no paragraph follows a hick-up)
        if len(content) > 0:
            predecessor = self._add_node(doc, content.strip(), NTYPE_PARAGRAPH, predecessor=predecessor,
                                         parent=abstract_title)

        return abstract_title, predecessor

    @staticmethod
    def _parse_figure(doc, figure, prefix, predecessor, parent):
        head = figure.find(f"{prefix}head")
        label = figure.find(f"{prefix}label")
        xid = figure.get('{http://www.w3.org/XML/1998/namespace}id')
        figDesc = figure.find(f"{prefix}figDesc")

        if head is None:
            return None, None

        if "type" in figure.attrib and figure.attrib["type"] == "table":
            table = figure.find(f"{prefix}table")
            content = str(ElementTree.tostring(table))
            table_node = TEIXMLParser._add_node(doc,
                                                head.text + ' ' + content if head.text is not None else "",
                                                NTYPE_TABLE,
                                                meta={"label": label.text if label is not None else None,
                                                      "id": xid,
                                                      "caption": figDesc.text if figDesc is not None else None},
                                                predecessor=predecessor,
                                                parent=parent)
            predecessor = table_node

            table_caption = TEIXMLParser._add_node(doc,
                                                   figDesc.text if figDesc is not None and figDesc.text is not None else "",
                                                   NTYPE_CAPTION,
                                                   meta={},
                                                   predecessor=predecessor,
                                                   parent=table_node)
            predecessor = table_caption
        else:
            graphic = figure.find(f"{prefix}graphic")
            if graphic:
                content = " " + TEIXMLParser._flatten_xml_element(graphic)
            else:
                content = ""
            figure_node = TEIXMLParser._add_node(doc,
                                                 head.text + content if head.text is not None else "",
                                                 NTYPE_FIGURE,
                                                 meta={"label": label.text if label is not None else None,
                                                       "id": xid,
                                                       "caption": figDesc.text if figDesc is not None else None},
                                                 predecessor=predecessor,
                                                 parent=parent)
            predecessor = figure_node

            figure_caption = TEIXMLParser._add_node(doc,
                                                    figDesc.text if figDesc is not None and figDesc.text is not None else "",
                                                    NTYPE_CAPTION,
                                                    meta={},
                                                    predecessor=predecessor,
                                                    parent=figure_node)
            predecessor = figure_caption

        return xid, predecessor

    @staticmethod
    def _parse_bibitem(bib_item, prefix):
        xid = bib_item.get('{http://www.w3.org/XML/1998/namespace}id')
        publishing_info = bib_item.find(f"{prefix}monogr")
        paper_info = bib_item.find(f"{prefix}analytic")

        # parse publishing information
        if publishing_info:
            pub_title = publishing_info.find(f"{prefix}title")
            pub_title = pub_title.text if pub_title is not None else None

            pub = publishing_info.find(f"{prefix}imprint/{prefix}publisher")
            pub = pub.text if pub is not None else None

            pub_date = publishing_info.find(f"{prefix}imprint/{prefix}date")
            pub_date = pub_date.attrib["when"] if pub_date is not None and "when" in pub_date.attrib else None
        else:
            pub_title = None
            pub = None
            pub_date = None

        # add paper information if present
        if paper_info:
            title = paper_info.find(f"{prefix}title")
            title = title.text if title is not None else None

            authors = paper_info.findall(f"{prefix}author/{prefix}persName")

            author_names = []
            for a in authors:
                forename = a.find(f'{prefix}forename')
                surname = a.find(f'{prefix}surname')
                canonical_author = f"{forename.text if forename is not None else ''} {surname.text if surname is not None else ''}"

                author_names += [canonical_author] if len(canonical_author) > 0 else []
        else:
            title = None
            author_names = None

        return xid, title, author_names, pub_title, pub, pub_date

    @staticmethod
    def _flatten_xml_element(element):
        stack = [(element, -1, None)]

        while True:
            elem, pred, content = stack.pop(-1)

            # visiting node the first time
            if content is None:
                content = elem.text if elem.text else ""

                # revisit element after children
                stack += [(elem, pred, content)]

                # add children
                stack += reversed([(child, len(stack) - 1, None) for child in elem])
            else:
                # revisiting the node (content is set)
                suffix = elem.tail if elem.tail else ""

                if pred >= 0:
                    pre_elem, pre_pred, pre_content = stack[pred]
                    stack[pred] = (pre_elem, pre_pred, pre_content + content + suffix)
                else:
                    # terminating at parent most element
                    return content + suffix

                # don't add to stack again
        # should always terminate

    @staticmethod
    def _flatten_xml_element_with_child_ix(element):
        # xml element, predecessor ix, pred. merged ixs, parsed content
        stack = [(element, -1, [], None)]

        while True:
            elem, pred, mergedix, content = stack.pop(-1)

            # visiting node the first time
            if content is None:
                content = elem.text if elem.text else ""

                # revisit element after children
                stack += [(elem, pred, mergedix, content)]

                # add children
                stack += reversed([(child, len(stack) - 1, [], None) for child in elem])
            else:
                # revisiting the node (content is set)
                suffix = elem.tail if elem.tail else ""

                if pred >= 0:
                    pre_elem, pre_pred, pre_mergedix, pre_content = stack[pred]

                    new_mergedix = pre_mergedix + \
                                   [(elem, len(pre_content), len(content))] + \
                                   [(t, prel + len(pre_content), contl) for (t, prel, contl) in mergedix]

                    stack[pred] = (pre_elem, pre_pred, new_mergedix, pre_content + content + suffix)
                else:
                    # terminating at parent most element
                    return content + suffix, mergedix

                # don't add to stack again
        # should always terminate

    @staticmethod
    def _add_node(doc, content, ntype, meta=None, predecessor=None, parent=None):
        new_node = Node(
            content=content,
            ntype=ntype,
            meta=meta
        )
        doc.add_node(new_node)

        if parent is not None:
            parent_edge = Edge(
                src_node=parent,
                tgt_node=new_node,
                etype=Etype.PARENT
            )
            doc.add_edge(parent_edge)

        if predecessor is not None:
            next_edge = Edge(
                src_node=predecessor,
                tgt_node=new_node,
                etype=Etype.NEXT
            )
            doc.add_edge(next_edge)

        return new_node

    @staticmethod
    def _add_subtree(doc, subTree, lastSubTree, targetParent, targetPredecessor):
        # add nodes
        new_nodes = {}
        for n in subTree.nodes:
            new_n = TEIXMLParser._add_node(doc, n.content, n.ntype, n.meta)
            new_nodes[n.ix] = new_n

        # add edges
        for e in subTree.edges:
            new_e = Edge(
                src_node=new_nodes[e.src_node.ix],
                tgt_node=new_nodes[e.tgt_node.ix],
                etype=e.etype
            )
            doc.add_edge(new_e)

        # get pseudo root and replace parent edges
        new_pseudo_root = new_nodes[subTree.root.ix]
        for ce in new_pseudo_root.get_edges(Etype.PARENT, outgoing=True, incoming=False):
            new_parent = Edge(
                src_node=targetParent,
                tgt_node=ce.tgt_node,
                etype=Etype.PARENT
            )
            doc.add_edge(new_parent)
            doc.remove_edge(ce)

        # if no other nodes except root: skip the next part
        new_pseudo_next_edges = new_pseudo_root.get_edges(Etype.NEXT, outgoing=True, incoming=False)
        if len(new_pseudo_next_edges) > 0:
            new_pseudo_next = new_pseudo_next_edges[0]
            new_next = Edge(
                src_node=targetPredecessor,
                tgt_node=new_pseudo_next.tgt_node,
                etype=Etype.NEXT
            )
            doc.add_edge(new_next)
            doc.remove_edge(new_pseudo_next)

        doc.remove_node(new_pseudo_root)

        # output
        start_subtree = targetParent
        end_subtree = new_nodes[lastSubTree.ix] if len(new_pseudo_next_edges) > 0 else None

        return start_subtree, end_subtree, new_nodes

    @staticmethod
    def _parse_filename(filename: str) -> tuple[str, dict]:
        """
        Parse the file id to get publication year and venue. This assumes a file
        id from the acl-anthology dataset
        :param file_id: string of the file id
        :return:
        """
        result = {
            'year': '',
            'venue': '',
            'venue_short': '',
            'paper_index': ''
        }
        # Remove ".tei.xml" from filename to get file_id
        file_id = '.'.join(filename.split('.')[:-2])

        parts = file_id.split('.')

        if parts[0].isdigit():
            year = parts[0]
            venue_short_raw = parts[1]
            paper_index = parts[2]
            result['year'] = year
            result['paper_index'] = paper_index

            if venue_short_raw in VENUE_INDEX_MAP:
                venue_index = VENUE_INDEX_MAP[venue_short_raw](year)
                venue = VENUE_NAME_MAP[venue_short_raw].format(index=venue_index)
                result['venue'] = venue
            if venue_short_raw in VENUE_SHORT_NAME_MAP:
                venue_short_name = VENUE_SHORT_NAME_MAP[venue_short_raw].format(year=year)
                result['venue_short'] = venue_short_name

        return file_id, result


    @staticmethod
    def _extract_authors(
            tree: ElementTree,
            prefix: str
    ):
        """
        Parse the xml header to extract authors. Return as a list of strings
        :param tree: The full xml tree
        :return: The list of author names as strings
        """
        # Extract authors from header
        extracted_authors = []
        authors = tree.getroot().find(
            f"{prefix}teiHeader/{prefix}fileDesc/{prefix}sourceDesc/{prefix}biblStruct/{prefix}analytic")
        for author in authors.findall(f"{prefix}author"):
            forenames = author.findall(f"{prefix}persName/{prefix}forename")
            surname = author.find(f"{prefix}persName/{prefix}surname")
            if len(forenames) > 0 and surname is not None:
                forename = ' '.join(fn.text for fn in forenames)
                author_name = f"{forename} {surname.text}"
                extracted_authors.append(author_name)
            else:
                logging.info(f"Skipping author without forename or surname.")
        return extracted_authors


    def _parse_document(self) -> IntertextDocument:
        """
        Parse the given TEI XML document.

        :return: resulting IntertextDocument
        """
        # create intertext document
        filename = self._xml_file_path.split(os.path.sep)[-1]
        file_id, publication_meta = self._parse_filename(filename)

        itg_doc = IntertextDocument(
            nodes=[],
            edges=[],
            prefix=file_id
        )
        itg_doc.meta['id'] = file_id
        itg_doc.meta.update(publication_meta)

        # the content of the document is completely derived from the TEI XML file
        tree = ElementTree.parse(self._xml_file_path)
        prefix = "{http://www.tei-c.org/ns/1.0}"

        # Get authors
        authors = self._extract_authors(tree, prefix)
        itg_doc.meta['authors'] = authors

        # create article title as root
        title = tree.getroot().find(f"{prefix}teiHeader/{prefix}fileDesc/{prefix}titleStmt/{prefix}title").text
        title = title if title is not None else ""
        article_title_node = self._add_node(itg_doc, title, NTYPE_DOCUMENT_TITLE)
        predecessor = article_title_node

        #
        # PARSE THE ABSTRACT
        #
        abstract = tree.getroot().find(f"{prefix}teiHeader/{prefix}profileDesc/{prefix}abstract/{prefix}div")
        abstract_title_node, predecessor = self._parse_abstract(itg_doc, abstract, prefix, predecessor,
                                                                article_title_node)

        #
        # PARSE BODY
        #
        body = tree.getroot().find(f"{prefix}text/{prefix}body")

        body_refs = []
        body_figs = {}
        content_graph = []
        last_section_title = []
        for section in body.findall(f"{prefix}div"):
            content, last_elem, refs, figures = self._parse_section_content(section, prefix)

            head = section.find(f"{prefix}head")
            if head is None:
                logging.info(f"Div without a heading in {self._xml_file_path}.")

            # empty section
            if len(content.edges) == 0 and len(content.nodes) <= 1 and head is None:
                logging.info(f"Encountered empty section in {self._xml_file_path}.")

            content_graph += [(content, last_elem, refs, figures)]

            # fixme currently erroneous head nodes (e.g. with text, but without number) are discarded entirely
            if head is not None and "n" in head.attrib:
                # pop current content
                current_content, current_last, current_refs, current_figures = content_graph.pop(-1)

                # add previous contents to the predecessor if existent, else create a dummy section first
                if len(content_graph) > 0:
                    if len(last_section_title) == 0:
                        dummy_node = self._add_node(itg_doc, "", NTYPE_SECTION_TITLE, {"section": "1"},
                                                    predecessor=article_title_node, parent=article_title_node)
                        last_section_title += [dummy_node]
                        predecessor = dummy_node

                    pred_parent = last_section_title[-1]
                    for c, l, r, f in content_graph:
                        sub_root, sub_last, node_map = self._add_subtree(itg_doc, c, l, pred_parent, predecessor)
                        predecessor = sub_last if sub_last is not None else predecessor

                        mapped_refs = [(node_map[n.ix], ref) for n, ref in r]
                        body_refs += mapped_refs

                        mapped_figs = {fxid: node_map[fig.ix] for fxid, fig in f.items()}
                        body_figs.update(mapped_figs)

                    # reset content stack -- added all previous contents
                    content_graph = []

                # get section name and number
                section_name = head.text
                section_n = head.attrib["n"]
                self._check_text(section_name, "section title")

                # find parent node
                section_parent_node = None
                if len(last_section_title) == 0:
                    # is first section
                    section_parent_node = article_title_node
                else:
                    for st in last_section_title:
                        st_n = st.meta["section"]

                        if self._is_child_section_count(section_n, st_n):
                            section_parent_node = st
                            break

                    if section_parent_node is None:
                        section_parent_node = article_title_node

                # add new section title with content
                section_title_node = self._add_node(itg_doc,
                                                    section_name,
                                                    NTYPE_SECTION_TITLE,
                                                    {"section": section_n},
                                                    predecessor=predecessor,
                                                    parent=section_parent_node)
                predecessor = section_title_node
                last_section_title += [section_title_node]

                sub_root, sub_last, node_map = self._add_subtree(itg_doc, current_content, current_last,
                                                                 section_title_node,
                                                                 predecessor)
                predecessor = sub_last if sub_last is not None else predecessor

                mapped_refs = [(node_map[n.ix], r) for n, r in refs]
                body_refs += mapped_refs

                mapped_figs = {fxid: node_map[fig.ix] for fxid, fig in current_figures.items()}
                body_figs.update(mapped_figs)

        # add left-over content-graph elements
        if len(content_graph) > 0:
            if len(last_section_title) == 0:
                dummy_node = self._add_node(itg_doc, "", NTYPE_SECTION_TITLE, {"section": "1"},
                                            predecessor=article_title_node, parent=article_title_node)
                last_section_title += [dummy_node]
                predecessor = dummy_node

            pred_parent = last_section_title[-1]
            for c, l, r, f in content_graph:
                sub_root, sub_last, node_map = self._add_subtree(itg_doc, c, l, pred_parent, predecessor)
                predecessor = sub_last if sub_last is not None else predecessor

                mapped_refs = [(node_map[n.ix], ref) for n, ref in r]
                body_refs += mapped_refs

                mapped_figs = {fxid: node_map[fig.ix] for fxid, fig in f.items()}
                body_figs.update(mapped_figs)

        for figure in body.findall(f"{prefix}figure"):
            fxid, figure_node = self._parse_figure(itg_doc, figure, prefix, predecessor, article_title_node)
            predecessor = figure_node
            body_figs[fxid] = figure_node

        #
        ## PARSE BACK MATTER
        #
        back = tree.getroot().find(f"{prefix}text/{prefix}back")

        bibliography = {}
        for bib_item in back.findall(f"{prefix}div/{prefix}listBibl/{prefix}biblStruct"):
            xid, title, authors, pub_title, pub, pub_date = self._parse_bibitem(bib_item, prefix)

            bib_node = self._add_node(itg_doc, f"{', '.join(authors) if authors is not None else 'UNKNOWN'}, "
                                               f"{title}, "
                                               f"{pub_date if pub_date else ''}, "
                                               f"{pub_title if pub_title else ''}, "
                                               f"{pub if pub else ''}.",
                                      ntype=NTYPE_REFERENCE,
                                      meta={"xid": xid,
                                            "authors": authors,
                                            "title": title,
                                            "pub_date": pub_date,
                                            "pub_title": pub_title,
                                            "pub": pub})
            bibliography[xid] = bib_node

        #
        ## ADD REFERENCES
        #
        for node, refs in body_refs:
            for r in refs:
                xml_elem, start, end = r
                rtype = xml_elem.attrib["type"]

                # skip invalid references with missing target (for now)
                if "target" not in xml_elem.attrib:
                    continue

                rtarget = xml_elem.attrib["target"][1:]

                # add span node
                ref_node = SpanNode(
                    ntype=NTYPE_BIB_REFERENCE if rtype == "bibr" else NYTPE_ELEMENT_REFERENCE,
                    src_node=node,
                    start=start,
                    end=end,
                    meta={"from_xml_type": rtype, "from_xml_target": rtarget}
                )
                itg_doc.add_node(ref_node)

                # add link (where possible)
                target_node = None
                if rtype == "bibr" and rtarget in bibliography:
                    target_node = bibliography[rtarget]
                elif (rtype == "figure" or rtype == "table") and target_node in body_figs:
                    target_node = body_figs[rtarget]

                if target_node:
                    link = Edge(ref_node, target_node, etype=Etype.LINK)
                    itg_doc.add_edge(link)

        #
        ## Add document title to metadata
        #
        itg_doc.meta['title'] = itg_doc.nodes[0].content

        return itg_doc

    @staticmethod
    def _check_text(text: str, element_name: str) -> str:
        assert isinstance(text, str), f"{element_name} is not a string, but a {type(text)}!"
        return text

    @staticmethod
    def _compare_section_counts(cntA: str, cntB: str) -> int:
        nAs = cntA.split(".")
        nBs = cntB.split(".")

        for i, nA in enumerate(nAs):
            if len(nBs) <= i:
                return 1  # b higher level than a (a > b)
            nB = nBs[i]
            if int(nA) != int(nB):
                return -1 if int(nA) < int(nB) else 1  # a earlier than b (a < b)

        return 0 if len(nAs) == len(nBs) else -1  # a higher level than b (a < b), else equal

    @staticmethod
    def _is_child_section_count(cntA: str, cntB: str) -> bool:
        return cntA.startswith(cntB)