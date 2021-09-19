import logging
from typing import Union, Optional, Iterable

import shutil
from tree_sitter import Language, Parser

from tokenizecode import get_project_root

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tree_sitter

from tensortree import TensorTree
import tensortree
from tokenizecode.utils import replace_whitespace, restore_whitespace

log = logging.getLogger(__name__)


def download_grammar(language: str, directory: Path) -> Path:
    import requests, zipfile, io

    log.info(f"Cloning grammar for language {language} to {directory}.")
    url = f"https://github.com/tree-sitter/tree-sitter-{language}/archive/refs/heads/master.zip"
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(directory.absolute())

    path = directory / f"tree-sitter-{language}-master"
    new_path = path.with_name(path.name.replace('-master', ''))

    assert path.exists(), "this directory should exist"

    shutil.move(
        str(path.absolute()),
        str(new_path.absolute()),
    )
    return new_path


class TreeSitterParser:
    """ A treesitter code parser that magically downloads and initializes itself"""

    LANGUAGES: dict[str, Language] = {}

    def __init__(self, libs_dir: Optional[Path] = None):
        if libs_dir is None:
            self.libs_dir = get_project_root() / 'libs'

        self.build_path = (self.libs_dir / 'langs.so').absolute()
        self.parser = Parser()
        self.language = None

    def _setup_grammar(self, language: str):

        self.libs_dir.mkdir(parents=True, exist_ok=True)
        downloaded_langs = {
            d.name.replace('tree-sitter-', '').replace('-master', ''): d for d in self.libs_dir.iterdir()
            if d.is_dir() if d.name.startswith('tree-sitter-')
        }
        print(downloaded_langs)
        if language not in downloaded_langs:
            downloaded_langs[language] = download_grammar(language, self.libs_dir)
            self.build_path.unlink(missing_ok=True)

        Language.build_library(
            str(self.build_path.absolute()),
            [str(dir_.absolute()) for dir_ in downloaded_langs.values()]
        )

        self.LANGUAGES = {
            lang: Language(str(self.build_path.absolute()), lang)
            for lang in downloaded_langs
        }
        print(self.LANGUAGES)
    def _set_language(self, language: str):
        if language not in self.LANGUAGES:
            try:
                self._setup_grammar(language)
            except AttributeError as e:
                # self.reset(reload_all=True)
                self._setup_grammar(language)
                # raise e

        self.parser.set_language(self.LANGUAGES[language])
        self.language = language

    def reset(self, reload_all: bool = False):
        if reload_all and self.libs_dir.exists():
            import shutil
            shutil.rmtree(self.libs_dir)

        self.build_path.unlink(missing_ok=True)
        self.LANGUAGES = {}

    def parse(self, code: Union[str, bytes], language: str = None):
        if self.language is None and language is None:
            raise ValueError("provide language")

        if language != self.language:
            self._set_language(language)

        if isinstance(code, str):
            code = code.encode("utf-8")

        return self.parser.parse(code)


@dataclass
class Point:
    row: int
    column: int

    def __eq__(self, other):
        if not isinstance(other, Point):
            return False

        return self.row == other.row and self.column == other.column

    def __lt__(self, other):
        if not isinstance(other, Point):
            return False

        if self.row < other.row:
            return True
        elif self.row == other.row and self.column < other.column:
            return True

        return  False

    def __le__(self, other):
        if not isinstance(other, Point):
            return False

        if self.row < other.row:
            return True
        elif self.row == other.row and self.column <= other.column:
            return True

        return False


@dataclass
class NodeSpan:
    start_byte: int
    end_byte: int
    start_point: Point
    end_point: Point


@dataclass
class _Node:
    """ Helper during parsing. """
    id_: int
    text: str
    parent_id: int
    descendants: int

    span: NodeSpan



class TreeTraversal:
    """ Implement a call method, that takes code and a treesitter and produces an iterable of nodes."""
    def __init__(self):
        self.errors = 0  # number of errors in last traversal

    def __call__(self, code: str, tree: tree_sitter.Tree) -> Iterable[_Node]:
        raise NotImplementedError


class FullTraversal(TreeTraversal):

    def __call__(self, code: str, tree: tree_sitter.Tree) -> Iterable[_Node]:
        self.errors = 0  # will be incremented in traverse

        nodes = list(self.traverse_tree(code, tree))

        if self.errors:
            import warnings
            warnings.warn(f"Found {self.errors} errors while parsing. Is the parser set to the correct language?")

        return nodes

    def traverse_tree(self, code: str, tree: tree_sitter.Tree):
        cursor = tree.walk()

        reached_root = False

        last_leaf: Optional[_Node] = None
        last_leaf_checked = None

        node_id = -1
        open_nodes: list[_Node] = []
        root: Optional[_Node] = None

        def text_between(node_start_byte):
            nonlocal last_leaf_checked, last_leaf

            if last_leaf is None:
                return

            if last_leaf_checked is not last_leaf:
                text = code[last_leaf.span.end_byte:node_start_byte]
                last_leaf_checked = last_leaf
                return text

        def add_node(text, span):
            nonlocal node_id, root

            node_id += 1
            if open_nodes:
                parent_id = open_nodes[-1].id_
                open_nodes[-1].descendants += 1
            elif last_leaf:
                # at the end
                parent_id = 0  # root
            else:
                parent_id = -1

            if text == "[ERROR]":
                self.errors += 1

            node = _Node(id_=node_id, text=text, parent_id=parent_id, descendants=0, span=span)

            if parent_id == -1:
                root = node

            return node

        def to_node(node, read_text: bool):
            nonlocal last_leaf

            span = NodeSpan(start_byte=node.start_byte, end_byte=node.end_byte,
                            start_point=Point(*node.start_point), end_point=Point(*node.end_point))
            text = code[span.start_byte:span.end_byte] if read_text else f"[{node.type}]"
            node = add_node(text, span)

            if read_text:
                last_leaf = node

            return node

        while not reached_root:
            code_between = text_between(cursor.node.start_byte)
            if code_between:
                code_span = NodeSpan(
                    start_byte=last_leaf.span.end_byte, end_byte=cursor.node.start_byte,
                    start_point=last_leaf.span.end_point, end_point=Point(*cursor.node.start_point)
                )
                yield add_node(code_between, code_span)

            if cursor.node.is_named and not cursor.node.children:
                node = to_node(cursor.node, read_text=False)
                open_nodes.append(node)
                yield node

            node = to_node(cursor.node, read_text=not bool(cursor.node.children))
            yield node

            if cursor.node.is_named and not cursor.node.children:
                if len(open_nodes) > 1:
                    open_nodes[-2].descendants += open_nodes[-1].descendants
                open_nodes = open_nodes[:-1]

            if cursor.goto_first_child():
                open_nodes.append(node)
                continue

            if cursor.goto_next_sibling():
                continue

            retracing = True
            while retracing:
                if not cursor.goto_parent():
                    retracing = False
                    reached_root = True
                else:
                    if len(open_nodes) > 1:
                        open_nodes[-2].descendants += open_nodes[-1].descendants

                    open_nodes = open_nodes[:-1]

                if cursor.goto_next_sibling():
                    retracing = False

        code_between = text_between(cursor.node.end_byte)
        if code_between:
            code_between_span = NodeSpan(
                start_byte=last_leaf.span.end_byte, end_byte=cursor.node.end_byte,
                start_point=last_leaf.span.end_point, end_point=Point(*cursor.node.end_point)
            )
            yield add_node(code_between, code_between_span)
            root.descendants += 1


class SplitLinesTraversal(TreeTraversal):
    def __call__(self, code: str, tree: tree_sitter.Tree) -> Iterable[_Node]:
        self.errors = 0  # will be incremented in traverse

        nodes = list(self.traverse_tree_and_splitlines(code, tree))

        if self.errors:
            import warnings
            warnings.warn(f"Found {self.errors} errors while parsing. Is the parser set to the correct language?")

        return nodes

    @staticmethod
    def traverse_tree_and_splitlines(code: str, tree: tree_sitter.Tree):
        cursor = tree.walk()

        reached_root = False

        last_leaf: Optional[_Node] = None
        last_leaf_checked = None

        node_id = -1
        open_nodes: list[_Node] = []
        root: Optional[_Node] = None

        def text_between(node_start_byte):
            nonlocal last_leaf_checked, last_leaf

            if last_leaf is None:
                return

            if last_leaf_checked is not last_leaf:
                text = code[last_leaf.span.end_byte:node_start_byte]
                last_leaf_checked = last_leaf
                return text

        def add_node(text, span):
            nonlocal node_id, root

            node_id += 1
            if open_nodes:
                parent_id = open_nodes[-1].id_
                open_nodes[-1].descendants += 1
            elif last_leaf:
                # at the end
                parent_id = 0  # root
            else:
                parent_id = -1

            node = _Node(id_=node_id, text=text, parent_id=parent_id, descendants=0, span=span)

            if parent_id == -1:
                root = node

            return node

        def to_node(node, read_text: bool):
            nonlocal last_leaf

            span = NodeSpan(start_byte=node.start_byte, end_byte=node.end_byte,
                            start_point=Point(*node.start_point), end_point=Point(*node.end_point))
            text = code[span.start_byte:span.end_byte] if read_text else f"[{node.type}]"
            node = add_node(text, span)

            if read_text:
                last_leaf = node

            return node

        while not reached_root:
            code_between = text_between(cursor.node.start_byte)

            if code_between:
                code_span = NodeSpan(
                    start_byte=last_leaf.span.end_byte, end_byte=cursor.node.start_byte,
                    start_point=last_leaf.span.end_point, end_point=Point(*cursor.node.start_point)
                )

                yield add_node(code_between, code_span)

            if cursor.node.is_named and not cursor.node.children:
                node = to_node(cursor.node, read_text=False)
                open_nodes.append(node)
                yield node

            node = to_node(cursor.node, read_text=not bool(cursor.node.children))
            yield node

            if cursor.node.is_named and not cursor.node.children:
                if len(open_nodes) > 1:
                    open_nodes[-2].descendants += open_nodes[-1].descendants
                open_nodes = open_nodes[:-1]

            if cursor.goto_first_child():
                open_nodes.append(node)
                continue

            if cursor.goto_next_sibling():
                continue

            retracing = True
            while retracing:
                if not cursor.goto_parent():
                    retracing = False
                    reached_root = True
                else:
                    if len(open_nodes) > 1:
                        open_nodes[-2].descendants += open_nodes[-1].descendants

                    open_nodes = open_nodes[:-1]

                if cursor.goto_next_sibling():
                    retracing = False

        code_between = text_between(cursor.node.end_byte)
        if code_between:
            code_between_span = NodeSpan(
                start_byte=last_leaf.span.end_byte, end_byte=cursor.node.end_byte,
                start_point=last_leaf.span.end_point, end_point=Point(*cursor.node.end_point)
            )
            yield add_node(code_between, code_between_span)
            root.descendants += 1


def to_tensortree(nodes: list[_Node]) -> tuple[TensorTree, list[NodeSpan]]:
    # correct descendants value will be set after iteration (!)
    node_data, parents, descendants, positions = [], [], [], []

    for node in nodes:
        node_data.append(replace_whitespace(node.text))
        parents.append(node.parent_id)
        descendants.append(node.descendants)
        positions.append(node.span)

    return tensortree.tree(parents, node_data, descendants), positions


class CodeParser:
    """
    Computes the full syntax tree with all tokens for a piece of code.
    """
    def __init__(self, traversal: Optional[TreeTraversal] = None):
        self.parser = TreeSitterParser()
        self.traverse = traversal if traversal is not None else FullTraversal()

    def parse(self, code: str = None, lang: str = "java",
              output_positions: bool = False, output_errors: bool = False) -> Union[TensorTree, tuple[TensorTree, int], tuple[TensorTree, list[NodeSpan]], tuple[TensorTree, list[NodeSpan], int]]:
        ts_tree = self.parser.parse(code, lang)
        nodes = list(self.traverse(code, ts_tree))
        num_errors = self.traverse.errors
        tree, positions = to_tensortree(nodes)

        res = (tree, positions) if output_positions else tree
        return (res, num_errors) if output_errors else res

    @staticmethod
    def unparse(tree: TensorTree) -> str:
        return "".join(restore_whitespace(tree.get_node_data(i)) for i in tree.leaf_indices())

    def pprint(self, tree: TensorTree) -> None:
        print(self.unparse(tree))
