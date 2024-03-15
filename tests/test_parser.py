import unittest

from tokenizecode.parser import to_tensortree, FullTraversal, CodeParser, TreeSitterParser

import tensortree
from .codesamples import SAMPLE_CODE


def to_tensortree_without_descendants(nodes: list) -> tensortree.TensorTree:
    """ Uses tensortrees descendant conversion"""
    node_data, parents, meta = [], [], []

    for node in nodes:
        node_data.append(node.text)
        parents.append(node.parent_id)
        meta.append(node.span)
    return tensortree.tree(parents, node_data)



class TestCodeParser(unittest.TestCase):

    def assertTreeEqual(self, tree1: tensortree.TensorTree, tree2: tensortree.TensorTree, msg = ...) -> None:
        import torch
        self.assertListEqual(
            tree1.descendants.tolist(),
            tree2.descendants.tolist(),
            msg
        )
        self.assertListEqual(
            tree1.parents.tolist(),
            tree2.parents.tolist(),
            msg
        )

        n1 = tree1.node_data
        n2 = tree2.node_data

        if isinstance(n1, torch.Tensor):
            n1 = n1.tolist()
        if isinstance(n2, torch.Tensor):
            n2 = n1.tolist()

        self.assertListEqual(n1, n2, msg)

    def parse(self, code: str, language: str) -> tensortree.TensorTree:
        parser = TreeSitterParser()
        traverse = FullTraversal()

        tree = parser.parse(code, language)
        nodes = traverse(code, tree)

        tree, positions = to_tensortree(nodes)

        new_code = "".join(tree.get_node_data(i) for i in tree.leaf_indices())
        self.assertSequenceEqual(new_code, code, "Original code should be able to be reconstructed")
        return tree

    def parse_without_descendants(self, code: str, language: str) -> tensortree.TensorTree:
        parser = TreeSitterParser()
        traverse = FullTraversal()

        tree = parser.parse(code, language)
        nodes = traverse(code, tree)

        tree = to_tensortree_without_descendants(nodes)

        new_code = "".join(tree.get_node_data(i) for i in tree.leaf_indices())
        self.assertSequenceEqual(new_code, code, "Original code should be able to be reconstructed")
        return tree

    def test_descendants_this_file(self):
        with open(__file__, "rt") as f:
            code = f.read()

        tree1 = self.parse(code, "python")
        tree2 = self.parse_without_descendants(code, "python")
        tree1.pprint()
        tree2.pprint()
        d1 = tree1.descendants.tolist()
        d2 = tree2.descendants.tolist()
        self.assertListEqual(d1, d2)

    def test_descendants(self):
        for language, code in SAMPLE_CODE.items():
            with self.subTest(f"Parsing {language}") as test:

                tree1 = self.parse(code, language)
                tree2 = self.parse_without_descendants(code, language)
                tree1.pprint()
                tree2.pprint()
                self.assertTreeEqual(tree1, tree2)

                self.assertNotIn("[ERROR]", tree1.node_data)

    def test_descendants_wrong_language(self):
        ruby_code = SAMPLE_CODE["ruby"]
        tree1 = self.parse(ruby_code, "java")
        tree2 = self.parse_without_descendants(ruby_code, "java")
        tree1.pprint()
        tree2.pprint()
        self.assertTreeEqual(tree1, tree2)
        self.assertIn("[ERROR]", tree1.node_data)

    def test_parsing_all(self):
        parser = CodeParser()
        from tensortree.utils import replace_whitespace
        for language, code in SAMPLE_CODE.items():
            with self.subTest(f"Parsing {language}") as t:
                output = parser.parse(code, language)
                tree = output.tree
                spans = output.positions

                decoded_code = parser.unparse(tree)
                tree.pprint(node_renderer=lambda node_idx, name: f"{replace_whitespace(name)} [({spans[node_idx].start_point.row};{spans[node_idx].start_point.column})->({spans[node_idx].end_point.row};{spans[node_idx].end_point.column})]")

                if code != decoded_code:
                    print('#' * 100)
                    print(code)
                    print()
                    print(decoded_code)
                    # tree.pprint()
                    print('#' * 100)
                self.assertSequenceEqual(code, decoded_code)




if __name__ == '__main__':
    unittest.main()
