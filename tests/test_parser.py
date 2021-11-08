import unittest

from tokenizecode import TreeSitterParser
from tokenizecode.parser import to_tensortree, FullTraversal, CodeParser

import tensortree
from codesamples import SAMPLE_CODE


def to_tensortree_without_descendants(nodes: list) -> tensortree.TensorTree:
    """ Uses tensortrees descendant conversion"""
    node_data, parents, meta = [], [], []

    for node in nodes:
        node_data.append(node.text)
        parents.append(node.parent_id)
        meta.append(node.span)
    return tensortree.tree(parents, node_data)



class TestCodeParser(unittest.TestCase):

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

                self.assertListEqual(
                    tree1.descendants.tolist(),
                    tree2.descendants.tolist(),
                )
                self.assertListEqual(
                    tree1.parents.tolist(),
                    tree2.parents.tolist(),
                )

                self.assertListEqual(
                    tree1.node_data,
                    tree2.node_data
                )

                self.assertNotIn("[ERROR]", tree1.node_data)

    def test_descendants_wrong_language(self):
        ruby_code = SAMPLE_CODE["ruby"]
        tree1 = self.parse(ruby_code, "java")
        tree2 = self.parse_without_descendants(ruby_code, "java")
        tree1.pprint()
        tree2.pprint()

        self.assertListEqual(
            tree1.descendants.tolist(),
            tree2.descendants.tolist(),
        )
        self.assertListEqual(
            tree1.parents.tolist(),
            tree2.parents.tolist(),
        )

        self.assertListEqual(
            tree1.node_data,
            tree2.node_data
        )

        self.assertIn("[ERROR]", tree1.node_data)

    def test_parsing_all(self):
        parser = CodeParser()

        for language, code in SAMPLE_CODE.items():
            with self.subTest(f"Parsing {language}") as t:
                tree = parser.parse(code, language)
                decoded_code = parser.unparse(tree)
                tree.pprint()
                if code != decoded_code:
                    print(code)
                    print()
                    print(decoded_code)
                    tree.pprint()
                    print('#' * 100)
                self.assertSequenceEqual(code, decoded_code)


if __name__ == '__main__':
    unittest.main()
