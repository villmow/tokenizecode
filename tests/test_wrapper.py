import unittest
from typing import Optional, Tuple, List

import torch

from tokenizecode import CodeTokenizer, CodeParser

from tokenizecode.utils import TensorTreeWithStrings, TensorTreeWithInts, TensorTree
from tokenizecode import WrappedParser

class TestWrapper(unittest.TestCase):

    def assertTreeEqual(self, tree1: TensorTree, tree2: TensorTree, msg = ...) -> None:
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

    def test_parse_wrapped_class(self) -> None:
        # some code that appears inside a function in java
        java_code_from_inside_class = """String concat(char chars) {
  if (chars.length == 0) {
    return "";
  }
  StringBuilder s = new StringBuilder(chars.length);
  for (char c : chars) {
    s.append(c);
  }
  return s.toString();
}
"""

        # check that this code cant be parsed by the default parser
        parser = CodeParser()
        parsing_original = parser.parse(java_code_from_inside_class, "java")
        # parsing_original.tree.pprint()
        self.assertGreater(parsing_original.num_errors, 0, "this code should not be parsable by the default parser")

        wrapped_parser = WrappedParser(parser)
        parsing_wrapped, method = wrapped_parser.parse(java_code_from_inside_class, "java")
        print(method)
        # parsing_wrapped.tree.pprint()
        self.assertEqual(parsing_wrapped.num_errors, 0, "this code should be parsable by the wrapped parser")
        # check that the tree is the same
        # self.assertTreeEqual(parsing_original.tree, parsing_wrapped.tree)

        p1 = parsing_original.positions
        p2 = parsing_wrapped.positions

        p1 = [pos for i, pos in enumerate(parsing_original.positions) if parsing_original.tree.is_leaf(i) and parsing_original.tree.get_node_data(i) != ""]
        p2 = [pos for i, pos in enumerate(parsing_wrapped.positions) if parsing_wrapped.tree.is_leaf(i) and parsing_wrapped.tree.get_node_data(i) != ""]

        l1  = parsing_original.tree.leaves()
        l1 = list(filter(lambda x: x != "", l1))
        l2 = parsing_wrapped.tree.leaves()
        l2 = list(filter(lambda x: x != "", l2))

        self.assertListEqual(l1, l2)

        # check that position mapping is the same
        self.assertListEqual(p1, p2)
    def test_parse_wrapped_method(self) -> None:
        # some code that appears inside a function in java
        java_code_from_inside_method = "Result4 = A + B + C + D + A*B + A*C + A*D + B*C + B*D + C*D\n        + A*B*C + A*B*D + A*C*D + B*C*D + A*B*C*D\n        =      A + B + C + A*B + A*C + B*C + A*B*C\n        + (1 + A + B + C + A*B + A*C + B*C + A*B*C) * D\n        = Result3 + (1 + Result3) * D"

        # check that this code cant be parsed by the default parser
        parser = CodeParser()
        parsing_original = parser.parse(java_code_from_inside_method, "java")
        # parsing_original.tree.pprint()
        self.assertGreater(parsing_original.num_errors, 0, "this code should not be parsable by the default parser")

        wrapped_parser = WrappedParser(parser)
        parsing_wrapped, method = wrapped_parser.parse(java_code_from_inside_method, "java")
        print(method)
        # parsing_wrapped.tree.pprint()
        self.assertEqual(parsing_wrapped.num_errors, 0, "this code should be parsable by the wrapped parser")
        # check that the tree is the same
        # self.assertTreeEqual(parsing_original.tree, parsing_wrapped.tree)

        p1 = [pos for i, pos in enumerate(parsing_original.positions) if parsing_original.tree.is_leaf(i) and parsing_original.tree.get_node_data(i) != ""]
        p2 = [pos for i, pos in enumerate(parsing_wrapped.positions) if parsing_wrapped.tree.is_leaf(i) and parsing_wrapped.tree.get_node_data(i) != ""]

        l1 = parsing_original.tree.leaves()
        l1 = list(filter(lambda x: x != "", l1))
        l2 = parsing_wrapped.tree.leaves()
        l2 = list(filter(lambda x: x != "", l2))

        self.assertListEqual(l1, l2)

        # check that position mapping is the same
        self.assertListEqual(p1, p2)



if __name__ == '__main__':
    unittest.main()
