import tree_sitter
from typing import Optional

import unittest

from tokenizecode import TreeSitterParser
from tokenizecode.parser import to_tensortree, FullTraversal

import tensortree
from tokenizecode.utils import replace_whitespace


def to_tensortree_without_descendants(nodes: list) -> tensortree.TensorTree:
    """ Uses tensortrees descendant conversion"""
    node_data, parents, meta = [], [], []

    for node in nodes:
        node_data.append(node.text)
        parents.append(node.parent_id)
        meta.append(node.span)
    return tensortree.tree(parents, node_data)


class TestParser(unittest.TestCase):

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


        d1 = tree1.descendants.tolist()
        d2 = tree2.descendants.tolist()
        self.assertListEqual(d1, d2)

    def test_descendants_java(self):
        code = """import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

class Main {

	public static void main(String[] args) throws IOException {
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		String strNum = br.readLine();
		System.out.printf("%.5f %.5f",(Double.parseDouble(strNum) * Double.parseDouble(strNum) * Math.PI), (2 * Double.parseDouble(strNum) * Math.PI));
	}

}
"""

        tree1 = self.parse(code, "java")
        tree2 = self.parse_without_descendants(code, "java")
        tree1.pprint(node_renderer=lambda n: replace_whitespace(n))
        tree2.pprint(node_renderer=lambda n: replace_whitespace(n))
        d1 = tree1.descendants.tolist()
        d2 = tree2.descendants.tolist()
        self.assertListEqual(d1, d2)

    def test_descendants_ruby(self):
        code = """n = gets.split("")

num = n.map { |n123| n123.to_i }

sum = ""

for i in num do
 if i == 9 then
  sum = sum + "1"
 elsif i == 1 then
  sum = sum + "9"
 end
end

sum = sum.to_i

puts sum"""

        tree1 = self.parse(code, "ruby")
        tree2 = self.parse_without_descendants(code, "ruby")
        tree1.pprint(node_renderer=lambda n: replace_whitespace(n))
        tree2.pprint(node_renderer=lambda n: replace_whitespace(n))
        d1 = tree1.descendants.tolist()
        d2 = tree2.descendants.tolist()
        self.assertListEqual(d1, d2)

    def test_descendants_typescript(self):
        code = """// Welcome to the TypeScript Playground, this is a website
// which gives you a chance to write, share and learn TypeScript.

// You could think of it in three ways:
//
//  - A place to learn TypeScript in a place where nothing can break
//  - A place to experiment with TypeScript syntax, and share the URLs with others
//  - A sandbox to experiment with different compiler features of TypeScript

const anExampleVariable = "Hello World"
console.log(anExampleVariable)

// To learn more about the language, click above in "Examples" or "What's New".
// Otherwise, get started by removing these comments and the world is your playground.
"""

        tree1 = self.parse(code, "typescript")
        tree2 = self.parse_without_descendants(code, "typescript")
        tree1.pprint(node_renderer=lambda n: replace_whitespace(n))
        tree2.pprint(node_renderer=lambda n: replace_whitespace(n))
        d1 = tree1.descendants.tolist()
        d2 = tree2.descendants.tolist()
        self.assertListEqual(d1, d2)

    def test_descendants_haskell(self):
        code = """import Test.QuickCheck
import Data.List (sort)

-- Going from left to right, swaps two adjacent elements if they are not in order.
-- After the first go, the largest element in the list has bubbled up to the end
-- of the list. In the next go, we start swapping from the first element to the
-- penultimate element and so forth.
bubbleSort :: Ord a => [a] -> [a]
bubbleSort xs = go xs (length xs -1)
  where go xs limit | limit > 0 = let swapped = swapTill xs limit in
                                  go swapped (limit -1)
                    | otherwise = xs

-- Swaps adjacent elements in a list if they are not in order, until a limit.
-- After this, the largest elements, from limit to (length xs),
-- are sorted at the list's end.
swapTill :: (Ord a, Num p) => [a] -> p -> [a]
swapTill xs limit = go xs 0
  where go xs count | count < limit = swap xs
                    | otherwise = xs
                      where swap [x] = [x]
                            swap (x:y:xs) | x < y     = x : (go (y:xs) (count +1))
                                          | otherwise = y : (go (x:xs) (count +1))

-- Tests
bubbleSortWorks :: [Int] -> Bool
bubbleSortWorks xs = bubbleSort xs == sort xs

runQuickCheck = quickCheck bubbleSortWorks
"""

        tree1 = self.parse(code, "haskell")
        tree2 = self.parse_without_descendants(code, "haskell")
        tree1.pprint(node_renderer=lambda n: replace_whitespace(n))
        tree2.pprint(node_renderer=lambda n: replace_whitespace(n))
        d1 = tree1.descendants.tolist()
        d2 = tree2.descendants.tolist()
        self.assertListEqual(d1, d2)

    def test_descendants_wrong_language(self):
        code = """n = gets.split("")

num = n.map { |n123| n123.to_i }

sum = ""

for i in num do
 if i == 9 then
  sum = sum + "1"
 elsif i == 1 then
  sum = sum + "9"
 end
end

sum = sum.to_i

puts sum"""

        tree1 = self.parse(code, "java")
        tree2 = self.parse_without_descendants(code, "java")
        tree1.pprint(node_renderer=lambda n: replace_whitespace(n))
        tree2.pprint(node_renderer=lambda n: replace_whitespace(n))
        d1 = tree1.descendants.tolist()
        d2 = tree2.descendants.tolist()
        self.assertListEqual(d1, d2)


if __name__ == '__main__':
    unittest.main()
