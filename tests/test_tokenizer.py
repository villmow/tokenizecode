import unittest

import torch

from tokenizecode import CodeTokenizer, CodeParser, SplitLinesTraversal

from codesamples import SAMPLE_CODE
from tokenizecode.utils import TensorTreeWithStrings, TensorTreeWithInts, TensorTree


class TestTokenizer(unittest.TestCase):

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

    def test_spans(self) -> None:
        bpe_file = "./assets/bpe40k-20ksamples.json"
        tokenizer = CodeTokenizer.from_file(bpe_file)
        language = "java"
        code = SAMPLE_CODE[language]

        # parsing produces a tree with strings.
        tree, spans = tokenizer.parser.parse(code, language, output_positions=True)
        print(spans)
        l = [span for i, span in enumerate(spans) if tree.is_leaf(i)]
        print(len(l))
        encoding = tokenizer.encode(tree)

        print(encoding.offsets)
        print(len(encoding.offsets))

    def test_encode(self) -> None:
        tokenizer = CodeTokenizer()

        language = "java"
        code = SAMPLE_CODE[language]

        code_encoding = tokenizer.encode(code, language)
        print(code_encoding.tokens)
        print(code_encoding.ids)
        code_decoded = tokenizer.decode(code_encoding.ids)
        self.assertSequenceEqual(code, code_decoded)

    def test_encode_with_a_tree(self) -> None:
        tokenizer = CodeTokenizer()

        language = "java"
        code = SAMPLE_CODE[language]

        tree = tokenizer.parse(code, language)
        code_encoding_with_tree = tokenizer.encode(tree)
        code_encoding_without_tree = tokenizer.encode(code, language)
        self.assertSequenceEqual(code_encoding_without_tree.tokens, code_encoding_with_tree.tokens)
        self.assertSequenceEqual(code_encoding_without_tree.ids, code_encoding_with_tree.ids)

        code_decoded = tokenizer.decode(code_encoding_with_tree.ids)
        self.assertSequenceEqual(code, code_decoded)

    def test_encode_to_tree(self):
        tokenizer = CodeTokenizer()

        language = "java"

        code = SAMPLE_CODE[language]

        parsed_tree = tokenizer.parse(code, language)

        tree_encoded = tokenizer.encode_to_tree(code, language)
        tree_encoded.pprint()

        decoded_tree = tokenizer.decode_tree(tree_encoded, keep_bpe=False)
        self.assertTreeEqual(parsed_tree, decoded_tree)



if __name__ == '__main__':
    unittest.main()
