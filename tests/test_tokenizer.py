import unittest
from typing import Optional, Tuple, List

import torch

from tokenizecode import CodeTokenizer, CodeParser, SplitLinesTraversal

from codesamples import SAMPLE_CODE
from tokenizecode.utils import TensorTreeWithStrings, TensorTreeWithInts, TensorTree

import datasets


DATASET = datasets.load_from_disk("assets/github_dataset_small")


def get_sample_idx(repo, path):
    for i, sample in enumerate(DATASET):
        if sample["repository"] == repo and sample["path"] == path:
            return i


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
        print("loaded tokenizer")
        code_encoding = tokenizer.encode(code, language)
        print(code_encoding)
        print(code_encoding.ids)
        code_decoded = tokenizer.decode(code_encoding.ids)
        self.assertSequenceEqual(code, code_decoded)

    def test_encode_weird_sample(self) -> None:
        from multicoder.utils import to_tree
        sample = DATASET[36135]
        tree = to_tree(sample)

        # tokenizer = CodeTokenizer.from_file("/home/johannes/projects/tokenizecode/trained_tokenizers/20211108_bpe20k-fpl40k.json")
        tokenizer = CodeTokenizer.from_file("/home/johannes/projects/multicoder/tokenizer/20211108_bpe30k-fpl40k-with-nonterminals.json")
        # tokenizer = CodeTokenizer.from_file("/home/johannes/projects/multicoder/my_bpe_on_dataset4.json")
        print(sample["repository"])
        print(sample["path"])
        code_encoding = tokenizer.encode_to_tree(tree)
        tokenizer.decode_tree(code_encoding, keep_bpe=True).pprint()
        # print(code_encoding)
        # print(code_encoding.ids)
        # code_decoded = tokenizer.decode(code_encoding.ids)
        # self.assertSequenceEqual(code, code_decoded)

    def test_encode_weird_sample2(self) -> None:
        from multicoder.utils import to_tree
        repo = "microsoft/vscode"
        path = "vscode-main/src/vs/workbench/contrib/codeEditor/browser/quickaccess/gotoSymbolQuickAccess.ts"
        idx = 36136

        if idx == None:
            idx = get_sample_idx(repo, path)
            print(idx)

        sample = DATASET[idx]
        tree = to_tree(sample)

        # tokenizer = CodeTokenizer.from_file("/home/johannes/projects/tokenizecode/trained_tokenizers/20211108_bpe20k-fpl40k.json")
        tokenizer = CodeTokenizer.from_file("/home/johannes/projects/multicoder/tokenizer/20211108_bpe30k-fpl40k-with-nonterminals.json")
        # tokenizer = CodeTokenizer.from_file("/home/johannes/projects/multicoder/my_bpe_on_dataset4.json")
        print(sample["repository"])
        print(sample["path"])
        tree.pprint()
        code_encoding = tokenizer.encode_to_tree(tree)
        tokenizer.decode_tree(code_encoding, keep_bpe=True).pprint()
        # print(code_encoding)
        # print(code_encoding.ids)
        # code_decoded = tokenizer.decode(code_encoding.ids)
        # self.assertSequenceEqual(code, code_decoded)


    def test_encode_weird_sample3(self) -> None:
        from multicoder.utils import to_tree
        repo = "tensorflow/tensorflow"
        path = "tensorflow-master/tensorflow/core/tfrt/eager/tfrt_context.cc"
        idx = 14454

        if idx == None:
            idx = get_sample_idx(repo, path)
            print(idx)

        sample = DATASET[idx]
        tree = to_tree(sample)

        # tokenizer = CodeTokenizer.from_file("/home/johannes/projects/tokenizecode/trained_tokenizers/20211108_bpe20k-fpl40k.json")
        tokenizer = CodeTokenizer.from_file("/home/johannes/projects/multicoder/tokenizer/20211108_bpe30k-fpl40k-with-nonterminals.json")
        # tokenizer = CodeTokenizer.from_file("/home/johannes/projects/multicoder/my_bpe_on_dataset4.json")
        print(sample["repository"])
        print(sample["path"])
        tree.pprint()
        code_encoding = tokenizer.encode_to_tree(tree)
        tokenizer.decode_tree(code_encoding, keep_bpe=True).pprint()
        # print(code_encoding)
        # print(code_encoding.ids)
        # code_decoded = tokenizer.decode(code_encoding.ids)
        # self.assertSequenceEqual(code, code_decoded)

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

    def test_encode_to_tree_gpt2(self):
        tokenizer = load_gpt2_tokenizer()
        language = "java"

        code = SAMPLE_CODE[language]

        parsed_tree = tokenizer.parse(code, language)

        tree_encoded = tokenizer.encode_to_tree(code, language)
        tokenizer.decode_tree(tree_encoded, keep_bpe=True).pprint()

        decoded_tree = tokenizer.decode_tree(tree_encoded, keep_bpe=False)
        self.assertTreeEqual(parsed_tree, decoded_tree)

    def test_encode_gpt2(self) -> None:
        from multicoder.utils import to_tree
        tokenizer = load_gpt2_tokenizer()

        for i, sample in enumerate(DATASET.shuffle(42)):
            with self.subTest(f"{sample['repository']}/{sample['path']}") as test:
                if sample["is_code_file"]:
                    tree = to_tree(sample)
                    tree_encoded = tokenizer.encode_to_tree(tree)
                    decoded_tree = tokenizer.decode_tree(tree_encoded, keep_bpe=False)
                    self.assertTreeEqual(tree, decoded_tree)
                else:
                    text = sample["tokens"][0]
                    # text = sample["tokens"]
                    text_encoded = tokenizer.encode_text(text)
                    text_decded = tokenizer.decode(text_encoded.ids)
                    self.assertSequenceEqual(text, text_decded)
                    # self.assertSequenceEqual(text[0], text_decded)
            if i > 100:
                break
                # tokenizer.decode_tree(code_encoding, keep_bpe=True).pprint()


    def test_encode_all(self) -> None:
        from multicoder.utils import to_tree
        tokenizer = CodeTokenizer()

        for i, sample in enumerate(DATASET.shuffle(42)):
            with self.subTest(f"{sample['repository']}/{sample['path']}") as test:
                if sample["is_code_file"]:
                    tree = to_tree(sample)
                    tree_encoded = tokenizer.encode_to_tree(tree)
                    decoded_tree = tokenizer.decode_tree(tree_encoded, keep_bpe=False)
                    self.assertTreeEqual(tree, decoded_tree)
                else:
                    text = sample["tokens"][0]
                    text_encoded = tokenizer.encode_text(text)
                    text_decded = tokenizer.decode(text_encoded.ids)
                    self.assertSequenceEqual(text, text_decded)
                    # self.assertSequenceEqual(text[0], text_decded)
            if i > 100:
                break
                # tokenizer.decode_tree(code_encoding, keep_bpe=True).pprint()



import json


def create_gpt2_tokenizer():
    from transformers import PreTrainedTokenizerFast
    from tokenizecode.bpe import TokenizerBPE
    bpe = PreTrainedTokenizerFast.from_pretrained(
        "gpt2",
        add_prefix_space=True,
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[CLS]",
        eos_token="[EOS]",
        model_max_len=1024
    )

    import json
    with open("/home/johannes/projects/multicoder/tokenizer/nonterminals.json", "rt") as f:
        nonterminals = json.load(f)

    added = bpe.add_tokens(new_tokens=["[MASK]", "[BPE]", "        ", "    ", "  ", " ", "\n\n", "\n"], special_tokens=True)
    added = bpe.add_tokens(new_tokens=list(nonterminals.keys()), special_tokens=True)

    bpe.save_pretrained("tokenizer")
    print(added)
    print(bpe)

    tokenizer = CodeTokenizer(TokenizerBPE(bpe))
    return tokenizer



def load_gpt2_tokenizer():
    create_gpt2_tokenizer()

    from transformers import PreTrainedTokenizerFast
    from tokenizecode.bpe import TokenizerBPE

    bpe = PreTrainedTokenizerFast.from_pretrained("tokenizer",         model_max_len=1024)
    print(bpe)

    tokenizer = CodeTokenizer(TokenizerBPE(bpe))
    return tokenizer





if __name__ == '__main__':
    unittest.main()
