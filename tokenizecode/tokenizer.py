from pathlib import Path
from typing import Optional, Union

import tokenizers
import torch
import transformers

from tensortree import TensorTree

from tokenizecode.bpe import TokenizerBPE
from tokenizecode.parser import CodeParser
from tokenizecode.utils import TensorTreeWithStrings, TensorTreeWithInts, is_tree_of_strings, get_project_root


DEFAULT_TOKENIZER_BPE = get_project_root() / "trained_tokenizers/20211108_bpe30k-fpl40k-with-nonterminals.json"


class CodeTokenizer:
    """ Combines tokenizer and bpe. """

    def __init__(self, tokenizer: Optional[TokenizerBPE] = None, parser: Optional[CodeParser] = None):
        if tokenizer is None:
            tokenizer = TokenizerBPE.from_pretrained(DEFAULT_TOKENIZER_BPE)

        self.tokenizer = tokenizer
        self._parser = parser if parser is not None else None

    @property
    def hf_tokenizer(self) -> transformers.PreTrainedTokenizerFast:
        return self.tokenizer.tokenizer

    @property
    def parser(self):
        if self._parser is None:
            self._parser = CodeParser()

        return self._parser

    def _inputs_to_tree(self, inputs: Union[str, TensorTreeWithStrings], lang: Optional[str] = None) -> TensorTreeWithStrings:
        """ Either parses a piece of code or uses the tree."""
        if isinstance(inputs, str):
            code = inputs

            if lang is None:
                raise ValueError("Will parse code. Language needs to be known.")

            tree = self.parse(code, lang)
        else:
            tree = inputs

            if not is_tree_of_strings(tree):
                raise ValueError("Tree should consist of strings.")

        return tree

    @classmethod
    def from_file(cls, tokenizer_file_or_directory: Path):
        from tokenizecode.bpe import TokenizerBPE
        return cls(TokenizerBPE.from_pretrained(tokenizer_file_or_directory))

    def parse(self, code: str, lang: str) -> TensorTreeWithStrings:
        """ Turns a piece code into a syntax tree. """
        return self.parser.parse(code, lang)

    @staticmethod
    def unparse(tree: TensorTreeWithStrings) -> str:
        """ Turns a syntax_tree tree back into code. """
        if not is_tree_of_strings(tree):
            raise ValueError("Tree should consist of strings.")

        return CodeParser.unparse(tree)

    def encode(self, inputs: Union[str, TensorTreeWithStrings], lang: Optional[str] = None) -> tokenizers.Encoding:
        """ Encodes a piece of code or a syntax tree and returns an encoding for all tokens."""
        tree = self._inputs_to_tree(inputs, lang)
        return self.tokenizer.encode_text(tree.leaves())

    def encode_text(self, text: Union[str, list[str]]) -> tokenizers.Encoding:
        """ Encodes any piece of text or list of text. """
        return self.tokenizer.encode_text(text)

    def encode_to_tree(
            self, inputs: Union[str, TensorTreeWithStrings], lang: Optional[str] = None
    ) -> TensorTreeWithInts:
        """ Encodes a piece of code or a syntax tree and returns **the full syntax tree** encoded (ie only ids as nodes)."""

        tree = self._inputs_to_tree(inputs, lang)
        return self.tokenizer.encode_tree(tree)

    def decode(self, ids) -> str:
        if isinstance(ids, TensorTreeWithInts):
            tree = ids
            decoded_tree = self.decode_tree(tree, keep_bpe=False)
            return self.unparse(decoded_tree)

        return self.tokenizer.decode_text(ids)

    def decode_tree(self, tree: TensorTreeWithInts, keep_bpe: bool = False) -> TensorTreeWithStrings:
        """ Returns a tree with strings as node data. keep_bpe keeps artificial BPE nodes. """
        return self.tokenizer.decode_tree(tree, keep_bpe)

    # def detokenize(self,
    #                tree: Optional[Union[TensorTree, list[TensorTree]]] = None,
    #                text: Optional[Union[str, list[str], list[list[str]]]] = None) -> Union[str, list[str]]:
    #     if tree is not None:
    #         return self.detokenize_tree(tree)
    #     elif text is not None:
    #         return self.detokenize_text(text)
    #     else:
    #         raise ValueError("Either tree or text must be set.")
    #
    # def detokenize_tree(self, tree: Union[TensorTree, list[TensorTree]]) -> Union[str, list[str]]:
    #     if isinstance(tree, list):
    #         return [self.detokenize_tree(t) for t in tree]
    #
    #     tree = self.tokenizer.decode(tree)
    #     code = self.parser.unparse(tree)
    #     return code
    #
    # def detokenize_text(self, text: Union[list[str], list[list[str]]]) -> Union[str, list[str]]:
    #     if not text:
    #         return ""
    #     if isinstance(text[0], list):
    #         return [self.detokenize_text(t) for t in text]
    #
    #     text = self.tokenizer.decode_text(text)
    #     return text

    @staticmethod
    def tree_to_tokens(tree: TensorTree) -> Union[torch.Tensor, list[str]]:
        return tree.leaves()

    def add_specials(self, tokens):
        self.tokenizer.tokenizer.add_special_tokens(
            {"additional_special_tokens": tokens}
        )

    def save(self, filepath: Union[str, Path], pretty: bool = False):
        self.tokenizer.tokenizer.save_pretrained(str(filepath), legacy_format=False)
        # self.tokenizer.tokenizer.save(str(filepath), pretty)
