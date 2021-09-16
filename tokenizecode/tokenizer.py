from typing import Optional, Union

from tensortree import TensorTree

from tokenizecode.bpe import BaseTreeBPE
from tokenizecode.parser import CodeParser


class CodeTokenizer:
    """ Combines tokenizer and bpe. """

    def __init__(self, bpe: BaseTreeBPE, parser: Optional[CodeParser] = None):
        self.bpe = bpe
        self.parser = parser if parser is not None else CodeParser()

    def tokenize(self, code: str = None, lang: str = "java") -> TensorTree:
        tree = self.parser.parse(code, lang)
        bpe_tree = self.bpe.encode(tree)
        return bpe_tree

    def detokenize(self,
                   tree: Optional[Union[TensorTree, list[TensorTree]]] = None,
                   text: Optional[Union[str, list[str], list[list[str]]]] = None) -> Union[str, list[str]]:
        if tree is not None:
            return self.detokenize_tree(tree)
        elif text is not None:
            return self.detokenize_text(text)
        else:
            raise ValueError("Either tree or text must be set.")

    def detokenize_tree(self, tree: Union[TensorTree, list[TensorTree]]) -> Union[str, list[str]]:
        if isinstance(tree, list):
            return [self.detokenize_tree(t) for t in tree]

        tree = self.bpe.decode(tree)
        code = self.parser.unparse(tree)
        return code

    def detokenize_text(self, text: Union[list[str], list[list[str]]]) -> Union[str, list[str]]:
        if not text:
            return ""
        if isinstance(text[0], list):
            return [self.detokenize_text(t) for t in text]

        text = self.bpe.decode_text(text)
        return text
