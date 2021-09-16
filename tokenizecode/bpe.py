import logging
from pathlib import Path

from typing import Callable, Union, Optional, Dict

import numpy as np

from tensortree import TensorTree
import tensortree


log = logging.getLogger(__name__)

BPE_NONTERMINAL= "<[BPE]>"


def encode_terminals(
        tree: TensorTree, encoder: Callable[[list[str]], list[list[str]]],
) -> TensorTree:
    """
    Splits all terminals at once in a sequence of tokens and parents in O(n).

    encoder is a function that takes a list of strings (i.e. terminal symbols) and
        returns a list of list of strings (one list of subwords for each word)
    """
    # encode all words at once, might save some time with sentencepiece
    leaf_indices = tree.leaf_indices()
    leaves = [tree.get_node_data(leaf_idx) for leaf_idx in leaf_indices]

    # actually encode the tokens
    encoded_tokens: list[list[str]] = encoder(list(leaves))

    # compute final amount of nodes
    num_final_nodes = len(tree) - len(leaves) + sum(len(splitted_word) for splitted_word in encoded_tokens)

    # additionally allocate space for 1 nonterminal for every splitted word (<[BPE]>)
    # note this may be to much (as unsplitted words will not have the additional nonterminal).
    # we will truncate the parents array at the end
    num_final_nodes += len(encoded_tokens)

    # init output arrays
    new_tokens = []
    new_parents = np.zeros((num_final_nodes,), dtype=np.int32)

    # copy old parents
    new_parents[:len(tree.data.parents)] = tree.data.parents.numpy()

    for i, token in enumerate(tree.data.node_data):
        new_idx = len(new_tokens)

        if tree.is_leaf(i):
            subtokens = encoded_tokens.pop(0)

            # use original token if BPE returns nothing (for spaces)
            if not subtokens:
                subtokens = [tree.get_node_data(i)]

            num_new_nodes = len(subtokens) - 1  # one token is already part of the tree

            # add the nonterminal only if we actually splitted the node
            if num_new_nodes > 0:
                subtokens = [BPE_NONTERMINAL] + subtokens
                num_new_nodes += 1

            new_tokens += subtokens
            if num_new_nodes > 0:
                # shift all parents except the one of the token thats already part of the tree
                new_parents[new_idx + num_new_nodes + 1:] = new_parents[new_idx + 1:-num_new_nodes]

                # and adjust parent pointers to nodes defined after this node
                new_parents[new_parents > new_idx] += num_new_nodes

                # each subtoken has its predecessor as parent (so just use the appropriate range)
                new_token_parents = new_idx #np.arange(new_idx, new_idx + num_new_nodes)
                new_parents[new_idx + 1: new_idx + num_new_nodes + 1] = new_token_parents
        else:
            # otherwise do nothing. parent is already copied
            new_tokens.append(token)

    new_parents = new_parents[:len(new_tokens)]
    return tensortree.tree(node_data=new_tokens, parents=new_parents)


def decode_terminals(
    tree: TensorTree, decoder: Callable[[list[str]], str],
) -> TensorTree:
    """
    Joins splitted terminals at once in O(n).

    decoder is a function that takes a list of strings (i.e. bpe symbols) and
        returns a string
    """
    # init output arrays
    new_tokens = []
    new_parents = []
    parent_map = {}  # save mapping between old and new parents

    num_nodes_removed = 0  # total
    skip_number_of_tokens = 0  # subword

    for i, (token, parent) in enumerate(zip(tree.node_data, tree.parents)):
        # adjust parents
        parent = parent.item()
        if parent in parent_map:
            new_parent = parent_map[parent]
        else:
            new_parent = parent - num_nodes_removed
            parent_map[parent] = new_parent

        # maybe skip tokens if we're inside a subword
        if skip_number_of_tokens > 0:
            skip_number_of_tokens -= 1
            continue

        # decode multiple tokens
        elif token == BPE_NONTERMINAL:
            subtokens = tree[i].node_data[1:] # remove NT
            new_token = decoder(subtokens)

            new_subwords_removed = len(subtokens)

            num_nodes_removed += new_subwords_removed
            skip_number_of_tokens = new_subwords_removed

        # decode single token
        elif tree.is_leaf(i):
            new_token = decoder(token)

            if not new_token:
                new_token = token
        else:
            new_token = token

        new_tokens.append(new_token)
        new_parents.append(new_parent)

    return tensortree.tree(node_data=new_tokens, parents=new_parents)


class BaseTreeBPE:

    def encode_text(self, text: Union[str, list[str]]) -> Union[list[str], list[list[str]]]:
        raise NotImplementedError()

    def decode_text(self, text: Union[str, list[str]]) -> str:
        raise NotImplementedError()

    def encode(self, tree: TensorTree) -> TensorTree:
        """
        Main function. Applies BPE to all terminals in the linearized tree.
        """
        if not isinstance(tree, TensorTree):
            raise ValueError(f"Can only encode a tensortree object and not {type(tree)}.")

        return encode_terminals(
            tree, encoder=self.encode_text
        )

    def decode(self, tree: TensorTree) -> TensorTree:
        if not isinstance(tree, TensorTree):
            raise ValueError("Can only encode a tensortree object.")

        return decode_terminals(
            tree, decoder=self.decode_text
        )


class SentencePieceBPE(BaseTreeBPE):

    def __init__(self, model: Union[str, Path]):
        super().__init__()
        try:
            import sentencepiece as spm
            self.sp = spm.SentencePieceProcessor(model_file=model)
        except ImportError:
            raise ImportError('Please install sentencepiece with: pip install sentencepiece')

    def encode_text(self, text: Union[str, list[str]]) -> Union[list[str], list[list[str]]]:
        return self.sp.encode(text, out_type=str)

    def decode_text(self, text: Union[str, list[str]]) -> str:
        if isinstance(text, list):
            return "".join(text).replace('\u2581', ' ').strip()

        return text.replace(' ', '').replace('\u2581', ' ').strip()


TreeBPE = SentencePieceBPE
