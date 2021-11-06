import logging
from collections import Counter
from pathlib import Path

from typing import Callable, Union, Optional, Dict, Generator

import datasets
import numpy as np
from scipy.stats import stats

from tensortree import TensorTree
import tensortree
from tokenizecode.utils import TensorTreeWithStrings, TensorTreeWithInts, is_tree_of_strings

try:
    import tokenizers
except ImportError:
    raise ImportError('Please install tokenizers with: pip install tokenizers')


log = logging.getLogger(__name__)


BPE_NONTERMINAL= "[BPE]"


def encode_terminals(
        tree: TensorTreeWithStrings, encoder: Callable[[list[str]], list[list[str]]],
) -> TensorTreeWithStrings:
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
                # print("-" * 100)
                # print("subtokens", subtokens)
                # print("new_idx", new_idx)
                # print("num_tokens_added_for_leaf", num_new_nodes)

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
    tree: TensorTreeWithStrings, decoder: Callable[[list[str]], str],
) -> TensorTreeWithStrings:
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


class TokenizerBPE:

    def __init__(self, model: Union[str, Path]):
        super().__init__()
        self.model = tokenizers.Tokenizer.from_file(str(model))
        self.bpe_nonterminal_id = self.model.token_to_id(BPE_NONTERMINAL)

    def encode_text(self, text: Union[str, list[str]]) -> tokenizers.Encoding:
        # either string or list of strings
        is_pretokenized = isinstance(text, list)
        # text = replace_whitespace(text)  # we only do this in text mode, because the parser does this while building the tree

        encoding = self.model.encode(text, is_pretokenized=is_pretokenized)
        return encoding

    def decode_text(self, ids) -> str:
        print(ids)
        print([self.model.id_to_token(i) for i in ids])
        out = self.model.decode(ids, skip_special_tokens=False)
        print(out)
        print("-" * 100)
        # out = restore_whitespace(out)
        return out

    def encode_tree(self, tree: TensorTree) -> TensorTree:
        """ Produces a tree with IDs and artificial BPE nodes. """
        assert isinstance(tree.node_data[0], str), "tree should consist of strings"
        import torch
        encoded_nodes = self.model.encode(tree.node_data, is_pretokenized=True)

        # compute final amount of nodes
        approx_num_nodes = len(encoded_nodes.ids) * 2

        # init output arrays
        new_tokens = torch.zeros((approx_num_nodes,), dtype=torch.int64)
        new_parents = np.zeros((approx_num_nodes,), dtype=np.int64)
        new_descendants = np.zeros((approx_num_nodes,), dtype=np.int64)

        # copy old parents
        new_parents[:len(tree.parents)] = tree.parents.numpy()

        new_descendants[:len(tree.descendants)] = tree.descendants.numpy()
        active = np.full_like(new_descendants, fill_value=False, dtype=bool)  # bool tensor with all false

        # torch version
        # new_descendants[:len(tree.descendants)] = tree.descendants
        # active = torch.full_like(new_descendants, fill_value=False, dtype=torch.bool)  # bool tensor with all false

        idx_new_node = 0
        idx_encoded_node = 0

        word_ids = encoded_nodes.word_ids
        token_ids = encoded_nodes.ids

        def _add_next_token(token=None, parent_added: bool = True):
            nonlocal new_tokens, idx_new_node, idx_encoded_node

            if token is None:
                token = token_ids[idx_encoded_node]
                idx_encoded_node += 1

            if parent_added:
                active[new_parents[idx_new_node] + 1:] = False
            active[idx_new_node] = True

            new_tokens[idx_new_node] = token
            idx_new_node += 1

        for old_node_idx, token in enumerate(tree.node_data):
            if not tree.is_leaf(old_node_idx):
                # Nonterminals -> are encoded as special symbols and should have been never splitted by the tokenizer

                assert word_ids[idx_encoded_node] != word_ids[idx_encoded_node + 1], "nonterminals should not have been splitted"
                _add_next_token()
            # Leaf, which has been splitted
            elif (idx_encoded_node + 1) < len(word_ids) and (word_ids[idx_encoded_node] == word_ids[idx_encoded_node + 1]):

                num_new_tokens_for_leaf = 0
                idx_bpe_token = idx_new_node

                # add BPE nonterminal
                _add_next_token(self.bpe_nonterminal_id, parent_added=True)

                # add every subword, but the last
                while (
                        (idx_encoded_node + 1) < len(word_ids)
                        and word_ids[idx_encoded_node] == word_ids[idx_encoded_node + 1]
                ):
                    _add_next_token(parent_added=False)
                    num_new_tokens_for_leaf += 1

                # add last subword
                _add_next_token(parent_added=False)
                num_new_tokens_for_leaf += 1

                if num_new_tokens_for_leaf > 0:
                    # shift all parents except the one of the token thats already part of the tree
                    new_parents[idx_bpe_token + num_new_tokens_for_leaf + 1:] = new_parents[idx_bpe_token + 1:-num_new_tokens_for_leaf]

                    # and adjust parent pointers to nodes defined after this node
                    new_parents[new_parents > idx_bpe_token] += num_new_tokens_for_leaf

                    # each subtoken has its predecessor as parent (so just use the appropriate range)
                    new_token_parents = idx_bpe_token  # np.arange(new_idx, new_idx + num_new_nodes)
                    new_parents[idx_bpe_token + 1: idx_bpe_token + num_new_tokens_for_leaf + 1] = new_token_parents

                # adjust descendants of newly added nodes
                idx_bpe_token = idx_bpe_token
                active[idx_bpe_token] = True  # set BPE token as active
                active[idx_bpe_token + 1:] = False  # and deactivate all other tokens
                new_descendants[idx_bpe_token + num_new_tokens_for_leaf + 1:] = new_descendants[idx_bpe_token + 1:-num_new_tokens_for_leaf]

                # torch version
                # new_descendants[idx_bpe_token + num_new_tokens_for_leaf + 1:] = new_descendants[idx_bpe_token + 1:-num_new_tokens_for_leaf].clone()

                new_descendants[idx_bpe_token:idx_bpe_token + num_new_tokens_for_leaf + 1] = 0
                new_descendants[active] += num_new_tokens_for_leaf
                active[idx_bpe_token:] = False

            elif (idx_encoded_node + 1) < len(word_ids) and (word_ids[idx_encoded_node] != word_ids[idx_encoded_node + 1]):
                # token has not been splitted, -> simply add it
                _add_next_token()
            elif (idx_encoded_node + 1) == len(word_ids):
                # token is last token (and has not been splitted)
                _add_next_token()
            else:
                assert False

        new_tokens = new_tokens[:idx_new_node]
        new_parents = new_parents[:idx_new_node]
        new_descendants = new_descendants[:idx_new_node]

        tree = tensortree.tree(node_data=new_tokens, parents=new_parents, descendants=new_descendants)

        # # FIXME load tree with new descendants
        # tree = tensortree.tree(node_data=new_tokens, parents=new_parents)
        # assert torch.equal(tree.descendants, torch.from_numpy(new_descendants)), " new_descendants should be equal"
        # torch version
        # assert torch.equal(tree.descendants, new_descendants), " new_descendants should be equal"
        return tree

    def decode_tree(self, tree, keep_bpe: bool = False) -> TensorTree:
        """ Returns a tree with strings as node data. keep_bpe keeps artificial BPE nodes. """

        import tensortree
        tree_with_strings = tensortree.tree(
            parents=tree.parents,
            descendants=tree.descendants,
            node_data=[self.model.id_to_token(t) for t in tree.node_data]
        )

        if not keep_bpe:
            return self.remove_bpe_from_tree(tree_with_strings)

        return tree_with_strings

    @staticmethod
    def remove_bpe_from_tree(tree_with_strings: TensorTree) -> TensorTree:
        return decode_terminals(tree_with_strings, decoder=lambda list_of_str: "".join(list_of_str))

    @classmethod
    def train_dataset(cls, dataset: datasets.Dataset, save_file: Path, vocab_size: int,
                      shuffle: bool = True, max_samples: Optional[int] = None, seed: int = 42,
                      **kwargs):
        """ dataset should have columns tokens, parents, descendants """
        if shuffle:
            dataset = dataset.shuffle(seed)

        def data_generator() -> Generator[str, None, Counter]:
            """
            A dataset with the columns ["tokens", "parents", "descendants"]

            :param dataset: datasets.Dataset
            :return:
            """
            from tqdm import tqdm

            nonterminals = {}

            total = len(dataset) if max_samples is None else max_samples
            stats = {}
            with tqdm(total=total, desc="at sample") as progress:
                for i, s in enumerate(dataset):
                    lang = s["language"]
                    stats[lang] = stats.get(lang, 0) + 1

                    if not s["descendants"]:
                        yield from s["tokens"]
                    else:
                        for token, num_descendants in zip(s["tokens"], s["descendants"]):
                            if num_descendants == 0:
                                yield token
                            else:
                                nonterminals[token] = nonterminals.get(token, 0) + 1

                    if max_samples is not None and i == max_samples:
                        break

                    progress.update()
                    progress.set_postfix(stats)

            log.info("Build dataset on the following files:")
            for lang, count in stats.items():
                log.info(f"{lang}: {count} files")

            nonterminals = Counter(dict(sorted(nonterminals.items(), key=lambda item: item[1])))
            return nonterminals

        return cls.train(data_generator(), save_file, vocab_size, **kwargs)

    @classmethod
    def train(cls, data_generator: Generator[str, None, Counter], save_file: Path, vocab_size: int, min_frequency: int, **kwargs):
        raise NotImplementedError


class WordPieceBPE(TokenizerBPE):

    @classmethod
    def train(cls, data_generator: Generator[str, None, Counter], save_file: Path, vocab_size: int, min_frequency: int, **kwargs):
        import tokenizers
        from tokenizers import normalizers
        from tokenizers import Tokenizer
        from tokenizers.models import BPE, WordPiece, Unigram
        from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer

        tokenizer = Tokenizer(WordPiece(unk_token="[UNK]", continuing_subword_prefix="@@"))

        # tokenizer.pre_tokenizer = Whitespace()
        # tokenizer.normalizer = normalizers.NFKC()

        trainer = WordPieceTrainer(
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", BPE_NONTERMINAL],
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            continuing_subword_prefix="@@"
        )

        ################### save counted nonterminals #########################
        nonterminals: Optional[Counter] = None

        def handle_return(generator, func):
            returned = yield from generator
            func(returned)

        def save_nonterminals(return_value):
            nonlocal nonterminals
            nonterminals = return_value

        ######################################################################

        gen = handle_return(
            generator=data_generator,
            func=save_nonterminals
        )
        tokenizer.train_from_iterator(gen, trainer)
        log.info(nonterminals)
        if nonterminals is not None:
            tokenizer.add_special_tokens(list(nonterminals.keys()))
        else:
            log.warning("Nonterminals is None")

        tokenizer.save(str(save_file.absolute()))

        return cls(save_file)


class BytePairBPE(TokenizerBPE):

    @staticmethod
    def specials():
        return [
                "[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", BPE_NONTERMINAL,
                # "↵", "↹", "↦", "⏎⏎", "⏎", "········", "····", "··", "·",
                "\r", "\t", "\v", "\n\n", "\n", "        ", "    ", "  ", " ",
        ]

    @classmethod
    def train(cls, data_generator: Generator[str, None, Counter], save_file: Path, vocab_size: int,
              total: Optional[int] = None, **kwargs):
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            special_tokens=cls.specials(),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            vocab_size=vocab_size,
        )

        ################### save counted nonterminals #########################
        nonterminals: Optional[Counter] = None

        def handle_return(generator, func):
            returned = yield from generator
            func(returned)

        def save_nonterminals(return_value):
            nonlocal nonterminals
            nonterminals = return_value

        ######################################################################

        gen = handle_return(
            generator=data_generator,
            func=save_nonterminals
        )
        tokenizer.train_from_iterator(gen, trainer, length=total)
        log.info(nonterminals)
        if nonterminals is not None:
            tokenizer.add_special_tokens(list(nonterminals.keys()))
        else:
            log.warning("Nonterminals is none. Need to add nonterminals afterwards!")

        log.info(f"Saving tokenizer to {save_file.absolute()}")
        tokenizer.save(str(save_file.absolute()))

        return cls(save_file)


TreeBPE = SentencePieceBPE
