import logging
from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from typing import List, Tuple, TypeVar, cast
from xml.dom.minidom import Document

import numpy as np
import torch
from random import shuffle

from src.data_new.types import Background, PersonDocument, EncodedDocument
from src.tasks.base import Task

log = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class MLM(Task):
    """
    Task for used with Masked language modelling.

    .. todo::
        Describe MLM

    :param mask_ratio: Fraction of tokens to mask.
    :param smart_masking: Whether to apply smart masking (use tokens from the same group when choosing randoms).
    """

    # MLM Specific params
    mask_ratio: float = 0.30
    smart_masking: bool = False

    def encode_document(self, document: PersonDocument) -> "MLMEncodedDocument":

        prefix_sentence = (
            ["[CLS]"] + Background.get_sentence(document.background) + ["[SEP]"]
        )

        ############################################
        ### CLS TASK
        document, targ_cls = self.cls_task(document)
        ############################################

        sentences = [prefix_sentence] + [s + ["[SEP]"] for s in document.sentences]
        sentence_lengths = [len(x) for x in sentences]

        def expand(x: List[T]) -> List[T]:
            assert len(x) == len(sentence_lengths)
            return list(
                chain.from_iterable(
                    length * [i] for length, i in zip(sentence_lengths, x)
                )
            )

        abspos_expanded = expand([0] + document.abspos)
        age_expanded = expand([0.0] + document.age)
        assert document.segment is not None
        segment_expanded = expand([1] + document.segment)

        flat_sentences = np.concatenate(sentences)

        token2index = self.datamodule.vocabulary.token2index
        unk_id = token2index["[UNK]"]

        #print(flat_sentences[500:550])
        token_ids = np.array([token2index.get(x, unk_id) for x in flat_sentences])
        masked_sentences, masked_indx, masked_tokens = self.mlm_mask(token_ids.copy())

        length = len(token_ids)

        input_ids = np.zeros((4, self.max_length))
        input_ids[0, :length] = masked_sentences
        input_ids[1, :length] = abspos_expanded
        input_ids[2, :length] = age_expanded
        input_ids[3, :length] = segment_expanded

        padding_mask = np.repeat(False, self.max_length)
        padding_mask[:length] = True

        # TODO: Consider renaming, to document/sentences instead of sequence...
        # would require refactoring of the modelling also though

        original_sequence = np.zeros(self.max_length)
        original_sequence[:length] = token_ids

        sequence_id = np.array(document.person_id)

        return MLMEncodedDocument(
            sequence_id=sequence_id,
            input_ids=input_ids,
            padding_mask=padding_mask,
            target_tokens=masked_tokens,
            target_pos=masked_indx,
            target_cls=targ_cls,
            original_sequence=original_sequence,
        )

    # These could (maybe should?) also be calculated in the __post_init__.
    # Accessing the serialized methods in a parallel context may give problems down
    # the line.
    @cached_property
    def token_groups(self) -> List[Tuple[int, int]]:
        """Return pairs of first and last index for each token category in the
        vocabulary excluding GENERAL.
        """

        vocab = self.datamodule.vocabulary.vocab()
        no_general = vocab.CATEGORY != "GENERAL"
        token_groups = (
            vocab.loc[no_general]
            .groupby("CATEGORY")
            .ID.agg(["first", "last"])
            .sort_values("first")
            .to_records(index=False)
            .tolist()
        )
        return cast(List[Tuple[int, int]], token_groups)

    def cls_task(self, document: PersonDocument):
        p = np.random.rand(1)
        if p <0.05:
            document.sentences.reverse()
            targ_cls = 1
        elif p>0.95:
            shuffle(document.sentences)
            targ_cls = 2
        else:
            targ_cls = 0

        return document, targ_cls

    def mlm_mask(
        self, token_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mask out the tokens for mlm training"""

        vocabulary = self.datamodule.vocabulary
        token2index = vocabulary.token2index

        unk_id = token2index["[UNK]"]
        mask_id = token2index["[MASK]"]
        sep_id = token2index["[SEP]"]

        # limit is length of an actual sequence
        n_tokens = len(token_ids)

        num_tokens_to_mask = np.floor(n_tokens * self.mask_ratio).astype(np.int32)
        # firt 10% of tokens won't be changed
        pos_unchange = np.floor(num_tokens_to_mask * 0.1).astype(np.int32)
        # last 10% of tokens would be random ; the rest will be changed
        pos_random = num_tokens_to_mask - pos_unchange

        # we do not mask SEP and UNK
        legal_mask = (token_ids[1:] != sep_id) & (token_ids[1:] != unk_id)
        legal_indx = np.arange(start=1, stop=n_tokens)[legal_mask]

        indx_to_mask = np.random.choice(
            a=legal_indx, size=num_tokens_to_mask, replace=False
        )

        max_masked_num = np.floor(self.mask_ratio * self.max_length).astype(np.int32)

        # positions of the masked tokens
        y_indx = np.full(shape=max_masked_num, fill_value=int(self.max_length - 1))
        y_indx[: len(indx_to_mask)] = indx_to_mask.copy()

        # remember the actual tokens on positions
        y_token = np.zeros(shape=max_masked_num)
        y_token[: len(indx_to_mask)] = token_ids[indx_to_mask].copy()

        # masked token_ids #rather change the sampling domain for accurate masking
        # ratio?
        token_ids[indx_to_mask[pos_unchange:pos_random]] = mask_id

        vocab_size = len(token2index)
        n_general_tokens = len(vocabulary.general_tokens)

        if self.smart_masking:

            smart_edge = int(pos_random + int(pos_unchange * 0.3))

            # Random 7% of all random cases
            token_ids[indx_to_mask[smart_edge:]] = np.random.randint(
                # low we do not mask any special tokens
                low=n_general_tokens,
                high=vocab_size,
                size=(1, len(indx_to_mask[smart_edge:])),
            )

            # Smart Random 3% of all the cases
            smart_values = token_ids[indx_to_mask[pos_random:smart_edge]]

            for i, j in self.token_groups:
                smart_values = self.smart_masked(smart_values, i, j + 1)  # background

            token_ids[indx_to_mask[pos_random:smart_edge]] = smart_values

        else:
            token_ids[indx_to_mask[pos_random:]] = torch.randint(
                # low we do not mask any special tokens
                low=n_general_tokens,
                high=vocab_size,
                size=(1, len(indx_to_mask[pos_random:])),
            )
        return token_ids, y_indx, y_token

    @staticmethod
    def smart_masked(x: np.ndarray, min_i: int, max_i: int) -> np.ndarray:
        """Applies the smart_masking scheme"""
        ix = np.argwhere((x >= min_i) & (x < max_i))
        if len(ix) > 0:
            x[ix] = np.random.randint(low=min_i, high=max_i, size=(len(ix), 1))
        return x


@dataclass
class MLMEncodedDocument(EncodedDocument[MLM]):
    sequence_id: np.ndarray
    input_ids: np.ndarray
    padding_mask: np.ndarray
    target_tokens: np.ndarray
    target_pos: np.ndarray
    target_cls: np.ndarray
    original_sequence: np.ndarray
