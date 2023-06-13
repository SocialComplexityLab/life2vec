from dataclasses import dataclass
from doctest import ELLIPSIS_MARKER
from itertools import chain
from typing import List, TypeVar, cast

import numpy as np
import pandas as pd

from src.data_new.types import Background, JSONSerializable, PersonDocument, EncodedDocument
from src.tasks.base import Task

T = TypeVar("T")

HEXACO_COLS = ['HEXACO_consc3',
 'HEXACO_agree7',
 'HEXACO_open8R',
 'HEXACO_agree8',
 'HEXACO_extra9R',
 'HEXACO_consc9R',
 'HEXACO_agree4',
 'HEXACO_emo7R',
 'HEXACO_extra8',
 'HEXACO_agree1R',
 'HEXACO_open6',
 'HEXACO_extra5',
 'HEXACO_open10R',
 'HEXACO_open3',
 'HEXACO_open5R',
 'HEXACO_agree9R',
 'HEXACO_emo8',
 'HEXACO_hh4',
 'HEXACO_agree5',
 'HEXACO_agree6R',
 'HEXACO_open9',
 'HEXACO_extra10R',
 'HEXACO_consc8R',
 'HEXACO_agree10',
 'HEXACO_consc5R',
 'HEXACO_extra7',
 'HEXACO_emo3',
 'HEXACO_hh3R',
 'HEXACO_consc2R',
 'HEXACO_consc10R',
 'HEXACO_extra2R',
 'HEXACO_emo1',
 'HEXACO_extra6',
 'HEXACO_agree2',
 'HEXACO_emo5',
 'HEXACO_extra4',
 'HEXACO_hh8',
 'HEXACO_consc7',
 'HEXACO_hh7R',
 'HEXACO_consc6',
 'HEXACO_emo10R',
 'HEXACO_hh2',
 'HEXACO_hh5R',
 'HEXACO_consc4R',
 'HEXACO_hh1R',
 'HEXACO_emo6',
 'HEXACO_hh9R',
 'HEXACO_emo2R',
 'HEXACO_extra3R',
 'HEXACO_open7R',
 'HEXACO_emo9',
 'HEXACO_hh6R',
 'HEXACO_open4',
 'HEXACO_agree3R',
 'HEXACO_hh10',
 'HEXACO_emo4R',
 'HEXACO_open2',
 'HEXACO_open1R',
 'HEXACO_extra1',
 'HEXACO_consc1']

@dataclass
class CLS(Task):
    """
    Pulls data from somewhere and uses it for classification?

    .. todo::
        Describe CLS
    """
    # CLS Specific params
    pooled: bool = False
    num_pooled_sep: int = 0


    def __post_init__(self) -> None:
        import warnings
        if self.pooled:
            raise NotImplementedError("Pooled version is not implemented")

    # CLS Specific params
    def get_document(self, person_sentences: pd.DataFrame) -> PersonDocument:
        document = super().get_document(person_sentences)
        target = int(person_sentences.TARGET.iloc[0])
        document.task_info = cast(JSONSerializable, target)  # makes mypy happy

        return document

    def encode_document(self, document: PersonDocument) -> "CLSEncodedDocument":

        prefix_sentence = (
            ["[CLS]"] + Background.get_sentence(document.background) + ["[SEP]"]
        )
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
        age_expanded = expand([0.0] + document.age)  # todo abs_age vs age?
        assert document.segment is not None
        segment_expanded = expand([1] + document.segment)

        token2index = self.datamodule.vocabulary.token2index
        unk_id = token2index["[UNK]"]

        flat_sentences = np.concatenate(sentences)
        token_ids = np.array([token2index.get(x, unk_id) for x in flat_sentences])

        length = len(token_ids)

        input_ids = np.zeros((4, self.max_length))
        input_ids[0, :length] = token_ids
        input_ids[1, :length] = abspos_expanded
        input_ids[2, :length] = age_expanded
        input_ids[3, :length] = segment_expanded

        padding_mask = np.repeat(False, self.max_length)
        padding_mask[:length] = True

        original_sequence = np.zeros(self.max_length)
        original_sequence[:length] = token_ids

        target = np.array(document.task_info).astype(np.float32)

        sequence_id = np.array(document.person_id)

        if self.pooled:
            sep_pos = self.extract_sep_positions(token_ids)
        else:
            sep_pos = np.array([0])


        return CLSEncodedDocument(
            sequence_id=sequence_id,
            input_ids=input_ids,
            padding_mask=padding_mask,
            target=target,
            sep_pos=sep_pos,
            original_sequence=original_sequence,
        )

    def extract_sep_positions(self, token_ids: np.ndarray) -> np.ndarray:

        token2index = self.datamodule.vocabulary.token2index
        sep_id = token2index["[SEP]"]

        MAX_LEN = self.num_pooled_sep 
        _sep_pos = np.where(token_ids == sep_id)[0]
        sep_pos = np.zeros(MAX_LEN)

        if len(_sep_pos) >= MAX_LEN:
            offset = len(_sep_pos) - MAX_LEN
            _sep_pos = _sep_pos[offset:]

        sep_pos[: len(_sep_pos)] = _sep_pos
        return sep_pos


@dataclass
class PSY(CLS):
    # TASK
    def get_document(self, person_sentences: pd.DataFrame) -> PersonDocument:
        document = super(CLS, self).get_document(person_sentences)
        usecols = ['HH','EM','EX','AG','CO','OP', "SDO", "SVOa", "RISK", "CRTi", "CRTr"]
        usecols += [c + "_w" for c in usecols[:-1]]
        target = []
        for col in usecols:
            target.append(float(person_sentences[col].iloc[0]))
        document.task_info = cast(JSONSerializable, target)  # makes mypy happy
        return document

@dataclass
class HEXACO(CLS):
    # TASK
    def get_document(self, person_sentences: pd.DataFrame) -> PersonDocument:
        document = super(CLS, self).get_document(person_sentences)
        usecols = ['HH','EM','EX','AG','CO','OP', "SDO", "SVOa", "RISK", "CRTi", "CRTr"] + HEXACO_COLS
        target = []
        for col in usecols:
            target.append(float(person_sentences[col].iloc[0]))
        document.task_info = cast(JSONSerializable, target)  # makes mypy happy
        return document



@dataclass
class CLSEncodedDocument(EncodedDocument[CLS]):
    sequence_id: np.ndarray
    input_ids: np.ndarray
    padding_mask: np.ndarray
    target: np.ndarray
    sep_pos: np.ndarray
    original_sequence: np.ndarray

@dataclass
class HANEncodedDocument(EncodedDocument[CLS_HAN]):
    sequence_id: np.ndarray
    input_ids: np.ndarray
    position: np.ndarray
    padding_mask: np.ndarray
    target: np.ndarray
    original_sequence: np.ndarray



