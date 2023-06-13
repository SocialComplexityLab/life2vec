from dataclasses import dataclass
from doctest import ELLIPSIS_MARKER
from itertools import chain
from typing import List, TypeVar, cast
from datetime import date

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from src.data_new.types import Background, JSONSerializable, PersonDocument, EncodedDocument
from src.tasks.base import Task

from functools import reduce
import logging

T = TypeVar("T")
log = logging.getLogger(__name__)



@dataclass
class TabularCLS(Task):

    def register(self, datamodule) -> None:
        self.datamodule = datamodule
  
        vocab = self.datamodule.vocabulary.vocab()
        vocab = vocab[~vocab.CATEGORY.isin(["GENERAL", "MONTH", "BACKGROUND", "YEAR"])]
        vocab = vocab["TOKEN"].tolist()
        log.info("Input size: %s" %(len(vocab) + 5))
        self.vectorizer = CountVectorizer(vocabulary=vocab,  token_pattern=r"\S+", lowercase=False)
        self.earliest_birthday = self.datamodule.corpus.population._earliest_birthday
        self.latest_birthday = self.datamodule.corpus.population._latest_birthday
        self.period_start = self.datamodule.corpus.population._period_start
        self.period_end = self.datamodule.corpus.population._period_end

        self.max_age = float((self.period_start - self.earliest_birthday).days // 365)
        self.min_age = float((self.period_start - self.latest_birthday).days // 365)

    def minmax_norm(self, birth_year, birth_month):
        age = (self.period_start - pd.to_datetime("1-%s-%s" %(birth_month, birth_year), dayfirst=True)).days // 365
        return (float(age) - self.min_age) / (self.max_age-self.min_age)


    # CLS Specific params
    def get_document(self, person_sentences: pd.DataFrame) -> PersonDocument:
        document = super().get_document(person_sentences)
        target = int(person_sentences.TARGET.iloc[0])
        document.task_info = cast(JSONSerializable, target)
        return document

    def encode_document(self, document: PersonDocument) -> "TCLSEncodedDocument":

        origin_dk  = 1. if document.background.origin == "DK" else 0.
        origin_nd  =  1. - origin_dk
        sex_m = 1. if document.background.gender == "M" else 0.
        sex_f = 1. - sex_m
        age = self.minmax_norm(birth_month=document.background.birth_month, birth_year=document.background.birth_year)

        background = np.array([origin_dk, origin_nd, sex_m, sex_f, age], dtype=np.float32)
        sequence = " ".join(reduce(lambda xs, ys: xs + ys, document.sentences))
        data = self.vectorizer.transform([sequence]).toarray().flatten().astype(np.float32)

        data /= data.sum()

        data = np.concatenate([background, data])


        target = np.array(document.task_info).astype(int)

        sequence_id = np.array(document.person_id)


        return TCLSEncodedDocument(
            sequence_id=sequence_id,
            input_ids=data,
            target=target,
            background=background,
        )

@dataclass
class TCLSEncodedDocument(EncodedDocument[TabularCLS]):
    sequence_id: np.ndarray
    input_ids: np.ndarray
    target: np.ndarray
    background: np.ndarray
