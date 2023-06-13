from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Dict, List, Tuple, Union, cast

import dask
import pandas as pd

from .decorators import save_pickle, save_tsv
from .serialize import DATA_ROOT
from .sources.base import TokenSource

if TYPE_CHECKING:
    from .datamodule import Corpus


class Vocabulary:
    """
    Vocabulary base class.

    :ivar name: Name of the vocabulary.
    :ivar general_tokens: General tokens.
    :ivar background_tokens: Background tokens.
    """

    name: str
    general_tokens: List[str]
    background_tokens: List[str]

    def vocab(self) -> pd.DataFrame:
        """
        Define for subclass to implement new vocabulary.

        :return:

            A :class:`pandas.DataFrame` object with the following columns
            * ID - The token ids
            * TOKEN - The token strings
            * CATEGORY - The token category


        """
        raise NotImplementedError

    def tokens(self) -> List[str]:
        """Return the tokens in order as list of strings"""
        return cast(List[str], self.vocab().TOKEN.astype("string").to_list())

    def token_ids(self) -> List[int]:
        """Return the token ids in order as a list of integers"""
        return cast(List[int], self.vocab().ID.astype(int).to_list())

    @cached_property
    def token2index(self) -> Dict[str, int]:
        """A dictionary mapping tokens to token ids"""
        return dict(zip(self.tokens(), self.token_ids()))

    @cached_property
    def index2token(self) -> Dict[int, str]:
        """A dictionary mapping token ids to tokens"""
        return dict(zip(self.token_ids(), self.tokens()))

    def size(self) -> int:
        """Returns the number of tokens"""
        return len(self.token2index)

    def prepare(self) -> None:
        """Prepareres the vocabulary by calling :meth:`vocab`"""
        self.vocab()


@dataclass
class TSVVocabulary(Vocabulary):
    # Todo: This could be used to implement an already existing tsv vocab
    ...


@dataclass
class CorpusVocabulary(Vocabulary):
    """
    Generate a vocabulary from the tokenized training data of a corpus.

    :param corpus: The :class:`src.data_new.Corpus` to generate the vocabulary from.
    :param name: Name of the vocabulary.
    :param general_tokens: General tokens.
    :param background_tokens: Background tokens.
    :param year_range: Range of years (inclusive) to generate tokens for.
    :param min_token_count: The minimum number of occurances of a token to be included
        in the vocabulary.
    :param min_token_count_field: Field-specific minimum token counts.

    """

    corpus: "Corpus"
    name: str
    general_tokens: List[str] = field(
        default_factory=lambda: [
            "[PAD]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
            "[PLCH0]",
            "[PLCH1]",
            "[PLCH2]",
            "[PLCH3]",
            "[PLCH4]",
            "[UNK]",
        ]
    )
    background_tokens: List[str] = field(
        default_factory=lambda: ["F", "M", "DK", "NON_DK"]
    )
    year_range: Tuple[int, int] = (1946, 1991)  # inclusive
    min_token_count: int = 1000
    min_token_count_field: Dict[str, int] = field(default_factory=dict)

    @save_tsv(DATA_ROOT / "processed/vocab/{self.name}/", on_validation_error="error")
    def vocab(self) -> pd.DataFrame:
        """Filters the tokens by count, sorts them lexicographically for each source,
        and computes the voculary with the field labels as categories.
        """

        general = pd.DataFrame({"TOKEN": self.general_tokens, "CATEGORY": "GENERAL"})
        background = pd.DataFrame(
            {"TOKEN": self.background_tokens, "CATEGORY": "BACKGROUND"}
        )
        month = pd.DataFrame(
            {"TOKEN": [f"MONTH_{i}" for i in range(1, 13)], "CATEGORY": "MONTH"}
        )
        year = pd.DataFrame(
            {
                "TOKEN": [
                    f"YEAR_{i}"
                    for i in range(self.year_range[0], self.year_range[1] + 1)
                ],
                "CATEGORY": "YEAR",
            }
        )

        vocab_parts = [general, background, month, year]

        def sort_key(x: str) -> Tuple[Union[str, int], ...]:
            def maybe_to_int(y: str) -> Union[str, int]:
                return int(y) if y.isdigit() else y

            return tuple(maybe_to_int(y) for y in x.split("_"))

        for source in self.corpus.sources:
            token_counts = self.token_counts(source)
            for label in source.field_labels():
                counts = token_counts[label]
                min_count = self.min_token_count_field.get(label, self.min_token_count)
                tokens = [k for k, v in counts.items() if v >= min_count]
                tokens.sort(key=sort_key)
                tokens_df = pd.DataFrame({"TOKEN": tokens, "CATEGORY": label})
                vocab_parts.append(tokens_df)

        return pd.concat(vocab_parts, ignore_index=True).rename_axis(index="ID")

    @save_pickle(
        DATA_ROOT / "interim/vocab/{self.name}/token_counts/{source.name}",
        on_validation_error="recompute", ###hmmm
    )
    def token_counts(self, source: TokenSource) -> Dict[str, Dict[str, int]]:
        """Returns the token counts for the source for the training data"""

        ids = self.corpus.population.data_split().train
        tokenized = self.corpus.tokenized_and_transformed(source).loc[
            lambda x: x.index.isin(ids)
        ]

        counts = {}
        for field_ in source.field_labels():
            ### Constrained Counts (number of sequences that have a token)
            counts[field_] = tokenized.reset_index()[["PERSON_ID", field_]].drop_duplicates()[field_].value_counts() 
            ### Full counts (without the uniqueness constraint)
            #counts[field_] = tokenized[field_].value_counts()
        (counts,) = dask.compute(counts)
        counts = {k: v.to_dict() for k, v in counts.items()}

        return counts
