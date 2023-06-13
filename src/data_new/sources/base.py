from dataclasses import dataclass
from typing import List, Union

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd


@dataclass
class Field:
    """
    Base class for fields.

    :param field_label: Column in the dataframe to transform
    """

    field_label: str

    def transform(self, x: dd.Series) -> dd.Series:
        """Returns data as-is. Implement in a subclass to apply a transformation"""
        return x

    def fit(self, x: dd.DataFrame) -> None:
        """Called on the train partition of the tokenized data. Prepares state for
        applying :meth:`transform`"""


@dataclass
class Binned(Field):
    """
    Specifies that a field should be binned based on the observed (approximate)
    quantiles in the data.

    :param field_label: Column in the dataframe to bin
    :param prefix: Prefix of resulting tokens, such that the tokens becomes
        "<PREFIX>_#".
    :param n_bins: Number of bins to calculate.
    """

    field_label: str
    prefix: str
    n_bins: int = 100

    def __post_init__(self) -> None:
        self.bins_ = None

    def transform(self, x: dd.Series) -> dd.Series:
        """
        Applies binning to supplied values.

        :param x: Values to bin.

        :return: Binned values as a :class:`dask.dataframe.Series`
        """

        assert self.field_label == x.name
        assert self.bins_ is not None

        n_bins = len(self.bins_) - 1
        categories = [f"{self.prefix}_{i+1}" for i in range(n_bins)]

        tmp_frame = x.to_frame()
        name = self.field_label

        tmp_frame["_digitized"] = da.digitize(
            tmp_frame[name].to_dask_array(lengths=True), bins=self.bins_
        )
        tmp_frame["_digitized"] = (
            self.prefix + "_" + tmp_frame["_digitized"].astype("string")
        )
        tmp_frame["_digitized"] = tmp_frame["_digitized"].where(
            ~tmp_frame[name].isna(), pd.NA
        )  # Set NA values to NA bin

        right_over_token = self.prefix + f"_{n_bins + 1}"
        right_over_token_to = self.prefix + f"_{n_bins}"

        tmp_frame["_digitized"] = tmp_frame["_digitized"].where(
            lambda x: x.isna() | (x != right_over_token), right_over_token_to
        )

        dtype = pd.CategoricalDtype(categories)

        result = tmp_frame["_digitized"].rename(name).astype(dtype)
        assert isinstance(result, dd.Series)
        return result

    def fit(self, x: dd.DataFrame) -> None:
        """
        Computes :attr:`n_bins`-1 quntiles of x[:attr:`field_label`]. The calculated
        bin-edges are stores in :attr:`bins_`

        :param x: :class:`dask.dataframe.DataFrame` with :attr:`field_label` column.
        """
        q = np.linspace(0.0, 1.0, self.n_bins - 1, endpoint=True)
        quantiles = x[self.field_label].quantile(q=q).compute().tolist()
        self.bins_ = quantiles


FIELD_TYPE = Union[str, Field]


@dataclass
class TokenSource:
    """
    Base class for token sources.
    This class defines the interface for defining a source of tokens.
    This class can not be used as is, but must be subclassed with the
    tokenized method implemented in order to implement new token sources.


    :param name: Name of source. Useful for dealing with serializing
        multiple different versions of the same object
    :param fields: Which fields from the tokenized data to concatenate
        into sentences
    :param downsample: Whether or not to downsample the data
    """

    name: str
    fields: List[FIELD_TYPE]
    downsample: bool = False

    def tokenized(self) -> dd.DataFrame:
        """
        Method to deliver the tokenized data. Should return a
        :class:`dask.dataframe.DataFrame` object.

        The dataframe should be indexed by PERSON_ID, have a datetime column called
        "START_DATE", as well as a column for each field in :attr:`self.fields`
        containing the tokenized data.
        The tokenized data should be strings (or *known* categoricals, see :mod:`dask`
        documentation). They can be NA in which case they are omitted from the final
        sentence.
        For each PERSON_ID, the observations should be sorted by START_DATE.

        Since further proccesing will be done with this dataframe, I recommend against
        returning dataframes containg large unevaluated computational graphs. This can
        be avoided by eg. returning the output of :func:`dask.dataframe.read_parquet`.
        Alternatively you can use the :func:`src.data.decorators.save_parquet`
        decorator.

        :return:

            A :class:`dask.dataframe.DataFrame` object with the following columns:

            * PERSON_ID (Index column) - The person ids.

            * START_DATE - The date for the event that the tokens describe

            * AGE (Optional) - If supplied, this age will be used instead of the age
              calculated based on the birthday.

            * Token columns - A column for each of the fields in :attr:`self.fields`

        """
        raise NotImplementedError

    def prepare(self) -> None:
        """Prepares the data by calling the :meth:`tokenized`."""
        self.tokenized()

    @staticmethod
    def downsample_persons(ddf: dd.DataFrame) -> dd.DataFrame:
        """
        Filters a dask DataFrame down to every every 100th person.
        It is the responsibility of each source implementation to use this method
        appropriately when :attr:`self.downsampling` is :obj:`True`.
        """

        if "PERSON_ID" in ddf.columns:
            result = ddf.loc[lambda x: (x.PERSON_ID % 100) == 0]
        elif ddf.index.name == "PERSON_ID":
            result = ddf.loc[lambda x: (x.index % 100) == 0]
        else:
            raise AttributeError("DataFrame has no PERSON_ID attribute")

        assert isinstance(result, dd.DataFrame)
        return result

    def field_labels(self) -> List[str]:
        return [f.field_label if isinstance(f, Field) else f for f in self.fields]
