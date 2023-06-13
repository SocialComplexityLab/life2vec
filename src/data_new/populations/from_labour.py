from dataclasses import dataclass
from functools import reduce
from typing import Tuple

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd

from ..decorators import save_parquet, save_pickle
from ..ops import concat_sorted
from ..serialize import DATA_ROOT
from ..sources.labour import LabourTokens
from .base import DataSplit, Population

nunique = dd.Aggregation(
    name="nunique",
    chunk=lambda s: s.apply(lambda x: list(set(x))),
    agg=lambda s0: s0.obj.groupby(level=list(range(s0.obj.index.nlevels))).sum(),
    finalize=lambda s1: s1.apply(lambda final: len(set(final))),
)

@dataclass
class FromLabour(Population):
    """
    A cohord defined based on the labour dataset.

    :param labour_data: Instance of :class:`src.data.sources.LabourTokens` token source
        to base the population on.
    :param year: Year in which to require 12 registrations in the labour data, in order
        for a person to be included.
    :param earliest_birthday: Earliest allowed birthday
    :param latest_birthday: Latest allowed birthday
    :param seed: Seed for splitting training, validation and test dataset
    :param train_val_test: Fraction of the data to be included in the three data splits.
        Must sum to 1.
    """

    labour_data: LabourTokens
    name: str = "labour_2015"
    year: int = 2015
    earliest_birthday: str = "01-01-1946"
    latest_birthday: str = "01-01-1991"
    

    seed: int = 123
    train_val_test: Tuple[float, float, float] = (0.7, 0.15, 0.15)

    def __post_init__(self) -> None:
        assert sum(self.train_val_test) == 1.0
        self._earliest_birthday = pd.to_datetime(self.earliest_birthday, format="%d-%m-%Y")
        self._latest_birthday = pd.to_datetime(self.latest_birthday, format="%d-%m-%Y")

    @save_pickle(
        DATA_ROOT / "processed/populations/{self.name}/population",
        on_validation_error="error",
    )
    def population(self) -> pd.DataFrame:
        """Loads the combined labour data, and filters according to:
            * Birthday
            * Birthday- and gender consistency.
            * Number of events during :attr:`year`.

        Includes the RES_ORIGIN column in the population data.
        """
        low_interval = pd.to_datetime(f"{self.year}-01-01")
        high_interval = pd.to_datetime(f"{self.year}-12-31")

        combined = self.combined()

        inconsistent_sex_bd = (
            combined.groupby(combined.index)
            .agg({"BIRTHDAY": nunique, "GENDER": nunique})
            .pipe(lambda x: (x > 1).any(1))
            .loc[lambda x: x]
            .index.values
        )

## DASK 
        insufficient_events_in_year = (
            combined.assign(
                is_in_year=lambda x: (x.START_DATE >= low_interval)
                & (x.START_DATE <= high_interval)
            )
            .groupby(combined.index)
            .is_in_year.sum()
            .loc[lambda x: x < 12]
            .index.values
        )

        bd_out_of_bound = (
            combined.loc[
                lambda x: (x.BIRTHDAY >= self._latest_birthday)
                | (x.BIRTHDAY <= self._earliest_birthday)
            ]
            .index.unique()
            .values
        )

        ids_to_remove = reduce(
            da.union1d,
            [inconsistent_sex_bd, insufficient_events_in_year, bd_out_of_bound],
        )

        result = (
            combined.loc[lambda x: ~x.index.isin(ids_to_remove.compute())]
            .reset_index()
            .groupby("PERSON_ID")
            .agg(
                {
                    "BIRTHDAY": "first",
                    "RES_ORIGIN": "first",
                    "GENDER": "first",
                }
            )
            .compute()
        )

        assert isinstance(result, pd.DataFrame)
        return result

    @save_parquet(
        DATA_ROOT / "interim/populations/{self.name}/combined",
        on_validation_error="recompute",
    )
    def combined(self) -> dd.DataFrame:
        """
        Pulls out the PERSON_ID (the index), START_DATE, BIRTHDAY, RES_ORIGIN and GENDER
        columns of the indexed labour data for each years, and combined them into a
        single dataframe.

        Maps RES_ORIGIN to DK or NON-DK and the GENDER codes 1/2 to M/F.
        """

        ls = self.labour_data

        year_data = [
            ls.indexed(i)[["START_DATE", "BIRTHDAY", "RES_ORIGIN", "GENDER"]]
            for i in range(ls.start_year, ls.end_year + 1)
        ]

        result = concat_sorted(year_data, columns=[]).assign(
            RES_ORIGIN=lambda x: x.RES_ORIGIN.where(lambda x: x == "5100", "0").map(
                {"0": "NON_DK", "5100": "DK"}
            ),
            GENDER=lambda x: x.GENDER.map({"1": "M", "2": "F"}),
        )
        result = result.reset_index().set_index("PERSON_ID", npartitions="auto", sorted=True) ## this line makes computations longer, but provides robust partitions
        assert isinstance(result, dd.DataFrame)
        return result

    @save_pickle(DATA_ROOT / "processed/populations/{self.name}/data_split")
    def data_split(self) -> DataSplit:
        """Split data based on :attr:`seed` using :attr:`train_val_test` as ratios"""
        ids = self.population().index.to_numpy()
        np.random.default_rng(self.seed).shuffle(ids)
        split_idxs = np.round(np.cumsum(self.train_val_test) * len(ids))[:2].astype(int)
        train_ids, val_ids, test_ids = np.split(ids, split_idxs)
        return DataSplit(
            train=train_ids,
            val=val_ids,
            test=test_ids,
        )
