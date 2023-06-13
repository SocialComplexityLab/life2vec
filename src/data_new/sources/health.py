from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import dask.dataframe as dd
import pandas as pd

from ..decorators import save_parquet
from ..ops import sort_partitions
from ..serialize import DATA_ROOT
from .base import FIELD_TYPE, TokenSource


@dataclass
class HealthTokens(TokenSource):
    """This generates tokens based on information from the LPR dataset.
    Currently loads data from a CSV dump of LPR.

    :param input_csv: CSV file from which to load the LPR dataset.
    :param earliest_start: The earliest start date of a hospital encounter.
    """

    name: str = "health"
    fields: List[FIELD_TYPE] = field(
        default_factory=lambda: ["C_ADIAG", "C_INDM", "C_PATTYPE"]
    )

    input_csv: Path = DATA_ROOT / "rawdata" / "health" / "LPRADM.csv"
    earliest_start: str = "01/01/2008"

    def _post_init__(self) -> None:
        self._earliest_start = pd.to_datetime(self.earliest_start)

    @save_parquet(
        DATA_ROOT / "processed/sources/{self.name}/tokenized",
        on_validation_error="error",
    )
    def tokenized(self) -> dd.DataFrame:
        """
        Loads the indexed data, then tokenizes it.
        Clamps the C_ADIAG field, and converts C_INDM and C_PATTYPE to strings.
        """

        result = (
            self.indexed()
            .assign(
                C_ADIAG=lambda x: x.C_ADIAG.str[1:4],
                C_INDM=lambda x: x.C_INDM.map(
                    {"1": "URGENT", "2": "NON_URGENT"}
                ).astype("string"),
                C_PATTYPE=lambda x: x.C_PATTYPE.map(
                    {"0": "INPAT", "2": "OUTPAT", "3": "EMERGENCY"}
                ).astype("string"),
            )
            .pipe(sort_partitions, columns=["START_DATE"])[
                ["START_DATE", *self.field_labels()]
            ]
        )
        assert isinstance(result, dd.DataFrame)
        return result

    @save_parquet(
        DATA_ROOT / "interim/sources/{self.name}/indexed",
        on_validation_error="recompute",
    )
    def indexed(self) -> dd.DataFrame:
        """Loads the parsed data, sets the index, then saves the indexed data"""
        result = self.parsed().set_index("PERSON_ID")
        assert isinstance(result, dd.DataFrame)
        return result

    @save_parquet(
        DATA_ROOT / "interim/sources/{self.name}/parsed",
        on_validation_error="error",
        verify_index=False,
    )
    def parsed(self) -> dd.DataFrame:
        """Parses the CSV file, applies some basic filtering, then saves the result
        as compressed parquet file, as this is easier to parse than the CSV for the
        next steps"""

        columns = [
            "PERSON_ID",
            "C_PATTYPE",
            "C_ADIAG",
            "C_INDM",
            "D_INDDTO",
        ]

        ddf = dd.read_csv(
            self.input_csv,
            low_memory=False,
            usecols=columns,
            on_bad_lines="error",
            assume_missing=True,
            dtype={
                "PERSON_ID": float,  # Deal with missing values
                "C_ADIAG": "string",
                "C_INDM": "string",
                "C_PATTYPE": "string",
            },
            blocksize="256MB",
        )

        # Drop missing values and deal with datatypes
        ddf = (
            ddf.dropna(subset=["PERSON_ID", "C_ADIAG"])
            .assign(
                PERSON_ID=lambda x: x.PERSON_ID.astype(int),
                D_INDDTO=lambda x: dd.to_datetime(
                    x.D_INDDTO,
                    format="%d%b%Y:%X",
                    errors="coerce",
                ),
            )
            .rename(columns={"D_INDDTO": "START_DATE"})
            .loc[lambda x: x.START_DATE >= self.earliest_start]
        )

        if self.downsample:
            ddf = self.downsample_persons(ddf)

        assert isinstance(ddf, dd.DataFrame)

        return ddf
