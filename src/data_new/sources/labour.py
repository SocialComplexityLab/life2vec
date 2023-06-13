from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import dask.dataframe as dd
import pandas as pd

from ..ops import concat_sorted, sort_partitions

from ..decorators import save_parquet
from ..serialize import DATA_ROOT
from .base import FIELD_TYPE, Binned, TokenSource


def parse_date(x: dd.Series, year: int) -> dd.Series:
    assert isinstance(x, dd.Series)
    if year > 2015:
        result = dd.to_datetime(x, format=r"%d/%m/%Y")
    else:
        result = dd.to_datetime(x, origin="01/01/1960", unit="D")
    assert isinstance(result, dd.Series)
    return result


@dataclass
class LabourTokens(TokenSource):
    """Tokens from the labour data."""

    name: str = "labour"
    fields: List[FIELD_TYPE] = field(
        default_factory=lambda: [
            #"WORK_MUNICIPALITY",
            Binned("WORK_INCOME", prefix="INCOME", n_bins=100),
            Binned("TIME_DAYS_RATE", prefix="TIME", n_bins=10),
            "WORK_INDUSTRY",
            "WORK_SECTOR",
            "ATP_RATE_CODE",
            "RES_MUNICIPALITY",
            "WORK_POSITION",
            "SOC_STATUS",
            "SOC_MODIFIER",
        ]
    )

    input_csv_dir: Path = DATA_ROOT / "rawdata" / "labour"
    start_year: int = 2008
    end_year: int = 2018

    @save_parquet(
        DATA_ROOT / "processed/sources/{self.name}/tokenized",
        on_validation_error="error",
    )
    def tokenized(self) -> dd.DataFrame:
        """
        Loads each tokenized year, and combines them into a single tokenized dataframe.
        """

        parts = [
            self.tokenized_part(year)
            for year in range(self.start_year, self.end_year + 1)
        ]
        result = concat_sorted(parts, columns=["START_DATE"])
        result = result.reset_index().set_index("PERSON_ID", npartitions="auto", sorted=True) ## this line makes computations longer, but provides robust partitions

        assert isinstance(result, dd.DataFrame)
        return result

    @save_parquet(
        DATA_ROOT / "interim/sources/{self.name}/tokenized_{year}",
        on_validation_error="recompute",
    )
    def tokenized_part(self, year: int) -> dd.DataFrame:
        """Loads a single indexed year, applies the tokenization logic, then saves
        the result."""

        ddf = self.indexed(year)

        result = (
            ddf.assign(
                #WORK_MUNICIPALITY=lambda x: "WMUN_" + x.WORK_MUNICIPALITY,
                WORK_INDUSTRY=lambda x: "IND_" + x.WORK_INDUSTRY,
                WORK_SECTOR=lambda x: "ENT_" + x.WORK_SECTOR,
                ATP_RATE_CODE=lambda x: "ATP_" + x.ATP_RATE_CODE,
                RES_MUNICIPALITY=lambda x: "RMUN_" + x.RES_MUNICIPALITY,
                WORK_POSITION=lambda x: "POS_" + x.WORK_POSITION,
                SOC_STATUS=lambda x: "SOC_" + x.SOC_STATUS,
                SOC_MODIFIER=lambda x: "SOM_" + x.SOC_MODIFIER,
            )
            # Remove INCOME_0
            .assign(WORK_INCOME=lambda x: x.WORK_INCOME.where(lambda x: x > 0, pd.NA))[
                ["START_DATE", "AGE", *self.field_labels()]
            ].astype(self.token_dtypes())
        )

        assert isinstance(result, dd.DataFrame)
        return result

    def token_dtypes(self) -> Dict[str, pd.CategoricalDtype]:
        """Supplies pandas categorical datatypes for each variable in the tokenized
        data. It is critical that we use these explicit types, as automatically
        determining categoricals in dask is 1: expensive, and 2: error prone as we might
        get mismatches between years/partitions
        """
        return {
            #"WORK_MUNICIPALITY": pd.CategoricalDtype(
            #    [f"WMUN_{i:03}" for i in range(1000)]
            #),
            "WORK_INDUSTRY": pd.CategoricalDtype([f"IND_{i:04}" for i in range(10000)]),
            "WORK_SECTOR": pd.CategoricalDtype([f"ENT_{i}" for i in range(11, 100)]),
            "ATP_RATE_CODE": pd.CategoricalDtype([f"ATP_{x}" for x in list("0ABCDEF")]),
            "RES_MUNICIPALITY": pd.CategoricalDtype(
                [f"RMUN_{i:03}" for i in range(1000)]
            ),
            "WORK_POSITION": pd.CategoricalDtype([f"POS_{i:04}" for i in range(10000)]),
            "SOC_STATUS": pd.CategoricalDtype([f"SOC_{i:03}" for i in range(1000)]),
            "SOC_MODIFIER": pd.CategoricalDtype([f"SOM_{i}" for i in range(100_000)]),
        }

    @save_parquet(
        DATA_ROOT / "interim/sources/{self.name}/indexed_{year}",
        on_validation_error="recompute",
    )
    def indexed(self, year: int) -> dd.DataFrame:
        """Loads the parses data, then indexes the data by PERSON_ID. I am converting
        each string column to categorical and back again since strings are not very
        efficient in pandas/dask.
        """
        result = (
            self.parsed(year)
            .pipe(lambda x: x.categorize(x.select_dtypes("string").columns))
            .set_index("PERSON_ID")
            .pipe(sort_partitions, columns=["START_DATE"])
            .pipe(
                lambda x: x.astype(
                    {k: "string" for k in x.select_dtypes("category").columns}
                )
            )
        )

        assert isinstance(result, dd.DataFrame)
        return result

    @save_parquet(
        DATA_ROOT / "interim/sources/{self.name}/parsed_{year}",
        on_validation_error="error",
        verify_index=False,
    )
    def parsed(self, year: int) -> dd.DataFrame:
        """Does a single pass of the CSV file, some basic filtering, then saves the
        data as parquet."""

        columns = [
            "ALDER_AMR",
            #"ARB_BEL_KOM_KODE",
            "ARB_HOVED_BRA_DB07",
            "ARB_SEKTORKODE",
            "ATP_BELOEB",
            "ATP_BIDRAG_SATS_KODE",
            "BOPAEL_KOM_KODE",
            "BREDT_LOEN_BELOEB",
            "DISCO_KODE",
            "FOED_DAG",
            "FRA_DATO",
            "IE_TYPE",
            "KOEN",
            "OPR_LAND",
            "PERSON_ID",
            "SOC_STATUS_KODE",
            "TILSTAND_KODE_AMR",
            "TILSTAND_LAENGDE_AAR",
            "I_BEFOLKNINGEN_KODE",
        ]

        translated = [
            "AGE",
            #"WORK_MUNICIPALITY",
            "WORK_INDUSTRY",
            "WORK_SECTOR",
            "ATP_AMOUNT",
            "ATP_RATE_CODE",
            "RES_MUNICIPALITY",
            "WORK_INCOME",
            "WORK_POSITION",
            "BIRTHDAY",
            "START_DATE",
            "RES_RESIDENCE",  # emigration status?
            "GENDER",
            "RES_ORIGIN",
            "PERSON_ID",
            "SOC_STATUS",
            "SOC_MODIFIER",
            "TIME_DAYS_RATE",
            "IN_POPULATION_CODE",
        ]

        path = self.input_csv_dir / f"amrun{year}.csv"

        ddf = dd.read_csv(
            path,
            assume_missing=True,
            dtype={
                "KOEN": "string",
                "I_BEFOLKNINGEN_KODE": "string",
                "ARB_HOVED_BRA_DB07": "string",
                "SOC_STATUS_KODE": "string",
                "DISCO_KODE": "string",
                "ATP_BIDRAG_SATS_KODE": "string",
                "BREDT_LOEN_BELOEB": float,
                "TILSTAND_LAENGDE_AAR": float,
                "TILSTAND_KODE_AMR": "string",
                #"ARB_BEL_KOM_KODE": "string",
                "ARB_SEKTORKODE": "string",
                "OPR_LAND": "string",
                "IE_TYPE": "string",
                "BOPAEL_KOM_KODE": "string",
            },
            usecols=columns,
            parse_dates=False,
            encoding="latin-1",
            engine="c",
            low_memory=False,
            blocksize="256MB",
        )

        ddf = (
            ddf.dropna(subset=["PERSON_ID", "FRA_DATO", "FOED_DAG", "IE_TYPE", "KOEN"])
            .loc[lambda x: (x.KOEN != "9") & (x.I_BEFOLKNINGEN_KODE == "1")]
            .assign(
                PERSON_ID=lambda x: x.PERSON_ID.astype(int),
                FOED_DAG=lambda x: parse_date(x.FOED_DAG, year),
                FRA_DATO=lambda x: parse_date(x.FRA_DATO, year),
                ARB_HOVED_BRA_DB07=lambda x: x.ARB_HOVED_BRA_DB07.str[:4],
                DISCO_KODE=lambda x: x.DISCO_KODE.str[:4],
                YEAR=year,
            )
            .rename(columns=dict(zip(columns, translated)))
        ) ## is it correct though ?  should we check whether KOEN is the same over all records ? 

        if self.downsample:
            ddf = self.downsample_persons(ddf)

        assert isinstance(ddf, dd.DataFrame)
        return ddf
