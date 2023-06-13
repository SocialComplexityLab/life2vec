from dataclasses import dataclass, field
from typing import List

import dask.dataframe as dd
import pandas as pd
from sqlalchemy import (
    Column,
    Date,
    ForeignKey,
    Integer,
    MetaData,
    Table,
    Text,
    and_,
    extract,
    select,
)

from ..decorators import save_parquet
from ..serialize import DATA_ROOT
from .base import FIELD_TYPE, TokenSource

sql_meta = MetaData()

student_register = Table(
    "PSD_ELEV3_CLEAN",
    sql_meta,
    Column("PERSON_ID", Integer),
    Column("ELEV3_VFRA", Date),
    Column("ELEV3_VTIL", Date),
    Column("INSTNR", Integer),
    Column("UDD", Integer, ForeignKey("D222112.UDD_DETAILS.UDD")),
    Column("UDEL", Integer),
    Column("REFERENCETID", Date),
    schema="D222112",
)

education_details = Table(
    "UDD_DETAILS",
    sql_meta,
    Column("UDD", Integer),
    Column("TEXT", Text),
    Column("UDD_UDDANNELSESNIVEAU", Integer),
    schema="D222112",
)


@dataclass
class EducationTokens(TokenSource):
    """
    .. attention::

        This source is mainly implemeted as a proof-of-concept for integrating data
        pulled from an SQL-database. Not much thought has been put into tokenization and
        data cleaning. We should probably also pull the data from a more thouroughly
        documented table.

    Tokens from the national student register.

    """

    name: str = "education"
    fields: List[FIELD_TYPE] = field(
        default_factory=lambda: [
            "START_END",
            "INST",
            "UDD",
            "UDEL",
            "LEVEL",
        ]
    )

    reference_year: int = 2020
    minimum_days_attended: int = 30
    start_year: int = 2008
    end_year: int = 2018

    @save_parquet(
        DATA_ROOT / "processed/sources/{self.name}/tokenized",
        on_validation_error="error",
    )
    def tokenized(self) -> dd.DataFrame:
        """
        Loads the indexed data, then melts the dataframe according to START_DATE, and
        END_DATE, resulting an EDU_START/EDU_END token beginning each sentence.

        The other variables are prtransformed to "EDU_<variable_name>_<value>" and
        sentences outside the date range are filtered out.

        Finally, the sentences is sorted by START_DATE.

        """

        ddf = self.indexed()

        def melt_and_sort(df: pd.DataFrame) -> pd.DataFrame:
            return df.melt(
                id_vars=["PERSON_ID"]
                + [x for x in self.field_labels() if x != "START_END"],
                value_vars=["START", "END"],
                var_name="START_END",
                value_name="START_DATE",
            ).sort_values(["PERSON_ID"])

        melted = (
            ddf.reset_index()
            .map_partitions(melt_and_sort)
            .set_index("PERSON_ID", sorted=True, divisions=ddf.divisions)
        )

        from ..ops import sort_partitions

        result = (
            melted.assign(
                START_END=lambda x: "EDU_" + x.START_END,
                INST=lambda x: "EDU_INST_" + x.INST,
                UDD=lambda x: "EDU_UDD_" + x.UDD,
                UDEL=lambda x: "EDU_UDEL_" + x.UDEL,
                LEVEL=lambda x: "EDU_LEVEL_" + x.LEVEL,
            )[["START_DATE", *self.field_labels()]]
            .loc[
                lambda x: (x.START_DATE.dt.year >= self.start_year)
                & (x.START_DATE.dt.year <= self.end_year)
            ]
            .pipe(sort_partitions, columns=["START_DATE"])
        )

        result = result.repartition("256MB")

        assert isinstance(result, dd.DataFrame)
        return result

    @save_parquet(
        DATA_ROOT / "interim/sources/{self.name}/indexed",
        on_validation_error="recompute",
    )
    def indexed(self) -> dd.DataFrame:
        """
        Loads the data from the oracle database.
        We need to the data by PERSON_ID in the SQL-statement, aswell as specify it with
        index_col in the read_sql call otherwise the index breaks.
        """
        stmt = (
            select(
                [
                    student_register.c.PERSON_ID,
                    student_register.c.ELEV3_VFRA,
                    student_register.c.ELEV3_VTIL,
                    student_register.c.INSTNR,
                    student_register.c.UDD,
                    student_register.c.UDEL,
                    education_details.c.UDD_UDDANNELSESNIVEAU,
                ]
            )
            .select_from(student_register.outerjoin(education_details))
            .where(
                and_(
                    extract("year", student_register.c.REFERENCETID)
                    == self.reference_year,
                    student_register.c.ELEV3_VTIL - student_register.c.ELEV3_VFRA
                    > self.minimum_days_attended,
                    extract("year", student_register.c.ELEV3_VFRA) <= self.end_year,
                    extract("year", student_register.c.ELEV3_VTIL) >= self.start_year,
                )
            )
            .order_by(student_register.c.PERSON_ID)
        )

        ddf = dd.read_sql(
            sql=stmt,
            con="oracle://@statprod",
            index_col="PERSON_ID",
            bytes_per_chunk="256MB",
        )

        assert ddf.index.name == "PERSON_ID"

        if self.downsample:
            ddf = self.downsample_persons(ddf)

        ddf = ddf.assign(
            ELEV3_VFRA=lambda x: dd.to_datetime(x.ELEV3_VFRA),
            ELEV3_VTIL=lambda x: dd.to_datetime(x.ELEV3_VTIL, errors="coerce").fillna(
                pd.Timestamp.max
            ),
            INSTNR=lambda x: x.INSTNR.astype("string"),
            UDD=lambda x: x.UDD.astype("string"),
            UDD_UDDANNELSESNIVEAU=lambda x: x.UDD_UDDANNELSESNIVEAU.astype("string"),
            UDEL=lambda x: x.UDEL.astype("string"),
        )

        ddf = ddf.rename(
            columns={
                "ELEV3_VFRA": "START",
                "ELEV3_VTIL": "END",
                "INSTNR": "INST",
                "UDD_UDDANNELSESNIVEAU": "LEVEL",
            }
        )

        assert isinstance(ddf, dd.DataFrame)
        return ddf
