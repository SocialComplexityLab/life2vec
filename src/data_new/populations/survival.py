from dataclasses import dataclass
from multiprocessing import resource_tracker
from pathlib import Path
from functools import reduce

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd

from ..decorators import save_pickle
from ..serialize import DATA_ROOT
from .base import DataSplit, Population


# HYDRA_FULL_ERROR=1 python -m src.prepare_data +data_new/population=survival_set target=\${data_new.population}
dateparser = lambda x: pd.to_datetime(x, format = '%d%b%Y:%X',  errors='coerce')

@dataclass
class SurvivalSubPopulation(Population):
    base_population: Population
    name: str = "survival"
    earliest_birthday: str = "01-01-1951"
    latest_birthday: str = "31-12-1981"

    target_path: Path =  DATA_ROOT / "rawdata" / "eos" / "PRETTY_LINES_v3.csv"
    period_start: str = "01-01-2016"
    period_end: str = "31-12-2020"
    
    def __post_init__(self) -> None:
        self._earliest_birthday = pd.to_datetime(self.earliest_birthday, format="%d-%m-%Y")
        self._latest_birthday = pd.to_datetime(self.latest_birthday, format="%d-%m-%Y")
        self._period_start = pd.to_datetime(self.period_start, format="%d-%m-%Y")
        self._period_end = pd.to_datetime(self.period_end, format="%d-%m-%Y")

    @save_pickle(
        DATA_ROOT / "processed/populations/{self.name}/population",
        on_validation_error="error",
    )
    def population(self) -> pd.DataFrame:
        """Loads the joined sub_population and target dataframe 
           Inherits columns from TARGET:
            * EVENT_FINAL_DATE: date of the event (if event does not happen, it is set to the :attr:`period_end`)
            * TARGET: outcome feature (1 - event happend, 0 otherwise)
        """
        population = self.sub_population()
        target = self.target()
        assert population.shape[0] == target.shape[0]

        result = population.join(target)
        assert isinstance(result, pd.DataFrame)
        return result

    @save_pickle(
        DATA_ROOT / "interim/populations/{self.name}/population",
        on_validation_error="error",
    )
    def sub_population(self) -> pd.DataFrame:
        """
        Return the FILTERED population as a pandas dataframe with an index named PERSON_ID.
        Filters according to:
            * NEW Birthday requirement

        Includes the RES_ORIGIN column in the population data.
        """
        base_population = self.base_population.population()

        bd_out_of_bound = (
            base_population.loc[
                lambda x: (x.BIRTHDAY >= self._latest_birthday)
                | (x.BIRTHDAY <= self._earliest_birthday)
            ]
            .index.unique()
            .values
        )

        ids_to_remove = set(bd_out_of_bound)

        result = (
            base_population.loc[lambda x: ~x.index.isin(ids_to_remove)]
        )

        assert isinstance(result, pd.DataFrame)
        return result


    @save_pickle(
        DATA_ROOT / "processed/populations/{self.name}/target",
        on_validation_error="error",
    )
    def target(self) -> pd.DataFrame:

        data = dd.read_csv(self.target_path, blocksize="64MB", encoding="latin", sep = ";",
                 usecols=["PERSON_ID", "EVENT_CAUSE_FINAL", "EVENT_FINAL_DATE", "QUALITY_INFORMATION_FINAL", "KILDE_FINAL", "NUMBER_EVENTS_PERSON"],
                 parse_dates=["EVENT_FINAL_DATE"], date_parser=dateparser)


        population_ids =  self.sub_population().index.to_numpy()
        
        result = data.loc[lambda x: (x.PERSON_ID.isin(population_ids)) & \
                                    (x.EVENT_FINAL_DATE >= self._period_start) & \
                                    (x.EVENT_FINAL_DATE <= self._period_end) & \
                                    (x.EVENT_CAUSE_FINAL.isin(["Doed"]))] \
                    .groupby("PERSON_ID").agg({"EVENT_FINAL_DATE": "max"})
        result = result.compute()
        result["TARGET"] = 1
        positive_ids = result.index.to_numpy()
        negative_ids = np.array(list(filter(lambda x: x not in positive_ids, population_ids)))


        assert population_ids.shape[0] == positive_ids.shape[0] + negative_ids.shape[0]

        temp_result_1 = data.loc[lambda x: (x.PERSON_ID.isin(negative_ids)) & \
                                    (x.EVENT_FINAL_DATE >= self._period_start) & \
                                    (x.EVENT_FINAL_DATE <= self._period_end) & \
                                    (x.EVENT_CAUSE_FINAL.isin(["Udvandret", "Forsvundet"]))] \
                    .groupby("PERSON_ID").agg({"EVENT_FINAL_DATE": "max"}).compute()
        temp_result_1["TARGET"] = 0
        print("Calculated TEMP1", temp_result_1.shape[0], negative_ids.shape[0] - temp_result_1.shape[0] )


        negative_ids = np.array(list(filter(lambda x: x not in temp_result_1.index.to_numpy(), negative_ids)))
        temp_result_2 = pd.DataFrame({"PERSON_ID": negative_ids, 
                                   "EVENT_FINAL_DATE":  np.full_like(negative_ids, 
                                                                    fill_value = dateparser(self._period_end),
                                                                    dtype = result.dtypes["EVENT_FINAL_DATE"]),
                                    "TARGET": np.full_like(negative_ids, 
                                                           fill_value= 0,
                                                           dtype = result.dtypes["TARGET"])}).set_index("PERSON_ID")

        print("Calculated TEMP2", temp_result_2.shape[0])


        result = pd.concat([result, temp_result_1, temp_result_2]).sort_index()

        cat_type = pd.api.types.CategoricalDtype(categories=[0,1], ordered=False)
        result["TARGET"] = result["TARGET"].astype(cat_type)

        assert result.shape[0] == population_ids.shape[0]
        assert isinstance(result, pd.DataFrame)
        return result

    @save_pickle(DATA_ROOT / "processed/populations/{self.name}/data_split")
    def data_split(self) -> DataSplit:
        """Split data based on subpopulation"""
        base_splits = self.base_population.data_split()
        current_population = self.population().index.to_numpy()

        train_ids = np.intersect1d(base_splits.train ,current_population, assume_unique=True)
        val_ids = np.intersect1d(base_splits.val ,current_population, assume_unique=True)
        test_ids = np.intersect1d(base_splits.test ,current_population, assume_unique=True)

        assert current_population.shape[0] == train_ids.shape[0] + test_ids.shape[0] + val_ids.shape[0]
        
        return DataSplit(
            train=train_ids,
            val=val_ids,
            test=test_ids,
        )

    def prepare(self) -> None:
        """
        Prepares the data by calling the :meth:`population` and :meth:`data_split`.
        """
        self.sub_population()
        self.target()
        self.population()
        self.data_split()


