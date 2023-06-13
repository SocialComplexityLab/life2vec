from collections import UserList
from dataclasses import dataclass
import enum
from pathlib import Path
from typing import List

import dask.dataframe as dd
import numpy as np
import pandas as pd

from ..decorators import save_pickle
from ..serialize import DATA_ROOT
from .base import DataSplit, Population

from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.ndimage import convolve1d

USECOLS_CON = ['HH','EM','EX','AG','CO','OP', "SDO", "SVOa", "RISK", "CRTi", "CRTr", "D"]
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
class HEXACOSubPopulation(Population):
    base_population: Population
    name: str = "hexaco"

    target_path: Path =  DATA_ROOT / "rawdata" / "psy" / "psy_personid.csv"

    @save_pickle(
        DATA_ROOT / "processed/populations/{self.name}/population",
        on_validation_error="error",
    )
    def population(self) -> pd.DataFrame:
        """Loads the joined sub_population and target dataframe 
           Inherits columns from TARGET
        """
        population = self.sub_population()
        target = self.target()
        assert population.shape[0] == target.shape[0]

        result = population.join(target)
        assert isinstance(result, pd.DataFrame)
        return result

    @save_pickle(
        DATA_ROOT / "interim/populations/{self.name}/target",
        on_validation_error="error",
    )
    def preprocess_target(self) -> pd.DataFrame:
        """Load and preprocess TARGET file"""

        usecols = ["PERSON_ID", "t1_participated__0___no__1___min"] + USECOLS_CON + HEXACO_COLS
        data = dd.read_csv(self.target_path, encoding="latin-1", blocksize="64MB", usecols=usecols)
        result = data[(~data["PERSON_ID"].isna()) & \
                    (data["t1_participated__0___no__1___min"] > 0) & \
                    (data["t1_participated__0___no__1___min"] < 3)].compute() \
                    .drop(["t1_participated__0___no__1___min"], axis=1) \
                    .dropna(axis = 0, how="any", subset= USECOLS_CON)

        ## keep only hexaco questions
        usecols = ["PERSON_ID"] + HEXACO_COLS + USECOLS_CON
        result = result[usecols]
        result = result.set_index("PERSON_ID", verify_integrity=True)
        ### add weights

        assert isinstance(result, pd.DataFrame)
        return result

    @save_pickle(
        DATA_ROOT / "interim/populations/{self.name}/population",
        on_validation_error="error",
    )
    def sub_population(self) -> pd.DataFrame:
        """
        Return the FILTERED population as a pandas dataframe with an index named PERSON_ID.
        Includes the RES_ORIGIN column in the population data.
        """
        # 1. Extract PERSON_IDs used for the PRETRAINING
        base_population = self.base_population.population()
        # 2. Extract PERSON_IDs that exist in the PSY Dataset
        ids_to_keep = self.preprocess_target().index.values.tolist()

        result = (
            base_population.loc[lambda x: x.index.isin(ids_to_keep)]
        )
        assert isinstance(result, pd.DataFrame)
        return result

    @save_pickle(
        DATA_ROOT / "interim/populations/{self.name}/target_qt",
        on_validation_error="error",
    )
    def target_quantiles(self) -> dict:
        """Returns quantiles (33% and 66%) for each Personality Trait (computed on Train Split)"""
        raw_trg = self.preprocess_target()
        train_ids = self.data_split().train.tolist()

        tmp_trg = raw_trg[raw_trg.index.isin(train_ids)]
        result = {}

        for col in USECOLS_CON:
            temp = np.quantile(tmp_trg[col].values, [0.33, 0.66])
            result[col] = temp
        
        assert isinstance(result, dict)
        return result


    @save_pickle(
        DATA_ROOT / "processed/populations/{self.name}/target",
        on_validation_error="error",
    )
    def target(self) -> pd.DataFrame:
        """Returns dataframe with Targets filtered by PERSON_IDs and DIGITIZED"""
        qt = self.target_quantiles()
        population_ids =  self.sub_population().index.to_numpy()
        flt_trg = self.preprocess_target()
        result = flt_trg[flt_trg.index.isin(population_ids)]

        result = result.sort_index()
 
        
        assert result.shape[0] == population_ids.shape[0]
        assert isinstance(result, pd.DataFrame)
        return result

    @save_pickle(
        DATA_ROOT / "processed/populations/{self.name}/column_ids",
        on_validation_error="error",
    )
    def column_ids(self) -> dict:
        """Save a dictionary that storex ids of columns: name -> id"""
        usecols = self.population().columns
        result = dict()
        for i, c in enumerate(usecols):
            result[c] = i
        assert isinstance(result, dict)
        return result


    @save_pickle(DATA_ROOT / "processed/populations/{self.name}/data_split")
    def data_split(self) -> DataSplit:
        """Split data based on subpopulation"""
        base_splits = self.base_population.data_split()
        current_population = self.sub_population().index.to_numpy()

        train_ids = np.intersect1d(base_splits.train , current_population, assume_unique=True)
        val_ids = np.intersect1d(base_splits.val , current_population, assume_unique=True)
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
        # 1. First we process the TARGET DATASET
        self.preprocess_target()
        # 2. Subpopulation: we keep only people who are in the preprocessed target dataset
        self.sub_population()
        # 3. We calculate data splits
        self.data_split()
        # 4. We process targets
        self.target()
        # 5. Make population
        self.population()
        # 6. Save column IDs
        self.column_ids()

