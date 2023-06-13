from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DataSplit:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


class Population:
    """
    Base class for cohort definitions. In order to implement a new cohort, you should
    subclass this class and implement :meth:`population` and :meth:`data_split`.
    """

    name: str

    def population(self) -> pd.DataFrame:
        """
        Return the population as a pandas dataframe with an index named PERSON_ID.
        This dataframe will be left joined on the sentences, so any additional columns
        will be included in sentence data of the corpus. Should atleast include BIRTHDAY
        and GENDER (M/F).
        """
        raise NotImplementedError

    def data_split(self) -> DataSplit:
        """Returns the training splits"""
        raise NotImplementedError

    def prepare(self) -> None:
        """
        Prepares the data by calling the :meth:`population` and :meth:`data_split`.
        """
        self.population()
        self.data_split()
