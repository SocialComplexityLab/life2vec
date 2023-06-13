from typing import List, Optional, Sequence

import dask.dataframe as dd
import numba
import numpy as np
import pandas as pd


def _sort_using_index(data: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    return data.set_index(columns, append=True).sort_index().reset_index(columns)


def sort_partitions(data: dd.DataFrame, columns: Sequence[str]) -> dd.DataFrame:
    result = data.map_partitions(_sort_using_index, columns=columns)
    assert isinstance(result, dd.DataFrame)
    return result


def concat_sorted(
    dfs: List[dd.DataFrame], columns: Optional[List[str]], partition_size: str = "256MB"
) -> dd.DataFrame:

    assert all(df.known_divisions for df in dfs)

    result = (
        dd.multi.concat(dfs, interleave_partitions=True)  # this combines
        .pipe(sort_partitions, columns=columns)
        .repartition(partition_size=partition_size)
    )

    assert isinstance(result, dd.DataFrame)
    return result


def concat_columns_dask(
    data: dd.DataFrame, columns: Optional[List[str]] = None, sep: str = " "
) -> dd.Series:
    result = data.map_partitions(concat_columns, columns=columns, sep=sep)
    assert isinstance(result, dd.Series)
    return result


def concat_columns(
    data: pd.DataFrame, columns: Optional[List[str]] = None, sep: str = " "
) -> pd.Series:

    if columns is not None:
        data = data[columns]

    old_index = data.index.copy()

    data = data.reset_index(drop=True)
    is_na = data.isna().all(axis=1)

    out: np.ndarray = np.ndarray(data.shape[0], dtype=object)
    out[is_na] = None

    stacked = data.stack()
    indices = stacked.index.get_level_values(0).to_numpy()
    values = stacked.to_numpy().astype(np.str_)

    res = _concat_sub(indices, values, sep=sep)
    out[~is_na] = res
    return pd.Series(out, index=old_index, dtype="string")


@numba.jit(nopython=True)
def _concat_sub(indices: np.ndarray, values: np.ndarray, sep: str = " ") -> List[str]:

    # Start each sequence
    (starts,) = np.nonzero(np.diff(indices))
    starts += 1

    out = []

    # If empty, all indices are the same, return all values concatenated
    if len(starts) == 0:
        out.append(sep.join(values))
        return out

    idx_prev = 0
    for idx in starts:
        out.append(sep.join(values[idx_prev:idx]))
        idx_prev = idx

    out.append(sep.join(values[starts[-1] :]))
    return out
