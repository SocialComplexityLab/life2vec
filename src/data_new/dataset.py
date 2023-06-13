import json as json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Generic, Iterable, Optional, TypeVar

import h5py
import numpy as np
from torch.utils.data import ConcatDataset, Dataset

from src.tasks.base import Task

from .types import Background, PersonDocument, EncodedDocument

log = logging.getLogger(__name__)

T1 = TypeVar("T1", bound=Any)
T2 = TypeVar("T2", bound=Any)


class HDF5Dataset(Dataset, Generic[T1, T2]):
    """
    Stores and loads compressed records of arbitrary json data using the hdf5 format.
    The data is stored in a zero-padded numpy string array on disk. Since we are using
    compression, the zero-padding does not increase the storage reguired significantly.

    This class should be subclasses and the serialize/deserialize methods implenented.

    We allow for a transform argument, mapping the records stored in the dataset to
    be transformed to something else, for instance in the case of augmentation.

    """

    def __init__(
        self,
        file: Path,
        encoding: str = "utf-8",
        transform: Optional[Callable[[T1], T2]] = None,
    ):

        self.file = Path(file)
        assert self.file.suffix == ".hdf5"
        self.encoding = encoding
        self.transform = transform

    def _transform(self, x: T1) -> T2:
        """Apply the transform if defined."""
        if self.transform is None:
            return x
        else:
            return self.transform(x)

    def deserialize(self, x: str) -> T1:
        """Instantiate the record from the json string"""
        raise NotImplementedError

    def serialize(self, x: T1) -> str:
        """Serialize the instantiated record to a json string"""
        return json.dumps(x, separators=(",", ":"))

    def __len__(self) -> int:
        """Return the number of records"""
        with h5py.File(self.file, "r") as f:  # TODO: set "r" attribute
            return len(f["data"])

    def __getitem__(self, idx: int) -> T2:
        """Retrieve a record string from the hdf5 array, apply the deserialization,
        then finally apply the transform.
        """

        # Sometimes, the OS would throw an error on file access. This is solved by
        # retrying a few times
        retries = 5
        for i in range(retries):
            try:
                with h5py.File(self.file, "r") as f:
                    content = self.deserialize(f["data"][idx].decode(self.encoding))
                break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"Encountered {e} while opening {self.file} at index {i}")
                if i < retries - 1:
                    print("Retrying...")
                    time.sleep(1 + i * 5)
                else:
                    print("Aborting...")
                    raise e

        content = self._transform(content)
        return content

    def save_data(self, data: Iterable[T1]) -> None:
        """Saves records to the dataset"""

        out_array = np.array([self.serialize(x).encode(self.encoding) for x in data])
        self.file.parent.mkdir(exist_ok=True, parents=True)
        with h5py.File(self.file, "w") as f:
            f.create_dataset(
                "data", data=out_array, compression="gzip", compression_opts=9
            )


TaskT = TypeVar("TaskT", bound=Task)


class DocumentDataset(HDF5Dataset[PersonDocument, EncodedDocument[TaskT]]):
    """Dataset implementation for storing PersonDocuments as records. The augmentation
    and encoding is done using the transform parameter. Returns the encoded documents.

    
    :param file:
    :param encoding:
    :param transform:
    """

    def deserialize(self, x: str) -> PersonDocument:
        """Loads the person document from the json data."""
        data = json.loads(x)
        data["background"] = Background(**data["background"])
        return PersonDocument(**data)

    def serialize(self, x: PersonDocument) -> str:
        """Dumps the person document to json data"""
        return json.dumps(asdict(x), separators=(",", ":"))


class ShardedDocumentDataset(ConcatDataset, Generic[TaskT]):
    """Wrapper around :class:`torch.utils.data.ConcatDataset` for combining multiple
    document datasets in the case of sharded data.
    """

    def __init__(
        self,
        directory: Path,
        encoding: str = "utf-8",
        transform: Optional[Callable[[PersonDocument], EncodedDocument[TaskT]]] = None,
    ):

        self.directory = Path(directory)
        self.encoding = encoding
        self.transform = transform

        datasets = []
        for file_ in sorted(self.directory.glob("*.hdf5")):
            datasets.append(
                DocumentDataset(
                    file_,
                    encoding=self.encoding,
                    transform=self.transform,
                )
            )

        super().__init__(datasets)
