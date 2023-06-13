import inspect
import json
import logging
import pickle
from dataclasses import asdict, dataclass, field, is_dataclass
from functools import wraps
from inspect import BoundArguments
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Protocol,
    TypeVar,
    Union,
    cast,
    get_args,
)

import dask
import dask.dataframe as dd
import pandas as pd

from src.utils import stringify

DATA_ROOT = Path.home() / ... / "data"
T = TypeVar("T")


log = logging.getLogger(__name__)


class ValidationError(Exception):
    pass


# Should be easily serialized using pickle
_PRIMITIVE = Union[int, float, str, bool]


# I only want to serialize 'simple' types. I check asdict() for
# dataclasses and throw the error on the rest
def _jsonify(x: Any) -> Union[Dict[str, Any], List[Any], _PRIMITIVE]:

    if x is None:
        return None
    elif isinstance(x, (list, tuple)):
        return [_jsonify(y) for y in x]
    elif isinstance(x, BoundArguments):
        return _jsonify(x.arguments)
    elif isinstance(x, dict):
        return {k: _jsonify(y) for k, y in x.items() if not k.endswith("__")}
    elif isinstance(x, Path):
        # We only check paths relative to data root folder
        return x.relative_to(DATA_ROOT).as_posix()
    elif is_dataclass(x):
        return _jsonify(asdict(x))
    else:
        assert isinstance(x, get_args(_PRIMITIVE))
        return x  # type: ignore


class PathFunc(Protocol):
    def __call__(self, f: Callable[..., T], ba: BoundArguments) -> Path:
        ...


@dataclass
class Serializer(Generic[T]):
    """
    Base class for defining serializers. The serializers are not really meant to be
    used directly in practice, instad use the :meth:`decorate` method or the
    decorators defined in :mod:`src.data_new.serialize`. In order to define new
    serializers, implement the :meth:`load_result` and :meth:`save_result` methods.

    In order to serialize the function the function call `f(*args, **kwargs)`, we call
    using the :meth:`get` method instead

    .. code-block ::

        serializer: Serializer
        result = serializer.get(f, *args, **kwargs)

    :ivar path: Either a literal path (str or pathlib.Path) object or a callable that
        return a path when called with the function, f, and arguments as an
        :class:`inspect.BoundArgument` instance.

    """

    path: Union[Path, str, PathFunc]

    def load_result(self, f: Callable[..., T], ba: BoundArguments) -> T:
        """Loads the result"""
        raise NotImplementedError

    def save_result(self, f: Callable[..., T], ba: BoundArguments, result: T) -> None:
        """Saves the result"""
        raise NotImplementedError

    def get(self, f: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Handles the overall serialization pipeline, and returns the return value of
        :code:`f(*args, **kwargs)`
        """

        sig = inspect.signature(f)
        ba = sig.bind(*args, **kwargs)

        try:
            result = self.validate_and_load(f, ba)
            log.debug("Serialized result found.")
            return result
        except (FileNotFoundError, EOFError):
            log.debug("Serialized result not found.")
        except ValidationError as e:
            log.debug("Serialized result does not validate.")
            self.handle_validation_error(f, ba, e)

        computed_result = self.compute(f, ba)
        self.save_result(f, ba, computed_result)
        self.save_validation_info(f, ba)

        call_string = (
            f.__name__
            + "("
            + ", ".join(f"{k}={stringify(x)}" for k, x in ba.arguments.items())
            + ")"
        )
        log.info("Computed result for %s", call_string)

        return self.validate_and_load(f, ba)

    def decorate(self, f: Callable[..., T]) -> Callable[..., T]:
        """
        Allows the Serializer to be used as a decorator ie. instead of

        .. code-block::

            serializer = Serializer(path)

            def f_impl(args):
                ...

            def f(args):
                return serializer.get(f_impl, args)

        We can write

        .. code-block::

            @Serializer(path)
            def f(args):
                ...

        """

        @wraps(f)
        def wrapped(*args: Any, **kwargs: Any) -> T:
            return self.get(f, *args, **kwargs)

        return wrapped

    def get_path(self, f: Callable[..., T], ba: BoundArguments) -> Path:
        """Resolves the :attr:`self.path` attribute to a literal path"""
        # Determining path
        if isinstance(self.path, Path):
            return self.path
        elif isinstance(self.path, str):
            return Path(self.path)
        elif callable(self.path):
            path_ = self.path(f, ba)
            assert isinstance(path_, Path)
            return path_
        else:
            raise ValueError

    def compute(self, f: Callable[..., T], ba: BoundArguments) -> T:
        """Computes f(...)"""

        return f(*ba.args, **ba.kwargs)

    def validate_and_load(self, f: Callable[..., T], ba: BoundArguments) -> T:
        """Tries to validate the stored result against the provided arguments.
        Raises an ValidationError if the result exists, but failes to validate.
        """

        if not self.validates(f, ba):
            raise ValidationError("Result found, but calling parameters does not match stored parameters.")

        return self.load_result(f, ba)

    def validates(self, f: Callable[..., T], ba: BoundArguments) -> bool:
        """Returns whether a stored result could be validated."""
        with open(self.get_path(f, ba) / "_arguments.json") as f_:
            stored_arguments = json.load(f_)

        return bool(_jsonify(ba) == stored_arguments)

    def save_validation_info(self, f: Callable[..., T], ba: BoundArguments) -> None:
        """Saves validation info for validation the result for future calls"""
        with open(self.get_path(f, ba) / "_arguments.json", "w") as f_:
            json.dump(_jsonify(ba), f_, indent=4)

    def handle_validation_error(
        self, f: Callable[..., T], ba: BoundArguments, e: ValidationError
    ) -> None:
        """Implement this method to alter how validation errors are handled"""
        raise e


@dataclass
class ParquetSerializer(Serializer[dd.DataFrame]):

    verify_index: bool = True
    on_validation_error: Literal["error", "recompute"] = "error"
    parquet_kwargs: Dict[str, Any] = field(default_factory=lambda: {})

    def load_result(
        self, f: Callable[..., dd.DataFrame], ba: BoundArguments
    ) -> dd.DataFrame:

        result = dd.read_parquet(self.get_path(f, ba), calculate_divisions=True)

        for field_ in result.select_dtypes("category").columns:
            log.debug(
                "Found categorical column %s. Attempting to set categories.", field_
            )
            categories = result[field_].head(1).cat.categories
            result[field_] = result[field_].cat.set_categories(categories)
            assert result[field_].cat.known

        assert isinstance(result, dd.DataFrame)

        return result

    def save_result(
        self, f: Callable[..., dd.DataFrame], ba: BoundArguments, result: dd.DataFrame
    ) -> None:

        path = self.get_path(f, ba)
        parquet_kwargs = {
            "engine": "pyarrow-dataset",
            "compression": "gzip",
            "allow_truncated_timestamps": True,
            "coerce_timestamps": "us",
        }
        parquet_kwargs.update(self.parquet_kwargs)

        for field_ in result.select_dtypes("category").columns:
            assert result[field_].cat.known

        if self.verify_index:
            # Verify that the index is properly monotinic, and cleanly divided across
            # partitions, since some operations may break these properties, and we
            # generally rely on them for operations with eg. .map_partition. These 
            # checks can be a bit slow, however are worth it imo since it may save
            # some debugging down the line.

            _, is_monotonic = dask.compute(
                result.to_parquet(path, compute=False, **parquet_kwargs),
                result.index.is_monotonic,
            )
            assert is_monotonic, "Index is not monotonic."
            
            # Additional test that the non-unique index is divided across partitions 
            # cleanly
            t = dd.read_parquet(path, calculate_divisions=True)
            n = t.npartitions
            for i in range(n-1):
                a = t.get_partition(i).tail(1).index.item()
                b = t.get_partition(i+1).head(1).index.item()
                assert a != b, "Index values are not divided cleanly across partitions."

        else:
            result.to_parquet(path, **parquet_kwargs)

    def handle_validation_error(
        self, f: Callable[..., T], ba: BoundArguments, e: ValidationError
    ) -> None:
        if self.on_validation_error == "error":
            raise e
        elif self.on_validation_error == "recompute":
            pass


@dataclass
class HDFSerializer(Serializer[dd.DataFrame]):
    verify_index: bool = True
    on_validation_error: Literal["error", "recompute"] = "error"
    hdf_kwargs: Dict[str, Any] = field(default_factory=lambda: {})

    def load_result(
        self, f: Callable[..., dd.DataFrame], ba: BoundArguments
    ) -> dd.DataFrame:

        result = dd.read_hdf(self.get_path(f, ba), calculate_divisions=True)

        for field_ in result.select_dtypes("category").columns:
            log.debug(
                "Found categorical column %s. Attempting to set categories.", field_
            )
            categories = result[field_].head(1).cat.categories
            result[field_] = result[field_].cat.set_categories(categories)
            assert result[field_].cat.known

        assert isinstance(result, dd.DataFrame)

        return result


    def save_result(
        self, f: Callable[..., dd.DataFrame], ba: BoundArguments, result: dd.DataFrame
    ) -> None:

        path = self.get_path(f, ba)
        hdf_kwargs = {
            "chunksize": 250000,
            "mode": "w",
            "format": "table",
            "complevel": 1,
            "complib": "zlib"
        }
        hdf_kwargs.update(self.hdf_kwargs)

        for field_ in result.select_dtypes("category").columns:
            assert result[field_].cat.known

        if self.verify_index:
            # Verify that the index is properly monotinic, and cleanly divided across
            # partitions, since some operations may break these properties, and we
            # generally rely on them for operations with eg. .map_partition. These 
            # checks can be a bit slow, however are worth it imo since it may save
            # some debugging down the line.

            _, is_monotonic = dask.compute(
                result.to_hdf(path, compute=False, **hdf_kwargs),
                result.index.is_monotonic,
            )
            assert is_monotonic, "Index is not monotonic."
            
            # Additional test that the non-unique index is divided across partitions 
            # cleanly
            t = dd.read_hdf(path)
            n = t.npartitions
            for i in range(n-1):
                a = t.get_partition(i).tail(1).index.item()
                b = t.get_partition(i+1).head(1).index.item()
                assert a != b, "Index values are not divided cleanly across partitions."

        else:
            result.to_parquet(path, **hdf_kwargs)

    def handle_validation_error(
        self, f: Callable[..., T], ba: BoundArguments, e: ValidationError
    ) -> None:
        if self.on_validation_error == "error":
            raise e
        elif self.on_validation_error == "recompute":
            pass

@dataclass
class PickleSerializer(Serializer[T]):

    on_validation_error: Literal["error", "recompute"] = "error"

    def load_result(self, f: Callable[..., T], ba: BoundArguments) -> T:
        with open(self.get_path(f, ba) / "result.pkl", "rb") as f_:
            result = pickle.load(f_)
        return cast(T, result)

    def save_result(self, f: Callable[..., T], ba: BoundArguments, result: T) -> None:
        path = self.get_path(f, ba)
        path.mkdir(exist_ok=True, parents=True)
        with open(path / "result.pkl", "wb") as f_:
            pickle.dump(result, f_)

    def handle_validation_error(
        self, f: Callable[..., T], ba: BoundArguments, e: ValidationError
    ) -> None:
        if self.on_validation_error == "error":
            raise e
        elif self.on_validation_error == "recompute":
            pass


@dataclass
class TSVSerializer(Serializer[pd.DataFrame]):

    on_validation_error: Literal["error", "recompute"] = "error"

    def load_result(
        self, f: Callable[..., pd.DataFrame], ba: BoundArguments
    ) -> pd.DataFrame:
        return pd.read_csv(self.get_path(f, ba) / "result.tsv", sep="\t")

    def save_result(
        self, f: Callable[..., pd.DataFrame], ba: BoundArguments, result: pd.DataFrame
    ) -> None:
        path = self.get_path(f, ba) / "result.tsv"
        path.parent.mkdir(exist_ok=True, parents=True)

        result.to_csv(path, sep="\t")

    def handle_validation_error(
        self, f: Callable[..., T], ba: BoundArguments, e: ValidationError
    ) -> None:
        if self.on_validation_error == "error":
            raise e
        elif self.on_validation_error == "recompute":
            pass
