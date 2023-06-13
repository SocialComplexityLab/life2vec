import logging
from inspect import BoundArguments
from pathlib import Path
from typing import Any, Callable, Protocol, Type, TypeVar

from .serialize import ParquetSerializer, PickleSerializer, Serializer, TSVSerializer

log = logging.getLogger(__name__)

T = TypeVar("T")


# Used for typing
class SerializerDecorator(Protocol):
    def __call__(self, f: Callable[..., T]) -> Callable[..., T]:
        ...


# Used for typing
class SerializerDecoratorConstructor(Protocol):
    def __call__(self, path: Path, **kwargs: Any) -> SerializerDecorator:
        ...


def make_decorator(
    serializer_cls: Type[Serializer[T]],
) -> SerializerDecoratorConstructor:
    """Create a method decorator based on a serializer using path interpolation."""

    def decorator(path: Path, **kwargs: Any) -> SerializerDecorator:
        def path_func(f: Callable[..., T], ba: BoundArguments) -> Path:
            path_string = Path(path).as_posix()
            log.debug("Path string: %s", path_string)
            arguments = ba.arguments.copy()
            path_string = eval('f"""' + path_string + '"""', dict(), arguments)
            log.debug("Interpolated string: %s", path_string)
            return Path(path_string)

        return serializer_cls(path=path_func, **kwargs).decorate

    return decorator


save_parquet = make_decorator(ParquetSerializer)
"""Parquet decorator"""

save_pickle = make_decorator(PickleSerializer)
"""Pickle decorator"""

save_tsv = make_decorator(TSVSerializer)
"""TSV decorator"""
