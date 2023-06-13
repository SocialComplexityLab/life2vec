from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, List, NewType, Optional, TypeVar

JSONSerializable = NewType("JSONSerializable", object)

if TYPE_CHECKING:
    from src.tasks.base import Task

_TaskT = TypeVar("_TaskT", bound="Task")


@dataclass
class PersonDocument:
    """Dataclass for defining the complete person document in a structured fashion"""

    person_id: int
    sentences: List[List[str]]
    abspos: List[int]
    age: List[float]
    timecut_pos: int  # What is this?
    segment: Optional[List[int]] = None
    background: Optional["Background"] = None
    shuffled: bool = False
    task_info: Optional[JSONSerializable] = None


@dataclass
class Background:
    """Defines the background information about a person"""

    origin: str
    gender: str
    birth_month: int
    birth_year: int

    @staticmethod
    def get_sentence(x: Optional["Background"]) -> List[str]:
        """Return sequence of tokens corresponding to this person. Implemented as
        classmethod since we can null the background in PersonDocument in case of
        unknown background.
        """

        if x is None:
            return 4 * ["[UNK]"]
        else:
            return [
                x.origin,
                x.gender,
                f"MONTH_{x.birth_month}",
                f"YEAR_{x.birth_year}",
            ]


class EncodedDocument(Generic[_TaskT]):
    """Generic class for encoded documents. Each task can then type-hint their
    specific encoding using a dataclass like

    .. code-block ::

        class MyTask:
            def encode_document(x: PersonDocument) -> "MyTaskEncodedDocument":
                return MyTaskEncodedDocument(target=1)

        @dataclass
        class MyTaskEncodedDocument(EncodedDocument[MyTask]):
            target: int

    """
