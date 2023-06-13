
import warnings
from pytorch_lightning import Callback, Trainer, profiler
import importlib
from typing import Any


def silence_warnings() -> None:

    ignore_patterns = [
        "The dataloader, [^,]*, does not have many workers",
        "`LightningModule.configure_optimizers` returned `None`, this fit will run with no optimizer",
        "GPU available but not used.",
        "torch.qr is deprecated in favor of torch.linalg.qr and will be removed in a future PyTorch release."
    ]

    for ptn in ignore_patterns:
        warnings.filterwarnings("ignore", message=ptn, category=UserWarning)
    

class SilenceWarnings(Callback):
    def on_init_start(self, trainer: Trainer) -> None:
        silence_warnings()


def import_from(module: str, name: str) -> Any:
    module_ = importlib.import_module(module)
    return getattr(module_, name)


import json

class DetailedProfiler(profiler.SimpleProfiler):

    def summary(self) -> str:
        with open("recorded_durations.json", "w") as f:
            json.dump(self.recorded_durations, f)
        return super().summary()




from collections.abc import Mapping, Iterable
from dataclasses import is_dataclass, fields


def stringify(obj):

    if isinstance(obj, str):
        return f"'{obj}'"

    if not is_dataclass(obj) and not isinstance(obj, (Mapping, Iterable)):
        return str(obj)

    start, end = f'{type(obj).__name__}(', ')'  # dicts, lists, and tuples will re-assign this

    if isinstance(obj, dict):
        start, end = '{}'

    elif isinstance(obj, list):
        start, end = '[]'
    
    elif isinstance(obj, tuple):
        start = '('

    return f'{start}...{end}'
