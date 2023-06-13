import yaml
from argparse import ArgumentParser
from typing import List, Dict
import glob

import os


def parse_config(path: str):
    """"""
    config = yaml.load(open(path, "r", newline=""), Loader=yaml.FullLoader)
    parser = ArgumentParser()
    for key, value in config.items():
        # print(key, value, type(value))
        if type(value) == bool:
            parser.add_argument("--%s" % key, dest=key, action="store_true")
            parser.add_argument("--no-%s" % key, dest=key, action="store_false")
        else:
            parser.add_argument("--%s" % key, type=type(value))
    args = parser.parse_args(serialize_config(config))
    # print(args)
    return args


def serialize_config(config: Dict) -> List[str]:
    """"""
    # Get an empty list for serialized config:
    serialized_config = []

    for key, value in config.items():
        # Append key:
        if str(value) == "True":
            serialized_config.append("--" + key)
            continue
        elif str(value) == "False":
            serialized_config.append("--no-" + key)
            continue

        serialized_config.append("--" + key)
        # Append value:
        if isinstance(value, str) or isinstance(value, float) or isinstance(value, int):
            serialized_config.append(str(value))

        elif isinstance(value, List):
            serialized_config += [str(val) for val in value]

        elif isinstance(value, bool):
            serialized_config.append(bool(value))
        elif isinstance(value, type(None)):

            serialized_config.append(None)
        else:
            raise ValueError(f"Invalid value in config file: {value}")
    # print(serialized_config)
    return serialized_config


def search_for_checkpoint(path, version):
    # * means all if need specific format then *.csv
    files = glob.glob("%s/%s*" % (path, version))
    if len(files) == 0:
        return None
    return max(files, key=os.path.getctime)


def mapping2d(value, width):
    """Map value to 2D Grid coordinates"""
    col = value % width
    row = value // width
    return (col, row)


def rindex(mylist, value):
    """list.index(value), but from the end of the list"""
    return len(mylist) - mylist[::-1].index(value) - 1
