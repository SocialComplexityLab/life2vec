"""
A given data compoment can be prepared using the "prepare_data.py" script
This script is configured using :ref:`hydra`, and works by instantiating what is in
the :attr:`target` package of the configuration, and calling its :meth:`prepare` method.
We can do this using Hydra command line overrides in various ways, depending on how the 
compoenents are configured. If the componet is configured using groups, we can select 
the group of interest and place it in the :attr:

.. code-block :: 
    bash
    python -m src.prepare_data +<path_to_config_folder>=<config_name> target=\${<path_to_config_folder>}

For instance, if we want to prepare the global_set corpus located in 
conf/data_new/corpus/global_set.yaml, we can write

.. code-block :: 
    bash
    python -m src.prepare_data +data_new/corpus=global_set target=\${data_new.corpus}


Since the sources are configured in a mapping, we can use variable interpolation. For
instance, we can prepare the labour tokens by writing 

.. code-block :: 
    bash

    python -m src.prepare_data target=\${data_new.sources.labour} # the \ escapes the $ sign


The data is prepared using a local dask cluster, as configured in the `client` package. 
The script provides some additional flags
    
* `interact=true` will drop you into an interactive shell after the component has
  been prepared. Useful for interacting with prepared compenents.

* `single_threaded=true` will use the single threaded scheduler for dask. This is
  useful for debugging, as otherwise using pdb.set_trace() will crash the program.

"""

from dask.distributed import Client
import hydra
from hydra.utils import instantiate

import os
from omegaconf import OmegaConf
import dask
import pandas as pd
from datetime import datetime


import logging
log = logging.getLogger(__name__)

@hydra.main(config_path="../conf", config_name="prepare_data", version_base=None)
def main(cfg):
    
    target = instantiate(cfg.target, _convert_="all")
    
    if cfg.single_threaded:
        dask.config.set(scheduler="single-threaded") 
        target.prepare()
    else:
        client: Client = instantiate(cfg.client)
        log.info("Monitor progress at: %s", client.dashboard_link)
        with client:
            target.prepare()

    if cfg.interact:
        import code
        code.interact(local=locals())


if __name__ == "__main__":
    main()
