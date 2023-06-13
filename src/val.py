import re
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
import sys
from pathlib import Path
import logging

HOME_PATH = str(Path.home())

log = logging.getLogger(__name__)


def last_ckpt(dir_):
    ckpt_path = Path(HOME_PATH, dir_, "best.ckpt")
    if ckpt_path.exists():
        print("CHECKPOINT EXISTS:", str(ckpt_path))
        return str(ckpt_path)
    else:
        print("CHECKPOINT DOES NOT EXISTS:", str(ckpt_path))
        return None

OmegaConf.register_new_resolver("last_ckpt", last_ckpt)

@hydra.main(config_path="../conf", config_name="config")
def main(cfg):

    # Workaround for hydra breaking import of local package - don't need if running as module "-m src.train"
    # sys.path.append(hydra.utils.get_original_cwd())
    
    ##GLOBAL SEED
    seed_everything(cfg.seed)
    
    data = instantiate(cfg.datamodule, _convert_="all")
    data.setup()
    ##MODEL
    model = instantiate(cfg.model, _convert_="all")

    ##TRAINER
    trainer: Trainer = instantiate(cfg.trainer, _convert_="all")
    trainer.logger.log_hyperparams(cfg)

    ##TRAINING
    trainer.validate(model, data, ckpt_path = cfg.ckpt_path)


if __name__ == "__main__":
    main()