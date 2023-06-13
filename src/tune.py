import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from .callbacks import ReseedTrainDataLoader
from ray.air import session
from ray.tune.schedulers import ASHAScheduler

import ray
from ray import air, tune
from ray.tune import CLIReporter
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.skopt import SkOptSearch
from pathlib import Path
import logging

HOME_PATH = str(Path.home())

log = logging.getLogger(__name__)


def last_ckpt(dir_):
    ckpt_path = Path(HOME_PATH, dir_, "last.ckpt")
    if ckpt_path.exists():
        log.info("Checkpoint exists:\n\t%s" %str(ckpt_path))
        return None
    else:
        log.info("Checkpoint DOES NOT exists:\n\t%s" %str(ckpt_path))
        return None

def home_path():
    return HOME_PATH

try:
    OmegaConf.register_new_resolver("last_ckpt", last_ckpt)
except Exception as e:
    print(e)


def tune_hyperparameters(config, cfg):
    print("I AM HERE")
    cfg.model.hparams.hidden_size = config["hidden_size"]
    cfg.model.hparams.att_dropout = config["dropout"]
    cfg.model.hparams.fw_dropout = config["dropout"]
    cfg.model.hparams.dc_dropout = config["dropout"]
    cfg.model.hparams.emb_dropout = config["dropout"]
    cfg.model.hparams.n_layers = config["n_layers"]
    #cfg.model.hparams.bidirectional = config["bidir"]

    data = instantiate(cfg.datamodule, _convert_="all")
    model = instantiate(cfg.model, _convert_="all")
    tune_callback = TuneReportCallback({"loss": "val/loss",
                                        "mcc": "val/mcc_corrected", 
                                        "aul": "aul"}, 
                                        on="validation_end")

    print("ALL good here")
    ##TRAINER
    trainer = Trainer(callbacks = [ tune_callback], accelerator="gpu", devices=[7])#default_root_dir =  cfg.trainer["default_root_dir"],
                      #max_epochs = 6,
                      ##accumulate_grad_batches = 4,
                      #log_every_n_steps = 10,
                      #num_sanity_val_steps = 10,
                      #limit_train_batches = 1200,
                      #limit_val_batches = 2000,
                      #limit_test_batches = 12500,
                      #accelerator="gpu",
                      #evices = 1)
                      #callbacks = [ tune_callback],
                      #check_val_every_n_epoch =  1)
    print("STARTED")
    ##TRAINING
    trainer.fit(model, data)


@hydra.main(config_path="../conf", config_name="config")
def main(cfg):

    # Workaround for hydra breaking import of local package - don't need if running as module "-m src.train"
    # sys.path.append(hydra.utils.get_original_cwd())
    
    ##GLOBAL SEED
    seed_everything(cfg.seed)
    ray.init()
    ray_config = {"hidden_size": tune.qrandint(64,512,2),
                  "dropout": tune.quniform(0.0,0.5, 0.01),
                  "n_layers": tune.randint(1,5)} 
    reporter = CLIReporter(
            parameter_columns=["hidden_size", "dropout", "n_layers"],
            metric_columns=["aul", "mcc", "loss", "training_iteration"],
            max_report_frequency=100)

    train_fn_with_parameters = tune.with_parameters(tune_hyperparameters,
                                                    cfg=cfg)

    scheduler = ASHAScheduler(max_t=6,
                             grace_period=1,
                             metric="val/aul",
                             mode="max",
                             reduction_factor=2)
    search = ConcurrencyLimiter(SkOptSearch(metric="aul", mode="max"), 1)
    tuner = tune.Tuner(train_fn_with_parameters,
            tune_config=tune.TuneConfig(search_alg = search,
                                    scheduler=scheduler,
                                    num_samples=25),
            run_config=air.RunConfig(name="eos_rnn", progress_reporter=reporter, local_dir="/home/x90/odrev/projekter/PY000017_D/ray/"),
            param_space=ray_config)

    result = tuner.fit()


if __name__ == "__main__":
    main()
