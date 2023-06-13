### Source Code 

This repository contains scripts and several notebooks for the  data processing, life2vec training, statistical analysis, and visualization. The model weights, experiment logs, and associated model outputs can be obtained in accordance with the rules of [Statistics Denmark's Research Scheme](https://www.dst.dk/en/TilSalg/Forskningsservice/Dataadgang). 

Paths (e.g. to data, or model weights) were **redacted** before submitting scripts to the CodeOcean platform.


### Overall Structure

We use [Hydra](https://hydra.cc/docs/intro/) to run the experiments. The `/conf` folder contain configs for the experiments:
1. `/experiment` contains configuration `yaml` for pretraining and finetuning,
2. `/tasks` contain the specification for data augmentation in MLM, SOP, CLS etc.,
3. `/trainer` contains configuration for logging (not used) and multithread training (not used),
4. `/data_new` contains configs for data loading and processing,
5. `/datamodule` contains configs that specify how data should be loaded to PyTorch and PyTorch Lightning
6. `callbacks.yaml` specifies configuration for the PyTorch Lightning Callbacks ,
7. `prepare_data.yaml` can be used to run data preprocessing.

The `/analysis` folder contains `ipynb` notebooks for post-hoc evaluation:
1. `/embedding` contains the analysis of the embedding spaces,
2. `/metric` containt notebooks for the model evaluation
3. `/visualisation` contains notebooks for the visualisation of spaces. 

The source folder, `/src`, contains the codes related to the data loading and model training. Due to specifics of the `hydra` package, this folder includes have TCAV implementation (in `/src/analysis/tcav`) and hyperparameter tuning (in `/src/analysis/hyperparameter`). Here is the overview of the `/src` folder:
1. The `/src/data_new` contains scripts to preprocess data as well as prepare data to load into the PyTorch or PyTorch Lightning,
2. The `/src/models` contains the implementation of baseline models,
3. The `/src/tasks` include code specific to the particular task, aka MLM, SOP, Mortality Prediction, Emigration Prediction etc.
4. `/src/tranformer` contains the implementation of the life2vec model:
      1. In `performer.py`, we overwrite the functionality of the `performer-pytorch` package,
      2. In `cls_model.py`, we have implementation of the finetuning stage (aka life2vec for personality, mortality, emigration),
      3. `models.py` contains the code for the life2vec pretraining (aka the base life2vec model),
      4. The `transformer_utils.py` contains the implementation of custom modules, like losses, ac tivation functions etc.
      5. The `metrics.py` contains code for the custom metric,
      6. The `modules.py`, `attention.py`, `att_utils.py`, `embeddings.py` contain the implementation of modules used in the transformer-network (aka life2vec encoders).

Scripts such as `train.py`, `test.py`, `tune.py`, `val.py` used to run a particular stage of the training; while `prepare_data.py` used to run the data processing (see below the example).


### Run the script
To run the code you would use the following commands: 

```
# run the pretraining:
HYDRA_FULL_ERROR=1 python -m src.train experiment=pretrain trainer.devices=[7]

# finetuning of the hyperparamaters (for the pretraining)
HYDRA_FULL_ERROR=1 python -m src.train experiment=pretrain_optim

# assemble general dataset (GLOBAL_SET)
HYDRA_FULL_ERROR=1 python -m src.prepare_data +data_new/corpus=global_set target=\${data_new.corpus}

# assemble dataset for the mortality prediction task (SURVIVAL_SET)
HYDRA_FULL_ERROR=1 python -m src.prepare_data +data_new/population=survival_set target=\${data_new.population}


# assemble labour source
python -m src.prepare_data +data_new/sources=labour target=\${data_new.sources}

# run emigration finetuning
HYDRA_FULL_ERROR=1 python -m src.train experiment=emm trainer.devices=[0] version=0.01
```

## How to cite 
**Research Square Preprint**
```bibtex
@article{savcisens2023using,
  title={Using Sequences of Life-events to Predict Human Lives},
  author={Savcisens, Germans and Eliassi-Rad, Tina and Hansen, Lars Kai and Mortensen, Laust and Lilleholt, Lau and Rogers, Anna and Zettler, Ingo and Lehmann, Sune},
  year={2023}
}
```