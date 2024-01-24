# Using Sequences of Life-events to Predict Human Lives

[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.10118620.svg)](https://zenodo.org/doi/10.5281/zenodo.10118620)

This repository contains code for the [Using Sequences of Life-events to Predict Human Lives](https://www.nature.com/articles/s43588-023-00573-5) (life2vec) paper.
We have **only one webpage** related to the project ([life2vec.dk](https://life2vec.dk)), and we do not have any specialized Facebook, Tweeter accounts, etc.
For more information refer to the [FAQ](https://life2vec.dk).


### Basic Implementation of life2vec
*Closer to the end of January 2024, we are planning to release a pipeline using the dummy data, so you can run the code on your machine*. We will keep keep this repository as is; a basic implemenetation of the model is going to be published in the following repository: [carlomarxdk/life2vec-light](https://github.com/carlomarxdk/life2vec-light)

### Source Code

This repository contains scripts and several notebooks for data processing, life2vec training, statistical analysis, and visualization. The model weights, experiment logs, and associated model outputs can be obtained in accordance with the rules of [Statistics Denmark's Research Scheme](https://www.dst.dk/en/TilSalg/Forskningsservice/Dataadgang).

Paths (e.g., to data, or model weights) were **redacted** before submitting scripts to GitHub.

### Overall Structure

We use [Hydra](https://hydra.cc/docs/intro/) to run the experiments. The `/conf` folder contains configs for the experiments:
1. `/experiment` contains configuration `yaml` for pretraining and finetuning,
2. `/tasks` contain the specification for data augmentation in MLM, SOP, etc.,
3. `/trainer` contains configuration for logging (not used) and multithread training (not used),
4. `/data_new` contains configs for data loading and processing,
5. `/datamodule` contains configs that specify how data should be loaded to PyTorch and PyTorch Lightning
6. `callbacks.yaml` specifies the configuration for the PyTorch Lightning Callbacks ,
7. `prepare_data.yaml` can be used to run data preprocessing.

The `/analysis` folder contains `ipynb` notebooks for post-hoc evaluation:
1. `/embedding` contains the analysis of the embedding spaces,
2. `/metric` contains notebooks for the model evaluation,
3. `/visualisation` contains notebooks for the visualisation of spaces,
4. `/tcav` includes TCAV implementation,
5. `/optimization` hyperparameter tuning.

The source folder, `/src`, contains the data loading and model training codes. Due to the specifics of the `hydra` package. Here is the overview of the `/src` folder:
1. The `/src/data_new` contains scripts to preprocess data as well as prepare data to load into the PyTorch or PyTorch Lightning,
2. The `/src/models` contains the implementation of baseline models,
3. The `/src/tasks` include code specific to the particular task, aka MLM, SOP, Mortality Prediction, Emigration Prediction, etc.
4. `/src/tranformer` contains the implementation of the life2vec model:
      1. In `performer.py`, we overwrite the functionality of the `performer-pytorch` package,
      2. In `cls_model.py`, we have an implementation of the finetuning stage for the binary classification tasks (i.e. early mortality and emigration),
      3. In `hexaco_model.py`, we have an implementation of the finetuning stage for the **personality nuance prediction** task,
      4. `models.py` contains the code for the life2vec **pretraining** (aka the base life2vec model),
      5. The `transformer_utils.py` contains the implementation of custom modules, like losses, activation functions, etc.
      6. The `metrics.py` contains code for the custom metric,
      7. The `modules.py`, `attention.py`, `att_utils.py`, and `embeddings.py` contain the implementation of modules used in the transformer network (aka life2vec encoders).

Scripts such as `train.py`, `test.py`, `tune.py`, and `val.py` used to run a particular stage of the training, while `prepare_data.py` was used to run the data processing (see below the example).


### Run the script
To run the code, you would use the following commands:

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

### Another Code Contributors
1. Søren Mørk Hartmann.

### How to cite

**Nature Computational Science**
```bibtex
@article{savcisens2023using,
   title={Using sequences of life-events to predict human lives},
   ISSN={2662-8457},
   url={http://dx.doi.org/10.1038/s43588-023-00573-5},
   DOI={10.1038/s43588-023-00573-5},
   journal={Nature Computational Science},
   publisher={Springer Science and Business Media LLC},
   author={Savcisens, Germans and Eliassi-Rad, Tina and Hansen, Lars Kai and Mortensen, Laust Hvas and Lilleholt, Lau and Rogers, Anna and Zettler, Ingo and Lehmann, Sune},
   year={2023},
   month=dec}
```

**ArXiv Preprint**
```bibtex
@article{savcisens2023using,
  title={Using Sequences of Life-events to Predict Human Lives},
  DOI = {arXiv:2306.03009},
  author={Savcisens, Germans and Eliassi-Rad, Tina and Hansen, Lars Kai and Mortensen, Laust and Lilleholt, Lau and Rogers, Anna and Zettler, Ingo and Lehmann, Sune},
  year={2023}
}
```
**Code**
```bibtex
@misc{life2vec_code,
  author = {Germans Savcisens},
  title = {Official code for the "Using Sequences of Life-events to Predict Human Lives" paper},
  note = {GitHub: SocialComplexityLab/life2vec},
  year = {2023},
  howpublished = {\url{https://doi.org/10.5281/zenodo.10118621}},
}
```
