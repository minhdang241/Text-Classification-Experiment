```
Text-Classification-Experiment on  master [?] via 🅒 base 
➜ tree   
.
├── data
│   └── raw
│       ├── metadata.toml
│       └── README.md
├── notebooks
│   └── 01-look-at-imdb.ipynb
├── README.md
├── text_classifier
│   ├── data
│   │   ├── base_data_module.py
│   │   ├── imdb.py
│   │   └── __init__.py
│   ├── __init__.py
│   ├── lit_models
│   │   ├── base.py
│   │   ├── __init__.py
│   │   └── transformers.py
│   ├── models
│   │   ├── distilBERT.py
│   │   └── __init__.py
│   └── utils.py
├── training
│   └── run_experiment.py
└── utils.py


```
The main codebase is splitted into 3 folders
`data` is a folder that we store our downloaded datasets. 
`text_classifier` is a Python package that we are developing and will deploy.
`training` is a Python package that is used to support developing `text_recognizer`.


### Data
There are three scopes of the code dealing with data: `DataModule`, `DataLoader`, `Dataset`
`DataModule` classes are responsible for a few things:
* Downloading raw data and/or generating synthetic data
* Processing data as needed to get i ready to go through Pytorch models
* Splitting data into train/val/test sets
* Specifying dimensions of the inputs
* Specifying information about the targets
* Specifying data augmentation transforms to apply in training

In the proecss of doing the above, `DataModule` classes make use of a couple of other classes:
1. They wrap underlying data in a `torch Dataset`. which returns individual data instances
2. They wrap the `torch Dataset` in a `torch DataLoader`, which samples batches, shffles their order, and delivers them to the GPU.

To avoid writing the same boilerplate for all the datasource, we define a base class `text_classifier.data.BaseDataModule` which in turn ingerits from `pl.LightningDataModule`. That enables us to use the data with Pytorch-Lightning `Trainer` and avoid common problems with distributed training.

### Models
Models are code that accept input, process it through layers of computations and produces an output.
Since we are using Pytorch, all of our models subclass `torch.nn.Module`.

### Lit Models
We use Pytorch-Lightning for training, which defines the `LightningModule` interface that handles the details of the learning algorithms including:
1. What loss should be computed from the output of the model and the label
2. Which optimizer should be used, with what learning rate

## Training
The `training/run_experiment.py` is a script that handles command-line parameters.

Here's an example:

```python3 training/run_experiment.py --model_class=MLP --data_class=MNIST --max_epochs=5 --gpus=1```

`model_class` and `data_class` are our own arguments, `max_epochs` and `gpus` are arugments automatically picked up from `pytorch_lightning.Trainer`. If you want to see more flags, see the doc https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags.