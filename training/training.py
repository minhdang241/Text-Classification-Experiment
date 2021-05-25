import argparse
import importlib
import pytorch_lightning as pl
from text_classifier import lit_models

import kfp
import kfp.components as components
import kfp.dsl as dsl
from kfp.components import InputPath, OutputPath

def train_model(train, dev):
    # Preprocess
    
    # Train
