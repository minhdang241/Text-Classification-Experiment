"UIT-VSFC DataModule"
import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from torchtext.datasets import IMDB as TorchIMDB
from transformers import AutoTokenizer

from text_classifier.data.base_data_module import BaseDataModule, load_and_print_info

DOWNLOADED_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded/uit_vsfc"

class UIT_VSFCTranformerDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings  # shape: {input_ids: [], attention_mask: []}
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class UIT_VSFCTransformer(BaseDataModule):
    """
    IMDB DataModule for transformer
    """

    def __init__(
        self, model_checkpoint: str, args: argparse.Namespace = None, dataset_paths=None
    ) -> None:
        super().__init__(args)
        if dataset_paths:
            self.train = dataset_paths.get("train")
            self.dev = dataset_paths.get("dev")
            self.test = dataset_paths.get("test")
        else:
            self.train = f"{DOWNLOADED_DATA_DIRNAME}/train.csv"
            self.dev = f"{DOWNLOADED_DATA_DIRNAME}/dev.csv" 
            self.test = f"{DOWNLOADED_DATA_DIRNAME}/test.csv"

        self.input_dims = None
        self.output_dims = (1,)
        self.mapping = {"neg": 0, "neutral": 1, "pos": 2}
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def prepare_data(self):
        "Download IMDB dataset"
        pass

    def setup(self):
        # Load dataset
        train_df = pd.read_csv(self.train)
        val_df = pd.read_csv(self.dev)
        test_df = pd.read_csv(self.test)

        train_texts, train_labels = list(train_df["sentences"]), list(train_df["labels"])
        val_texts, val_labels = list(val_df["sentences"]), list(val_df["labels"])
        test_texts, test_labels = list(test_df["sentences"]), list(test_df["labels"])

        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True)
        val_encodings = self.tokenizer(val_texts, truncation=True, padding=True)
        test_encodings = self.tokenizer(train_texts, truncation=True, padding=True)
        # Create IMDB dataset
        self.data_train = UIT_VSFCTranformerDataset(train_encodings, train_labels)
        self.data_val = UIT_VSFCTranformerDataset(val_encodings, val_labels)
        self.data_test = UIT_VSFCTranformerDataset(test_encodings, test_labels)

    def __repr__(self) -> str:
        """Print infor about the dataset"""
        basic = f"UIT-VSFC Dataset\nNum classes: {len(self.mapping)}\nMapping: {self.mapping}\n"
        data = (
            f"Train/val/test sizes: {len(self.data_train), len(self.data_val), len(self.data_test)}"
        )
        return basic + data


if __name__ == "__main__":
    load_and_print_info(UIT_VSFCTransformer)
