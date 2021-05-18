"IMDB DataModule"
import argparse
import os
from pathlib import Path

import torch
from text_classifier.data.base_data_module import BaseDataModule, load_and_print_info
from torch.utils.data import Dataset, random_split
from torchtext.datasets import IMDB as TorchIMDB
from transformers import AutoTokenizer

DOWNLOADED_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded/imdb"


class IMDBDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings # shape: {input_ids: [], attention_mask}
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class IMDB(BaseDataModule):
    """
    IMDB DataModule
    """

    def __init__(self, args: argparse.Namespace = None) -> None:
        super().__init__(args)
        self.data_dir = DOWNLOADED_DATA_DIRNAME
        self.input_dims = None
        self.output_dims = (1,)
        self.mapping = {"neg": 0, "pos": 1}

    def prepare_data(self):
        "Download IMDB dataset"
        if not os.path.exists(DOWNLOADED_DATA_DIRNAME / "aclImdb"):
            TorchIMDB(self.data_dir)

    def setup(self, stage: str = None):
        "Split into train, val, test"
        train_texts, train_labels = read_imdb_split(self.data_dir / "aclImdb/train")
        test_texts, test_labels = read_imdb_split(self.data_dir / "aclImdb/test")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        train_encodings = tokenizer(train_texts, truncation=True, padding=True)
        test_encodings = tokenizer(train_texts, truncation=True, padding=True)
        # Create IMDB dataset
        self.data_test = IMDBDataset(test_encodings, test_labels)
        train_ds = IMDBDataset(train_encodings, train_labels)
        self.data_train, self.data_val = random_split(train_ds, [20000, 5000])

    def __repr__(self) -> str:
        """Print infor about the dataset"""
        basic = (
            f"IMDB Dataset\nNum classes: {len(self.mapping)}\nMapping: {self.mapping}\n"
        )
        data = f"Train/val/test sizes: {len(self.data_train), len(self.data_val), len(self.data_test)}"
        return basic + data


def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir / label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir is "neg" else 1)
    return texts, labels


if __name__ == "__main__":
    load_and_print_info(IMDB)
