import argparse
from typing import Any, Dict

import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, data_config: Dict[str, Any], args: argparse.Namespace = None):
        super().__init__()
        self.data_config = data_config
        self.num_classes = len(data_config["mapping"])
        self.idx_2_label = {v: k for k, v in data_config["mapping"].items()}
        self.embedding 

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        # labels = batch["labels"]
        outputs = self.model(input_ids, attention_mask)
        return outputs

    def predict(self, sentence: str) -> torch.Tensor:
        tokenized_input = AutoTokenizer(sentence)
        output = self.model(
            tokenized_input["input_ids"], tokenized_input["attention_mask"]
        )
        logit = output.logits
        pred_idx = torch.argmax(logit, dim=-1)
        return self.idx_2_label[pred_idx]