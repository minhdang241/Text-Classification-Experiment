import torch
import argparse
import torch.nn as nn
from typing import Any, Dict
from transformers import DistilBertForSequenceClassification, AutoTokenizer



class DistilBERTClassifier(nn.Module):
    def __init__(self, data_config: Dict[str, Any], args: argparse.Namespace = None):
        super().__init__()
        self.model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased"
        )
        self.data_config = data_config
        self.idx_2_label = {v: k for k, v in data_cofig["mapping"].items()}
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = model(input_ids, attention_mask)
        return outputs
    
    def predict(self, sentence: str) -> torch.Tensor:
        tokenized_input = AutoTokenizer(sentence)
        output = model(tokenized_input["input_ids"], tokenized_input["attention_mask"])
        logit = output.logits
        pred_idx = torch.argmax(logit, dim=-1)
        return self.idx_2_label[pred_idx]
