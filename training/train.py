import os
import torch
import numpy as np
import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
# import datasets
from utils import download_url, process_raw_dataset



class Trainer():
    def __init__(self, args, log):
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.device = args.device
        self.eval_every = args.eval_every
        self.path = os.path.join(args.save_dir, 'checkpoint')
        self.save_dir = args.save_dir
        self.log = log

    def train(self, model, train_dataloader, valid_dataloader):
        model.to(self.device)
        optim = AdamW(model.parameters(), lr=self.lr)
        global_idx = 0
        best_scores = {'Acc': -1.0, 'F1': -1.0}
        for epoch_num in self.num_epochs:
            with torch.enable_grad(), tqdm(total=len(train_dataloader.dataset)) as progress_bar:
                for batch in train_dataloader:
                    optim.zero_grad()
                    model.train() # put the model in training mode, default: evaluating mode (only apply for huggingface models)
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    outputs = model(input_ids, attention_mask, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    optim.step() # update model's weights
                    progress_bar.update(len(batch))
                    progress_bar.set_postfix(epoch=epoch_num, CELoss=loss)
                    
                    # make a evaluation
                    if global_idx % self.eval_every == 0:
                        # evaluate the model
                        self.evaluate(model, valid_dataloader, return_preds=True)
                        pass
                    global_idx += 1
    
    def evaludate(self, model, valid_dataloader, return_preds=False):
        model.eval() # put the model in evaluating mode, default: evaluating mode (only apply for huggingface models)
        results = dict()
        preds_list = []
        labels = []
        with torch.no_grad(), tqdm(total=len(valid_dataloader.dataset)) as progress_bar:
            for batch in valid_dataloader:
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                outputs = model(input_ids, attention_mask)
                logits = outputs.logits
                pred_indices = torch.argmax(logits, dim=-1)
                preds_list.append(pred_indices)
                labels.append(batch["labels"])
                progress_bar.update(len(batch))
        
        preds_list = torch.cat(preds_list).cpu().numpy()
        labels = torch.cat(labels).cpu().numpy()
        accuracy = np.mean(preds_list == labels, axis=-1)
        
        if return_preds:
            return pred_indices, accuracy
        else:
            return accuracy
    
    def save(self, model):
        model.save_pretrained(self.path)
    
    # def predict(self, model, tokenizer, input_text, data_dict):
    #     model.eval()
    #     with torch.no_grad():
    #         tokenized_input = tokenizer(input_text)
    #         output = model(tokenized_input["input_ids"], tokenized_input["attention_mask"])
    #         logit = output.logits
    #         pred_id = torch.argmax(logit, axis=-1).squeeze()[0]
    #         label = data_dict.idx_2_label[pred_id]
    #     return label

def get_dataset():
    pass

def main():
    fpath = download_url("http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", "./data/downloaded/raw")
    process_raw_dataset(fpath)
    # create dataloader for training and validation
    


if __name__ == "__main__":
    main()
