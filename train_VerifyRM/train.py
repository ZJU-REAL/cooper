from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import Qwen2Model, Qwen2PreTrainedModel
import torch.nn as nn
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from typing import Optional, Union, List, Tuple
from transformers import TrainingArguments, Trainer
from datasets import Dataset
from functools import partial
import json
from transformers import AutoConfig
from transformers import DataCollatorWithPadding
from transformers import Qwen2ForSequenceClassification
import pandas as pd

model_path = "../models/Qwen/Qwen2.5-Math-1.5B-Instruct"
data_path = "../dataset/VerifyRM_training_data.parquet"


class BinaryRewardModelTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)
        loss_fct = BCEWithLogitsLoss()
        loss = loss_fct(logits, labels.float())
        return (loss, outputs) if return_outputs else loss


def apply_template(data_path):
    df = pd.read_parquet(data_path)
    filtered_data = []
    
    for _, item in df.iterrows():
        item_dict = item.to_dict()
        item_dict["prompt"] = f"<question>{item_dict['question']}</question>\n<reference_answer>{item_dict['reference_answer']}</reference_answer>\n<completion>{item_dict['completion']}</completion>"
        if len(item_dict["prompt"]) <= 8000:
            filtered_data.append(item_dict)
    return filtered_data

def process_function(data, tokenizer):
    tokenized_examples = tokenizer(data["prompt"], max_length=8192, truncation=True,padding=False)
    tokenized_examples["labels"] = data["label"]
    return tokenized_examples

def main():
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    data = apply_template(data_path)
    dataset = Dataset.from_list(data)
    tokenized_dataset = dataset.map(partial(process_function, tokenizer=tokenizer), batched=True)

    model = Qwen2ForSequenceClassification.from_pretrained(model_path, num_labels=1, trust_remote_code=True)

    train_args = TrainingArguments(
        output_dir="./checkpoints",      
        per_device_train_batch_size=128,  
        gradient_accumulation_steps=32,
        logging_steps=10,                
        save_strategy="epoch",           
        save_total_limit=3,              
        learning_rate=2e-5,              
        weight_decay=0.01,               
        num_train_epochs=3,
        bf16=True,
    )    

    trainer = BinaryRewardModelTrainer(model=model, 
                    args=train_args, 
                    train_dataset=tokenized_dataset, 
                    data_collator=DataCollatorWithPadding(tokenizer=tokenizer))

    trainer.train()

if __name__ == "__main__":
    main()
