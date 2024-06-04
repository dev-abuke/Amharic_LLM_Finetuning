import numpy as np

import os
import sys
import subprocess

from datasets import load_dataset, load_metric, Dataset, DatasetDict, concatenate_datasets, load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorWithPadding,
    Trainer,
    LlamaTokenizer,
    TrainingArguments,
)
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import LlamaForCausalLM, GenerationConfig

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

class Tokenizer:
    def __init__(self):
        self.HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
        self.model_name = "Samuael/llama-2-7b-tebot-amharic"  # Replace with your chosen model name
        self.checkpoint = "iocuydi/llama-2-amharic-3784m"
        # The commit hash is needed, because the model repo was rearranged after this commit (files -> finetuned/files),
        self.commit_hash = "04fcac974701f1dab0b8e39af9d3ecfce07b3773"
        self.cache_dir= './cache'
        # "rasyosef/bert-amharic-tokenizer"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        self.garii_tokenizer = LlamaTokenizer.from_pretrained(
            self.checkpoint, 
            revision=self.commit_hash, 
            cache_dir=self.cache_dir
        )
        self.llama_model = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            load_in_8bit=True,
            device_map="auto",
            cache_dir=self.cache_dir, # optional
        )
        # this is the model we want:
        garii_model = PeftModel.from_pretrained(
            self.llama_model, 
            self.checkpoint,
            revision =self.commit_hash, 
            cache_dir= self.cache_dir
        )
        # Prepare the model for int8 training (optional, but helps reduce memory usage)
        model = prepare_model_for_kbit_training(garii_model)

        # Define LoRA configuration
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none"
        )

        # Apply LoRA to the model
        lora_model = get_peft_model(model, lora_config)

        # Tokenize example input
        input_text = "ሰላም እንዴት ነህ?"
        inputs = self.tokenizer(input_text, return_tensors="pt")

        # Define training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2, 
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
        )

        # Define trainer
        self.trainer = Trainer(
            model=lora_model,
            args=training_args,
            train_dataset=dataset['train'],  # Assuming `dataset` is your prepared dataset
            eval_dataset=dataset['validation']
        )
    def hugging_face_login(self):
        # Login to Hugging Face Hub
        # !huggingface-cli login
        huggingface_bin_path = "/home/user/.local/bin"
        os.environ["PATH"] = f"{huggingface_bin_path}:{os.environ['PATH']}"
        subprocess.run(["huggingface-cli", "login", "--token", self.HUGGINGFACE_TOKEN])
    def train(self):
        self.trainer.train()

    def save_model(self):
        self.trainer.save_model("./finetuned_model")
    
    def save_tokenizer(self):
        self.tokenizer.save_pretrained("./tokenizer_save")