import json
import torch
import re
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torch.optim import AdamW
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

from jsonschema import validate, ValidationError
from tqdm import tqdm
import os
import evaluate
import pickle
import copy
from transformers import LogitsProcessor
from transformers import LogitsProcessorList
from transformers import LogitsProcessor
import torch
from typing import List
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.ast_dataset import ASTDataset


class ASTGenerator:
    def __init__(self, model_name="t5-small", parser=None, load_dir_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        if self.model_name=="t5-small":
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.parser = parser
        self.load_dir_path = load_dir_path
        my_tokens = [
            # Structural tags
            "[AST:SEG]", "[/AST:SEG]",
            "[CLASS]", "[/CLASS]",
            "[ATTRIBUTE]",
            "[METHOD]",
            "[CONTEXT]", "[/CONTEXT]",
            "[REQUIREMENT]", "[/REQUIREMENT]",
            "[HISTORY]", "[/HISTORY]",

            # Visibility
            "[PUBLIC]", "[PRIVATE]", "[PROTECTED]", "[PACKAGE]",

            # Multiplicity
            "[ONE]", "[MANY]", "[MOM]", "[OOM]", "[ZOO]",

            # Other tags
            "[TYPE]", "[LABEL]", "[NOLABEL]", "[NOHISTORY]",

            # Relation types (general)
            "[ASSOCIATION]", "[DEPENDENCY]",

            # Directional and specific relation tags
            "[RIGHT:EXTENSION]", "[LEFT:EXTENSION]",
            "[RIGHT:IMPLEMENTATION]", "[LEFT:IMPLEMENTATION]",
            "[LEFT:COMPOSITION]", "[RIGHT:COMPOSITION]",
            "[LEFT:AGGREGATION]", "[RIGHT:AGGREGATION]",
            "[LEFT:ASSOCIATION]", "[RIGHT:ASSOCIATION]",
            "[LEFT:DEPENDENCY]", "[RIGHT:DEPENDENCY]",

            # task tags
            "[TASK:CLASS_IDENTIFICATION]",
            "[TASK:ATTRIBUTE_IDENTIFICATION]",
            "[TASK:METHOD_IDENTIFICATION]",
            "[TASK:RELATION_IDENTIFICATION]",
            "[TASK:RELATION_CONTEXT]",
            "[TASK:ATTRIBUTE_SUGGESTION]",
            "[TASK:METHOD_SUGGESTION]",
            "[TASK:RELATION_SUGGESTION]",
        ]


        self.tokenizer.add_tokens(my_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))


        if os.path.exists(self.load_dir_path) and self.load_dir_path!=None:
            print("reload model from saved checkpoint...")
            if self.model_name=="t5-small":
                self.tokenizer = T5Tokenizer.from_pretrained(load_dir_path)
                self.model = T5ForConditionalGeneration.from_pretrained(load_dir_path).to(self.device)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(load_dir_path)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(load_dir_path).to(self.device)
        else:
            print("No checkpoint found.")




    def train(self, data_path, epochs=3, batch_size=4, lr=3e-4):
        os.makedirs(self.load_dir_path, exist_ok=True)
        dataset = ASTDataset(data_path, self.tokenizer)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=ASTDataset.collate_fn)

        optimizer = AdamW(self.model.parameters(), lr=lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, epochs * len(loader))

        current_epoch = 1
        if os.path.exists(self.load_dir_path) and self.load_dir_path!=None:
            epoch_file_path = os.path.join(self.load_dir_path, "current_epoch.txt")

            if not os.path.exists(epoch_file_path):
                with open(epoch_file_path, "w") as f:
                    f.write("1")
                current_epoch = self.get_current_epoch(epoch_file_path)
            else:
                current_epoch = self.get_current_epoch(epoch_file_path)
                current_epoch=current_epoch+1


        for epoch in range(current_epoch, epochs+1, 1):
            self.model.train()
            total_loss = 0
            progress = tqdm(loader, desc=f"Epoch {epoch}")
            for batch in progress:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                labels[labels == self.tokenizer.pad_token_id] = -100
                output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = output.loss
                total_loss += loss.item()

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.set_postfix(loss=loss.item())

            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch} - Loss: {avg_loss:.4f}")


            if (epoch) % 1 == 0:
                # Save model checkpoint
                self.save_model(self.load_dir_path, epoch)

            if epoch >= 100:
                break


    def train_kfold(self, data_path, epochs=3, batch_size=4, lr=3e-4, k=5):
        os.makedirs(self.load_dir_path, exist_ok=True)
        full_dataset = ASTDataset(data_path, self.tokenizer)
        folds_path = os.path.join(self.load_dir_path, "fold_indices.pkl")

        if not os.path.exists(folds_path):
            kfold = KFold(n_splits=k, shuffle=True, random_state=42)
            splits = list(kfold.split(full_dataset))
            with open(folds_path, "wb") as f:
                pickle.dump(splits, f)
        else:
            with open(folds_path, "rb") as f:
                splits = pickle.load(f)


        for fold, (train_idx, val_idx) in enumerate(splits):
            print(f"\n🔁 Fold {fold + 1}/{k}")
            fold_model_path = os.path.join(self.load_dir_path, f"_fold{fold + 1}")
            os.makedirs(fold_model_path, exist_ok=True)
            epoch_file_path = os.path.join(fold_model_path, "current_epoch.txt")

            current_epoch = 1
            if not os.path.exists(epoch_file_path):
                with open(epoch_file_path, "w") as f:
                    f.write("0")
            current_epoch = self.get_current_epoch(epoch_file_path) + 1

            model_file = os.path.join(fold_model_path, "pytorch_model.bin")
            if os.path.exists(fold_model_path):
                print(f"📦 Loading model from {fold_model_path}")
                if self.model_name=="t5-small":
                    self.tokenizer = T5Tokenizer.from_pretrained(fold_model_path)
                    self.model = T5ForConditionalGeneration.from_pretrained(fold_model_path).to(self.device)
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(fold_model_path)
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(fold_model_path).to(self.device)
                model = copy.deepcopy(self.model)
            else:
                model = copy.deepcopy(self.model)

            train_subset = Subset(full_dataset, train_idx)
            val_subset = Subset(full_dataset, val_idx)

            train_loader = torch.utils.data.DataLoader(
                train_subset, batch_size=batch_size, shuffle=True, collate_fn=ASTDataset.collate_fn
            )
            val_loader = torch.utils.data.DataLoader(
                val_subset, batch_size=batch_size, shuffle=False, collate_fn=ASTDataset.collate_fn
            )

            optimizer = AdamW(model.parameters(), lr=lr)
            scheduler = get_linear_schedule_with_warmup(
                optimizer, 0, epochs * len(train_loader)
            )

            for epoch in range(current_epoch, epochs + 1):
                model.train()
                total_loss = 0
                progress = tqdm(train_loader, desc=f"[Fold {fold + 1}] Epoch {epoch}")
                for batch in progress:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    labels[labels == self.tokenizer.pad_token_id] = -100

                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = output.loss
                    total_loss += loss.item()

                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    progress.set_postfix(loss=loss.item())

                avg_loss = total_loss / len(train_loader)
                print(f"[Fold {fold + 1}] Epoch {epoch} - Train Loss: {avg_loss:.4f}")

                self.save_model(fold_model_path, epoch)

            self.evaluate_on_validation(model, val_loader)



    def evaluate_on_validation(self, model, val_loader):
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                labels[labels == self.tokenizer.pad_token_id] = -100

                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                total_val_loss += output.loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"🔍 Validation Loss: {avg_val_loss:.4f}")

    def evaluate_bleu_on_data(self,data_path, batch_size=4,  max_length=128, num_samples=None):

        metric = evaluate.load("bleu")
        self.model.eval()
        dataset = ASTDataset(data_path, self.tokenizer)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, collate_fn=ASTDataset.collate_fn
        )
        total = 0

        for batch in loader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    num_beams=4,

                )

            decoded_preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            for pred, label in zip(decoded_preds, decoded_labels):
                print("\nPrediction:\n", pred)
                print("Target:\n", label)
                print("-" * 40)
                total += 1
                if num_samples and total >= num_samples:
                    break

            metric.add_batch(predictions=decoded_preds, references=[[l] for l in decoded_labels])

            if num_samples and total >= num_samples:
                break

        result = metric.compute()
        score = result.get('bleu', 0.0)
        print("BLEU score on dataset:", score)
        return score





    def generate(self, input_text,  max_length=100):
        self.model.eval()
        input_enc = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        output_ids = self.model.generate(input_enc["input_ids"],
                       max_length=100,
                       do_sample=True,
                       num_beams=4,
                       eos_token_id=self.tokenizer.convert_tokens_to_ids("[/AST:SEG]"),
                                         decoder_start_token_id=self.tokenizer.pad_token_id
                       )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


    def generateStructuredOutput(self, input_text,  max_length=100):
        logits_processor = LogitsProcessorList([
            T5GrammarConstrainedLogitsProcessor(self.tokenizer, parser)
            ])

        input_enc = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(
            input_enc["input_ids"],
            max_length=100,
            do_sample=True,
            num_beams=4,
            eos_token_id=self.tokenizer.convert_tokens_to_ids("[/AST:SEG]"),
            decoder_start_token_id=self.tokenizer.pad_token_id,
            logits_processor=logits_processor,
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def save_model(self, path, epoch):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self.save_current_epoch(f"{path}/current_epoch.txt", epoch)

    def save_current_epoch(self, path, epoch):
        with open(path, "w") as f:
            f.write(str(epoch))

    def get_current_epoch(self, path):
        try:
            with open(path, "r") as f:
                return int(f.read().strip())
        except FileNotFoundError:
            return 1