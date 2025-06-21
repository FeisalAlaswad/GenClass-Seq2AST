import json
import torch
import re
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

class ASTDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_input_len=256, max_output_len=512):
        with open(data_file) as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Input with explicit instruction for generating AST
        field_to_prefix = {
            "Class Identification": "[TASK:CLASS_IDENTIFICATION] Extract classes:",
            "Attribute Identification": "[TASK:ATTRIBUTE_IDENTIFICATION] Extract attributes for:",
            "Method Identification": "[TASK:METHOD_IDENTIFICATION] Extract methods for:",
            "Relation Identification": "[TASK:RELATION_IDENTIFICATION] Extract relations for:",
            "Class Relation Identification": "[TASK:RELATION_CONTEXT] Extract all relations for the context:",
            "Attribute Suggestion": "[TASK:ATTRIBUTE_SUGGESTION] Suggest attributes for:",
            "Method Suggestion": "[TASK:METHOD_SUGGESTION] Suggest methods for:",
            "Relation Suggestion": "[TASK:RELATION_SUGGESTION] Suggest relations for:"
            }



        if item['task'] == "Class Identification":
                    history_text = ", ".join(item['history_context']) if isinstance(item['history_context'], list) else item['history_context']
                    input_text = (
                        f"{field_to_prefix[item['task']]}\n\n"
                        f"[CONTEXT]\n{item['scenario_context']}\n[/CONTEXT]\n\n"
                        f"[HISTORY]\n{history_text}\n[/HISTORY]\n\n"
                        f"[REQUIREMENT]\n{item['input_sentence']}\n[/REQUIREMENT]"
                        )
        elif item['task'] == "Attribute Identification":
                    input_text = (
                        f"{field_to_prefix[item['task']]}\n\n"
                        f"[CONTEXT]\n{item['scenario_context']}\n[/CONTEXT]\n\n"
                        f"[CLASS]\n{item['class']}\n[/CLASS]\n\n"
                        f"[REQUIREMENT]\n{item['input_sentence']}\n[/REQUIREMENT]"
                        )
        elif item['task'] == "Method Identification":
                    input_text = (
                        f"{field_to_prefix[item['task']]}\n\n"
                        f"[CONTEXT]\n{item['scenario_context']}\n[/CONTEXT]\n\n"
                        f"[CLASS]\n{item['class']}\n[/CLASS]\n\n"
                        f"[REQUIREMENT]\n{item['input_sentence']}\n[/REQUIREMENT]"
                        )
        elif item['task'] == "Relation Identification":
                    input_text = (
                        f"{field_to_prefix[item['task']]}\n\n"
                        f"[CONTEXT]\n{item['scenario_context']}\n[/CONTEXT]\n\n"
                        f"[CLASS]\n{item['class']}\n[/CLASS]\n\n"
                        f"[REQUIREMENT]\n{item['input_sentence']}\n[/REQUIREMENT]"
                        )
        elif item['task'] == "Context Relation Identification":
                    input_text = (
                        f"{field_to_prefix[item['task']]}\n\n"
                        f"[CONTEXT]\n{item['scenario_context']}\n[/CONTEXT]\n\n"
                        f"[CLASS]\n{item['class']}\n[/CLASS]\n\n"
                        )
        elif item['task'] == "Attribute Suggestion":
                    input_text = (
                        f"{field_to_prefix[item['task']]}\n\n"
                        f"[CONTEXT]\n{item['scenario_context']}\n[/CONTEXT]\n\n"
                        f"[CLASS]\n{item['class']}\n[/CLASS]\n\n"
                        )
        elif item['task'] == "Method Suggestion":
                    input_text = (
                        f"{field_to_prefix[item['task']]}\n\n"
                        f"[CONTEXT]\n{item['scenario_context']}\n[/CONTEXT]\n\n"
                        f"[CLASS]\n{item['class']}\n[/CLASS]\n\n"
                        )
        elif item['task'] == "Relation Suggestion":
                    input_text = (
                        f"{field_to_prefix[item['task']]}\n\n"
                        f"[CONTEXT]\n{item['scenario_context']}\n[/CONTEXT]\n\n"
                        f"[CLASS]\n{item['class']}\n[/CLASS]\n\n"
                        )

        output_text = item['output_ast_tagged']  # Tokenized output (e.g., "[CLASS] Transaction ...")

        input_enc = self.tokenizer(
            input_text, max_length=self.max_input_len, truncation=True, padding='max_length', return_tensors='pt'
        )
        output_enc = self.tokenizer(
            output_text, max_length=self.max_output_len, truncation=True, padding='max_length', return_tensors='pt'
        )

        return {
            'input_ids': input_enc['input_ids'].squeeze(0),
            'attention_mask': input_enc['attention_mask'].squeeze(0),
            'labels': output_enc['input_ids'].squeeze(0),
            'original': item
        }

    @staticmethod
    def collate_fn(batch):
        """Custom collate function to handle padding."""
        pad_token_id = 0  # T5 pad token ID
        labels_pad_id = -100  # Ignore index for loss calculation

        # Pad input sequences
        input_ids = [item['input_ids'] for item in batch]
        max_input_len = max(len(input_id) for input_id in input_ids)
        input_ids_padded = [F.pad(input_id, (0, max_input_len - len(input_id)), value=pad_token_id) for input_id in input_ids]

        # Pad attention masks
        attention_masks = [item['attention_mask'] for item in batch]
        attention_masks_padded = [F.pad(att_mask, (0, max_input_len - len(att_mask)), value=0) for att_mask in attention_masks]

        # Pad labels
        labels = [item['labels'] for item in batch]
        max_output_len = max(len(label) for label in labels)
        labels_padded = [F.pad(label, (0, max_output_len - len(label)), value=labels_pad_id) for label in labels]

        return {
            'input_ids': torch.stack(input_ids_padded),
            'attention_mask': torch.stack(attention_masks_padded),
            'labels': torch.stack(labels_padded),
            'original': [item['original'] for item in batch]
        }