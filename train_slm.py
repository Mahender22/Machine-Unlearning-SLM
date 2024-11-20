# train_slm.py

import os
import pandas as pd
# from datasets import Dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split
from config import SLM_MODEL_NAME, DEVICE, DATA_DIR, SLM_MODEL_DIR

def load_data():
    with open(os.path.join(DATA_DIR, 'positive_examples.txt'), 'r', encoding='utf-8') as f:
        positive_examples = f.readlines()

    with open(os.path.join(DATA_DIR, 'negative_examples.txt'), 'r', encoding='utf-8') as f:
        negative_examples = f.readlines()

    texts = [text.strip() for text in positive_examples + negative_examples]
    labels = [1] * len(positive_examples) + [0] * len(negative_examples)
    return texts, labels

def train_slm():
    texts, labels = load_data()
    data = pd.DataFrame({'text': texts, 'label': labels})
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        data['text'], data['label'], test_size=0.1, random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained(SLM_MODEL_NAME)

    def preprocess_function(texts):
        return tokenizer(
            texts.tolist(),
            truncation=True,
            padding='max_length',
            max_length=128,
        )

    train_encodings = preprocess_function(train_texts)
    val_encodings = preprocess_function(val_texts)

    class SLM_Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels.reset_index(drop=True)

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels.iloc[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = SLM_Dataset(train_encodings, train_labels)
    val_dataset = SLM_Dataset(val_encodings, val_labels)

    model = AutoModelForSequenceClassification.from_pretrained(SLM_MODEL_NAME, num_labels=2)
    model.to(DEVICE)

    training_args = TrainingArguments(
        output_dir='./slm_results',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        logging_dir='./slm_logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save the trained model
    os.makedirs(SLM_MODEL_DIR, exist_ok=True)
    model.save_pretrained(SLM_MODEL_DIR)
    tokenizer.save_pretrained(SLM_MODEL_DIR)

if __name__ == '__main__':
    train_slm()