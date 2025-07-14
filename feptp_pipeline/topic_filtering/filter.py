"""
Further distinguish the subtopics about phase transformation.
"""
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
import datetime
import numpy as np
from bs4 import BeautifulSoup
import re
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from tokenizers import normalizers
import jsonlines


# Define normalizer
normalizer = normalizers.BertNormalizer(clean_text=True, lowercase=True, strip_accents=None, handle_chinese_chars=False)


class ClassificationDatasetTrain(Dataset):
    def __init__(self, path_to_file):
        self.dataset = pd.DataFrame(columns=['text', 'label'])
        self.path_to_file = path_to_file
        self._load_data()

    def _load_data(self):
        with jsonlines.open(self.path_to_file) as reader:
            for idx, obj in enumerate(reader):
                self.dataset.loc[idx, "text"] = normalizer.normalize_str(obj.get('text'))
                self.dataset.loc[idx, "label"] = obj.get('label')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {"text": self.dataset.loc[idx, "text"], "label": self.dataset.loc[idx, "label"]}


class ClassificationDatasetDeploy(Dataset):
    def __init__(self, path_to_file):
        self.dataset = pd.DataFrame(columns=['text'])
        self.path_to_file = path_to_file
        self._load_data()

    def _load_data(self):
        df = pd.read_csv(self.path_to_file, usecols=["abstract"], encoding='utf-8', encoding_errors='ignore')
        df.dropna(subset=['abstract'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["abstract"] = df["abstract"].str.replace("\t|\n|\r|\"", "", regex=True).str.strip()

        for i in range(len(df)):
            text = df.loc[i, "abstract"]
            text = normalizer.normalize_str(str(text))
            text = BeautifulSoup(text, "lxml").text
            text = re.sub('\s+', " ", text)
            df.loc[i, "abstract"] = text

        self.dataset['text'] = df['abstract']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {"text": self.dataset.loc[idx, "text"]}


class ModelDeployer:
    def __init__(self, model_path, threshold=0.5, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path, num_labels=2)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config).to(self.device)
        self.batch_size = batch_size
        self.threshold = threshold

    def _format_time(self, elapsed):
        return str(datetime.timedelta(seconds=int(round(elapsed))))

    def classify(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=SequentialSampler(dataset))
        self.model.eval()
        res = []
        t0 = time.time()

        for batch in dataloader:
            text = batch["text"]
            tokens = self.tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**tokens, return_dict=False)
            logits = outputs[0].detach().cpu().numpy()
            prob = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
            preds = (prob[:, 1] > self.threshold).astype(int)
            for i in range(len(preds)):
                res.append([preds[i], logits[i, 0], logits[i, 1], prob[i, 0], prob[i, 1]])

        elapsed = self._format_time(time.time() - t0)
        return res, elapsed


def convert_tokenized_data(tokenized_data, labels):
    df = pd.DataFrame({
        'input_ids': tokenized_data['input_ids'],
        'token_type_ids': tokenized_data['token_type_ids'],
        'attention_mask': tokenized_data['attention_mask'],
        'label': labels
    })
    return Dataset.from_pandas(df)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}


def model_init(model_file, num_labels, hidden_dropout_prob):
    config = AutoConfig.from_pretrained(model_file, num_labels=num_labels, hidden_dropout_prob=hidden_dropout_prob)
    return AutoModelForSequenceClassification.from_pretrained(model_file, config=config)


def hp_space_optuna(trial):
    return {
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32]),
        "num_train_epochs": trial.suggest_categorical("num_train_epochs", [3, 4]),
        "learning_rate": trial.suggest_categorical("learning_rate", [2e-5, 3e-5, 5e-5]),
        "weight_decay": trial.suggest_categorical("weight_decay", [0, 0.01, 0.05, 0.2]),
    }

