#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time: 2024/7/6 10:09
# @Author: LL
# @File：classify_reason.py
"""
Predict the reason for transformation pathways based on summary and operation description.
"""
import os
import re
from typing import List, Any
import pandas as pd
import numpy as np
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse

CURRENT_PATH = os.path.dirname(__file__)
GRANDPARENT_PATH = os.path.abspath(os.path.join(CURRENT_PATH, "../.."))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def predict_reason(data, model_path) -> list[Any]:
    """
    Predict the reason for transformation pathways based on given description.
    Agrs:
        data(list): The list of description strings.
        model(str): The path to the trained model.
    Returns:
        output(list): The list of probability on each category and predicted reasons.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    encodings = tokenizer(data, padding=True, truncation=True, max_length=512)
    dataset = TextDataset(encodings, np.zeros(len(data)))
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    model.eval()
    outputs = []
    # Progress
    for batch in tqdm(loader):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs.append(model(input_ids, attention_mask=attention_mask).logits)
    # Get probabilities on each class
    probs = torch.cat(outputs).softmax(dim=1)
    # Get the predicted reasons
    predicted_reasons = probs.argmax(dim=1)
    print(predicted_reasons)
    print(probs)
    output = []
    label_encoder = joblib.load(os.path.join(GRANDPARENT_PATH, 'model/reason-classifier/label_encoder.joblib'))
    # Decode label sequence
    label_sequence = label_encoder.inverse_transform(np.arange(len(label_encoder.classes_)).tolist())
    print(label_sequence)
    for i in range(len(data)):
        output.append({'category': label_encoder.inverse_transform([predicted_reasons[i].item()])[0],
                       'probabilities': dict(zip(label_sequence, probs[i].tolist()))})

    return output


def main():
    argparser = argparse.ArgumentParser(description="Predict the reason for transformation pathways.")
    argparser.add_argument('--model_path', type=str,
                           default=os.path.join(GRANDPARENT_PATH, 'model/reason-classifier/final-325'),
                           help='Path to the trained model.')
    argparser.add_argument('--input_path', type=str,
                           default=os.path.join(GRANDPARENT_PATH, 'data/curation/pathways.jsonl'),
                           help='Path to the input data file.')
    argparser.add_argument('--pred_res_path', type=str, default=os.path.join(GRANDPARENT_PATH,
                                                                             'data/reason-classifier'
                                                                             '/reason_prediction_for_pathways.csv'),
                           help='Path to save the prediction results.')
    argparser.add_argument('--output_path', type=str,
                           default=os.path.join(GRANDPARENT_PATH, 'data/curation/pathways.jsonl'),
                           help='Path to save the updated data with predicted reasons.')
    args = argparser.parse_args()
    data = pd.read_json(args.input_path, lines=True)
    summary = data['pathway'].apply(
        lambda x: x['reason']['summary'] if 'reason' in x and 'summary' in x['reason'] else '')
    description_of_operation = data['pathway'].apply(
        lambda x: [' '.join(procedure['original_description']) for procedure in x['procedure']
                   if 'original_description' in procedure])
    # Combine the summary and description of operation to a single string
    description = [re.sub(r'\s+', ' ', summary[i] + ' ' + ' '.join(description_of_operation[i])) for i in
                   range(len(data))]

    output = predict_reason(description, args.model_path)
    # Save output as csv
    pd.DataFrame(output).to_csv(args.output_path, index=False)

    # Update the reason in the data
    for i in range(len(data)):
        data['pathway'][i]['reason']['category'] = output[i]['category']
    # Save the updated data
    pd.DataFrame(data).to_json(args.output_path, lines=True, orient='records')


if __name__ == '__main__':
    main()
