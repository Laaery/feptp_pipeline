"""
Train the reason classifier to predict the reason for transformation pathways based on summary.
Use as a supplementary method to calibrate the results from curator.
"""
import os
import pandas as pd
import numpy as np
import joblib
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import wandb
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from transformers import EarlyStoppingCallback
import argparse

CURRENT_PATH = os.path.dirname(__file__)
GRANDPARENT_PATH = os.path.abspath(os.path.join(CURRENT_PATH, "../.."))
SEED = 42
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
# Set random seeds and deterministic pytorch for reproducibility
random.seed(SEED)  # python random seed
torch.manual_seed(SEED)  # pytorch random seed
torch.cuda.manual_seed(SEED)  # pytorch cuda random seed
np.random.seed(SEED)  # numpy random seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



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


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    balanced_accuracy = balanced_accuracy_score(labels, preds, adjusted=False)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')

    return {
        'accuracy': balanced_accuracy,  # Use balanced accuracy as the main metric for optimization
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def model_init():
    model = AutoModelForSequenceClassification.from_pretrained(r'D:\Huggingface_model\matscibert', num_labels=5)
    model.to(device)
    return model


def objective(trial, train_dataset, val_dataset):
    wandb.login()
    wandb.init(project='fempt_reason_classification_325', name='fempt_reason_classification_optuna')

    # Hyperparameters to tune
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    weight_decay = trial.suggest_float('weight_decay', 0.0, 0.3)
    epochs = trial.suggest_int('epochs', 3, 4)

    training_args = TrainingArguments(
        logging_dir=os.path.join(GRANDPARENT_PATH, 'logs/reason_classifier'),
        output_dir=os.path.join(GRANDPARENT_PATH, 'model/reason-classifier/opt-325'),
        overwrite_output_dir=True,
        evaluation_strategy='epoch',
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        logging_strategy="epoch",
        save_strategy="epoch",
        weight_decay=weight_decay,
        metric_for_best_model='accuracy',
        report_to=['wandb'],  # Enable wandb logging
        run_name='fempt_reason_classification_optuna',
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Evaluate the model on the validation set
    eval_result = trainer.evaluate(eval_dataset=val_dataset)
    wandb.finish()

    return eval_result['eval_accuracy']


def main():
    parser = argparse.ArgumentParser(description='Train a reason classifier for transformation pathways.')
    parser.add_argument('--train_dataset', type=str, default=os.path.join(GRANDPARENT_PATH, 'data/reason-classifier/train_dataset_325.csv'),
                        help='Path to the training dataset CSV file.')
    parser.add_argument('--val_dataset', type=str, default=os.path.join(GRANDPARENT_PATH, 'data/reason-classifier/validation_dataset.csv'),
                        help='Path to the validation dataset CSV file.')
    parser.add_argument('--test_dataset', type=str, default=os.path.join(GRANDPARENT_PATH, 'data/reason-classifier/test_dataset.csv'),
                        help='Path to the test dataset CSV file.')
    parser.add_argument('--pretrained_model', type=str, default=r'D:\Huggingface_model\matscibert',
                        help='Path to the pretrained model directory.')
    parser.add_argument('--wandb_project', type=str, default='fempt_reason_classification_325',
                        help='WandB project name for logging.')
    parser.add_argument('--wandb_run_name', type=str, default='fempt_reason_classification_final',
                        help='WandB run name for logging.')
    parser.add_argument('--output_dir', type=str, default=os.path.join(GRANDPARENT_PATH, 'model/reason-classifier/final-325'),
                        help='Directory to save the trained model and tokenizer.')
    parser.add_argument('--test_result', type=str, default=os.path.join(GRANDPARENT_PATH, 'data/reason-classifier/test_results_325.csv'),
                        help='Path to save the test results CSV file.')
    parser.add_argument('--test_metrics', type=str, default=os.path.join(GRANDPARENT_PATH, 'data/reason-classifier/test_metrics_325.csv'),
                        help='Path to save the test metrics CSV file.')
    args = parser.parse_args()


    # Step 1: Load and preprocess data
    # Load data
    train_dataset = pd.read_csv(args.train_dataset)
    X_train, y_train = train_dataset['summary'], train_dataset['category']
    val_dataset = pd.read_csv(args.val_dataset)
    X_val, y_val = val_dataset['summary'], val_dataset['category']
    test_dataset = pd.read_csv(args.test_dataset)
    X_test, y_test = test_dataset['summary'], test_dataset['category']
    entire_train_dataset = pd.concat([train_dataset, val_dataset])
    X_entire_train, y_entire_train = entire_train_dataset['summary'], entire_train_dataset['category']

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)
    y_entire_train_encoded = label_encoder.transform(y_entire_train)

    # Save the label encoder
    joblib.dump(label_encoder, os.path.join(GRANDPARENT_PATH, 'model/reason-classifier/label_encoder.joblib'))
    print("Unique labels:", np.unique(y_train_encoded))

    # Step 2: Tokenize data
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    train_encodings = tokenizer(list(X_train), padding=True, truncation=True, max_length=512)
    val_encodings = tokenizer(list(X_val), padding=True, truncation=True, max_length=512)
    test_encodings = tokenizer(list(X_test), padding=True, truncation=True, max_length=512)
    entire_train_encodings = tokenizer(list(X_entire_train), padding=True, truncation=True, max_length=512)

    # Step 3: Prepare datasets
    train_dataset = TextDataset(train_encodings, y_train_encoded)
    val_dataset = TextDataset(val_encodings, y_val_encoded)
    test_dataset = TextDataset(test_encodings, y_test_encoded)
    entire_train_dataset = TextDataset(entire_train_encodings, y_entire_train_encoded)
    # Step 4: Hyperparameter Tuning with Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    print(f'Best hyperparameters: {study.best_params}')

    # Step 5: Train the best model and evaluate on test set
    best_params = study.best_params
    wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=os.path.join(GRANDPARENT_PATH, 'logs/reason_classifier'),
        evaluation_strategy='epoch',
        learning_rate=best_params['learning_rate'],
        per_device_train_batch_size=best_params['batch_size'],
        per_device_eval_batch_size=best_params['batch_size'],
        num_train_epochs=3,
        weight_decay=best_params['weight_decay'],
        metric_for_best_model='accuracy',
        report_to=['wandb'],  # Enable wandb logging
        run_name= args.wandb_run_name,
    )

    # Train with the entire training dataset
    trainer = Trainer(
        model=model_init(),
        args=training_args,
        train_dataset=entire_train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    wandb.finish()
    # Step 5: Save the model and create inference function
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Step 6: Evaluation on test set
    # Evaluate the model on the test dataset
    predictions, labels, metrics = trainer.predict(test_dataset)

    # Extract predicted labels
    predicted_labels = predictions.argmax(axis=-1)
    # Probability of each class
    probs = torch.softmax(torch.tensor(predictions), dim=1).numpy()

    # Save the results to a CSV file
    results = pd.DataFrame({
        'y_preds': predicted_labels,
        'y_true': labels,
        'probs': probs.tolist()
    })
    results.to_csv(args.test_result, index=False)
    # Calculate the accuracy
    accuracy = balanced_accuracy_score(labels, predicted_labels, adjusted=False)
    precision = precision_score(labels, predicted_labels, average='weighted')
    recall = recall_score(labels, predicted_labels, average='weighted')
    f1 = f1_score(labels, predicted_labels, average='weighted')

    print(f'Balanced Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print('Finished training and evaluation on test set.')

    # Save metrics to csv
    metrics = pd.DataFrame({
        'metric': ['balanced_accuracy', 'precision', 'recall', 'f1'],
        'score': [accuracy, precision, recall, f1]
    })
    metrics.to_csv(args.test_metrics, index=False)


if __name__ == '__main__':
    main()
    print('Finished training and evaluation on test set.')
