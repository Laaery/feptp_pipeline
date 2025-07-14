"""
Train a filter for a specific task using a dataset.
"""
import argparse
from transformers import Trainer, TrainingArguments, AutoTokenizer
from feptp_pipeline.topic_filtering.filter import ClassificationDatasetTrain, convert_tokenized_data, compute_metrics, model_init, hp_space_optuna


def main():
    parser = argparse.ArgumentParser(description="Run fine-tuning with Optuna hyperparameter search")
    parser.add_argument('--model_path', required=True, help='Path to the pretrained model')
    parser.add_argument('--train_data', required=True, help='Path to the training data JSONL file')
    parser.add_argument('--val_data', required=True, help='Path to the validation data JSONL file')
    parser.add_argument('--output_dir', required=True, help='Output directory for saving model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1, help='Dropout probability for the model')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length for tokenization')
    parser.add_argument('--num_labels', type=int, default=2, help='Number of labels for classification')
    parser.add_argument('--num_trails', type=int, default=20, help='Number of trials for hyperparameter search')

    args = parser.parse_args()

    hidden_dropout_prob = args.hidden_dropout_prob
    num_labels = args.num_labels

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    train_data = ClassificationDatasetTrain(args.train_data)
    val_data = ClassificationDatasetTrain(args.val_data)

    tokenized_train = tokenizer([train_data[i]['text'] for i in range(len(train_data))], max_length=args.max_length,
                                truncation=True, padding=True)
    tokenized_val = tokenizer([val_data[i]['text'] for i in range(len(val_data))], max_length=args.max_length, truncation=True,
                              padding=True)

    train_dataset = convert_tokenized_data(tokenized_train, [train_data[i]['label'] for i in range(len(train_data))])
    val_dataset = convert_tokenized_data(tokenized_val, [val_data[i]['label'] for i in range(len(val_data))])

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_eval_batch_size=16,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        seed=args.seed,
    )

    trainer = Trainer(
        model_init=lambda: model_init(args.model_path, num_labels, hidden_dropout_prob),
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )

    best_run = trainer.hyperparameter_search(
        hp_space=hp_space_optuna,  # Go to hp_space_optuna to change the hyperparameter search space if needed
        backend="optuna",
        n_trials=args.num_trails,
        direction="maximize",
    )

    print("Best run:", best_run)
    for n, v in best_run.hyperparameters.items():
        setattr(trainer.args, n, v)

    trainer.train()


if __name__ == '__main__':
    main()
