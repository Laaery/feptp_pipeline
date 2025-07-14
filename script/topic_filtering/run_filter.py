"""
Run a filter to select documents based on specific topics.
"""
import argparse
import pandas as pd
from feptp_pipeline.topic_filtering.filter import ClassificationDatasetDeploy, ModelDeployer


def main():
    parser = argparse.ArgumentParser(description="Apply a fine-tuned model to classify text abstracts.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV file with abstracts")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save classification results")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for class 1")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")

    args = parser.parse_args()

    dataset = ClassificationDatasetDeploy(args.data_path)
    deployer = ModelDeployer(model_path=args.model_path, threshold=args.threshold, batch_size=args.batch_size)
    results, run_time = deployer.classify(dataset)

    print(f"Classification completed in {run_time}")

    # Save to CSV
    df = pd.DataFrame(results, columns=["label", "logits_0", "logits_1", "prob_0", "prob_1"])
    df.insert(0, "index", range(1, len(df) + 1))
    df.to_csv(args.output_path, index=False)
    print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
