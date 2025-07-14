"""
Selects uncertain samples from a dataset based on a given uncertainty threshold.
"""
import pandas as pd
import jsonlines
import argparse
import os


def select_uncertain_samples(csv_file_path, num_samples=50):
    """
    Select most uncertain samples based on probability closest to 0.5.
    """
    df = pd.read_csv(csv_file_path)
    if 'prob_1' not in df.columns or 'index' not in df.columns:
        raise ValueError("CSV must contain 'prob_1' and 'index' columns.")

    df['distance_to_0.5'] = abs(df['prob_1'] - 0.5)
    df_sorted = df.sort_values(by='distance_to_0.5', ascending=True)
    selected_samples = df_sorted.head(num_samples)['index'].tolist()
    return selected_samples


def extract_and_save_abstracts(selected_samples, input_csv_path, output_jsonl_path):
    """
    Extract abstracts from cleaned data CSV and save to JSONL.
    """
    all_data = pd.read_csv(input_csv_path, encoding='utf-8', encoding_errors='ignore')
    selected_abstracts = all_data.loc[all_data['index'].isin(selected_samples)]['abstract']
    with jsonlines.open(output_jsonl_path, mode='w') as writer:
        for abstract in selected_abstracts:
            data = {'abstract': abstract, 'label': 'unk'}
            writer.write(data)


def main(args):
    # Select uncertain samples
    selected_samples = select_uncertain_samples(args.pred_csv, args.num_samples)
    print(f"Selected {len(selected_samples)} uncertain samples: {selected_samples[:5]}...")

    # Save selected indices
    os.makedirs(os.path.dirname(args.output_index), exist_ok=True)
    pd.DataFrame(selected_samples, columns=['index']).to_csv(args.output_index, index=False, encoding='utf-8')
    print(f"Sample indices saved to: {args.output_index}")

    # Extract and save abstracts
    extract_and_save_abstracts(selected_samples, args.abstract_csv, args.output_jsonl)
    print(f"Abstracts saved to: {args.output_jsonl}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Active Learning: Select uncertain samples by least confidence.")
    parser.add_argument('--round', type=int, default=1, help='Round number for output naming.')
    parser.add_argument('--pred_csv', type=str, required=True, help='Path to prediction result CSV with prob_1 column.')
    parser.add_argument('--abstract_csv', type=str, required=True, help='Path to full cleaned abstract CSV.')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of uncertain samples to select.')
    parser.add_argument('--output_index', type=str, required=True, help='Path to save selected indices CSV.')
    parser.add_argument('--output_jsonl', type=str, required=True, help='Path to save selected abstracts JSONL.')

    args = parser.parse_args()
    main(args)