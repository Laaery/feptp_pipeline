"""
Add published date to the curated data using crossref API.
"""
import os
import argparse
import json
import requests
import pandas as pd
from tqdm import tqdm


def get_publication_date(doi: str) -> str:
    """
    Query the CrossRef API to get the publication date (YYYY/MM) of a given DOI.

    Args:
        doi (str): The DOI of the paper.

    Returns:
        str or None: Publication date in the format 'YYYY/MM', or None if not found.
    """
    url = f"https://api.crossref.org/works/{doi}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        try:
            date_parts = data['message']['published-print']['date-parts'][0]
        except KeyError:
            return None
        try:
            return "{}/{}".format(date_parts[0], date_parts[1])
        except IndexError:
            return "{}/1".format(date_parts[0])
    else:
        print(f"Failed to get publication date for {doi}")
        return None


def enrich_curated_data_with_date(input_path: str, output_path: str):
    """
    Add publication dates to each DOI in the curated data.

    Args:
        input_path (str): Path to the original curated JSONL file.
        output_path (str): Path to save the enriched JSONL file.
    """
    tqdm.pandas()
    curated_data = pd.read_json(input_path, lines=True)
    print(f"Total records loaded: {curated_data.shape[0]}")

    curated_data = curated_data[curated_data['pathway'].apply(lambda x: x != [])]
    print(f"Records with non-empty pathways: {curated_data.shape[0]}")

    if not os.path.exists(output_path):
        print("Fetching publication dates...")
        curated_data['published_date'] = curated_data['doi'].progress_apply(get_publication_date)
        curated_data.to_json(output_path, orient='records', lines=True)
        print(f"Saved to: {output_path}")
    else:
        existing_data = pd.read_json(output_path, lines=True)
        missing = curated_data[~curated_data['doi'].isin(existing_data['doi'])]

        if not missing.empty:
            print(f"Appending missing publication dates for {missing.shape[0]} DOIs...")
            for doi in tqdm(missing['doi']):
                published_date = get_publication_date(doi)
                pathway = missing[missing['doi'] == doi]['pathway'].values[0]
                with open(output_path, 'a+') as f:
                    f.write(json.dumps({'doi': doi, 'pathway': pathway, 'published_date': published_date}) + '\n')
            print("Missing publication dates added.")
        else:
            print("All publication dates already present.")


def main():
    parser = argparse.ArgumentParser(description="Add publication dates to curated data using CrossRef API.")
    parser.add_argument(
        "--input_path", "-i", required=True,
        help="Path to the input curated JSONL file"
    )
    parser.add_argument(
        "--output_path", "-o", required=True,
        help="Path to the output JSONL file with publication dates"
    )
    args = parser.parse_args()

    enrich_curated_data_with_date(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
