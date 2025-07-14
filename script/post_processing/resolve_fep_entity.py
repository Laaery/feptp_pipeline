"""
Entity resolution for Fe-containing phases.
"""
import argparse
import os
import json
import pandas as pd
import unicodedata
from tqdm import tqdm


def preprocess(term):
    """Normalize Unicode and replace common special characters."""
    normalized_term = unicodedata.normalize("NFD", term)
    normalized_term = normalized_term.replace('\u22c5', '\u00b7').replace('\u2219', '\u00b7').replace('\u2022', '\u00b7')
    return normalized_term


def update_pathway_with_glossary(pathway_file_path, glossary_file_path, output_file_path):
    df = pd.read_json(pathway_file_path, lines=True)
    with open(glossary_file_path, 'r', encoding='utf-8') as file:
        glossary = json.load(file)

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        for phase in ['precursor_phase', 'product_phase']:
            for p in row['pathway'][phase]:
                name = preprocess(p['name'])
                formula = preprocess(p['formula'])
                original_description = preprocess(p['original_description'])

                suggestions_by_description = [entry for entry in glossary
                                              if original_description in entry['possible_names']
                                              and formula in entry['possible_formulas']]
                suggestions_by_name = [entry for entry in glossary
                                       if name in entry['possible_names']
                                       and formula in entry['possible_formulas']]

                if len(suggestions_by_description) == 1:
                    p['name'] = suggestions_by_description[0]['standard_name']
                    p['id'] = suggestions_by_description[0]['id']
                    print(f"[#{index}] Updated: {name} → {p['name']} (via original_description)")
                elif len(suggestions_by_name) == 1:
                    p['name'] = suggestions_by_name[0]['standard_name']
                    p['id'] = suggestions_by_name[0]['id']
                    print(f"[#{index}] Updated: {name} → {p['name']} (via name)")
                elif len(suggestions_by_description) > 1:
                    p['id'] = None
                    print(f"[#{index}] Conflict: multiple matches for original_description '{original_description}' and formula '{formula}'")
                else:
                    p['id'] = None
                    print(f"[#{index}] No match: '{original_description}' / '{name}' with formula '{formula}'")

    df.to_json(output_file_path, orient='records', force_ascii=False, lines=True)
    print(f"\n Output saved to: {output_file_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Update mineral phase names in transformation pathways using a glossary"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to the input pathway .jsonl file"
    )
    parser.add_argument(
        "--glossary", "-g", required=True,
        help="Path to the mineral glossary .json file"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Path to the output .jsonl file"
    )

    args = parser.parse_args()

    # Absolute path resolution
    input_path = os.path.abspath(args.input)
    glossary_path = os.path.abspath(args.glossary)
    output_path = os.path.abspath(args.output)

    update_pathway_with_glossary(
        pathway_file_path=input_path,
        glossary_file_path=glossary_path,
        output_file_path=output_path
    )


if __name__ == "__main__":
    main()
    # Example usage:
    # python script/resolve_fep_entity.py \
    #     --input data/curation/pathways_finalxx.jsonl \
    #     --glossary data/kb/extended_mineral_glossary_final.json \
    #     --output data/curation/pathways_finalxx.jsonl
