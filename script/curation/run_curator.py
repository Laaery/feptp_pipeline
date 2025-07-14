import os
from feptp_pipeline.IE.curator import Curator
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run full-text curation pipeline")
    parser.add_argument("--input", required=True, help="Path to input JSONL file with extracted info")
    parser.add_argument("--output", default="../../data/curation/curated_pathways.jsonl", help="Path to save curated JSONL output")
    parser.add_argument("--curated_dois", default="../../data/exp/full_ext_curated/curated_doi.txt", help="Path to save curated DOIs")
    parser.add_argument("--missed_dois", default="../../data/exp/full_ext_curated/missed_dois.txt", help="Path to save missed DOIs")
    parser.add_argument("--keynote_path", default="../../data/basic_keynotes/notebook.jsonl", help="Path to the keynote file")
    parser.add_argument("--prompt_path", default="../../prompt/prompt_curator.json", help="Prompt file path")
    parser.add_argument("--log_config", default="../../config/logging_curator.yaml", help="Path to logging config file")

    args = parser.parse_args()

    curator = Curator(
        input_path=args.input,
        output_path=args.output,
        curated_doi_path=args.curated_dois,
        missed_doi_path=args.missed_dois,
        keynote_path=args.keynote_path,
        prompt_path=args.prompt_path,
        log_config_path=args.log_config
    )
    curator.run()


if __name__ == "__main__":
    main()

