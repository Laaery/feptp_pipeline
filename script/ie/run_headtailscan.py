"""
Head-Tail Scanner for Document Processing

This script scans document heads and tails to extract key information using Weaviate and MongoDB.
"""
import argparse
import logging
import os
from pymongo import MongoClient
import weaviate
from feptp_pipeline.IE.ie_core import HeadTailScanner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ie.headtailscan")

GRANDPARENT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


def initialize_clients(mongo_uri="mongodb://localhost:27017/"):
    """Initialize Weaviate and MongoDB clients."""
    try:
        wv_client = weaviate.connect_to_local(
            port=8080,
            grpc_port=50051
        )
        mg_client = MongoClient(mongo_uri)
        return wv_client, mg_client
    except Exception as e:
        logger.error(f"Failed to initialize clients: {e}")
        raise


def process_documents(args, wv_client, mg_client):
    """Process documents using HeadTailScanner."""
    logger.info("=" * 20 + "Head-tail scanner initialize..." + "=" * 20)

    output_base = os.path.abspath(args.output_base)

    # Create output directory structure
    os.makedirs(output_base, exist_ok=True)

    # Initialize scanner
    scanner = HeadTailScanner(
        prompt=os.path.abspath(args.prompt_path),
        kb_path=os.path.join(GRANDPARENT_PATH, "data/kb/mineral_glossary.csv"),
        wv_client=wv_client,
        model_version=args.model_version
    )

    # File paths
    scanned_file = os.path.join(output_base, "scanned_dois.txt")
    missed_file = os.path.join(output_base, "missed_dois.txt")
    notebook_file = os.path.join(output_base, "notebook.jsonl")

    # Initialize tracking files
    for f in [scanned_file, missed_file, notebook_file]:
        if not os.path.exists(f):
            open(f, 'w').close()

    # Get MongoDB collection
    db = mg_client[args.mongo_db]
    collection = db[args.mongo_collection]

    count = 0
    for document in collection.find():
        doi = document.get('doi', '').strip()

        # Skip if already processed
        with open(scanned_file, 'r') as f:
            scanned_dois = [line.strip() for line in f.readlines()]
            if doi in scanned_dois:
                logger.info(f"{doi} has been scanned. Skip.")
                continue

        try:
            scanner.scan(
                doi=doi,
                storage_path=notebook_file
            )

            # Mark as successfully processed
            with open(scanned_file, 'a') as f:
                f.write(doi + "\n")

            count += 1
            logger.info(f"{count} papers scanned. Last DOI: {doi}")

            # Early exit for testing
            if args.max_documents and count >= args.max_documents:
                logger.info(f"Reached maximum document limit ({args.max_documents}). Stopping.")
                break

        except Exception as e:
            logger.error(f"Failed to process {doi}: {str(e)}")
            with open(missed_file, 'a') as f:
                f.write(doi + "\n")
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Head-Tail Document Scanner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Optional arguments with defaults
    parser.add_argument(
        "--mongo-uri",
        default="mongodb://localhost:27017/",
        help="MongoDB connection URI"
    )
    parser.add_argument(
        "--mongo-db",
        default="fe_mineral",
        help="MongoDB database name"
    )
    parser.add_argument(
        "--mongo-collection",
        default="elsevier_full_text_v4",
        help="MongoDB collection name"
    )
    parser.add_argument(
        "--output-base",
        default="../../data/basic_keynotes",
        help="Base directory for output files"
    )
    parser.add_argument(
        "--prompt-path",
        default="../../prompt/prompt_extract_keynote.json",
        help="Path to prompt JSON file"
    )
    parser.add_argument(
        "--model-version",
        default="gpt-4-0125-preview",
        help="Model version to use"
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        help="Maximum number of documents to process (for testing)"
    )

    args = parser.parse_args()

    try:
        wv_client, mg_client = initialize_clients(
            mongo_uri=args.mongo_uri
        )
        process_documents(args, wv_client, mg_client)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
    finally:
        if 'wv_client' in locals():
            wv_client.close()
        if 'mg_client' in locals():
            mg_client.close()


if __name__ == "__main__":
    main()
