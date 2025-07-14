import argparse
import logging
import os
import time
from pymongo import MongoClient
import weaviate
from feptp_pipeline.IE.ie_core import Extractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ie.extractor")


def initialize_clients(mongo_uri="mongodb://localhost:27017/"):
    """Initialize Weaviate and MongoDB clients."""
    try:
        wv_client = weaviate.connect_to_local(
            port=8080,
            grpc_port=50051,
        )
        mg_client = MongoClient(mongo_uri)
        return wv_client, mg_client
    except Exception as e:
        logger.error(f"Failed to initialize clients: {e}")
        raise


def process_documents(args, wv_client, mg_client):
    """Process documents from MongoDB and extract information."""
    logger.info("=" * 20 + "Extractor initialize..." + "=" * 20)

    db = mg_client[args.mongo_db]
    collection = db[args.mongo_collection]
    output_folder = os.path.abspath(args.output_folder)

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Initialize extractor
    extractor = Extractor(
        wv_client=wv_client,
        output_folder=output_folder,
        keynote_path=os.path.abspath(args.keynote_path),
    )

    # Files to track processed and missed DOIs
    scanned_file = os.path.join(output_folder, "scanned_dois.txt")
    missed_file = os.path.join(output_folder, "missed_dois.txt")

    # Create files if they don't exist
    for f in [scanned_file, missed_file]:
        if not os.path.exists(f):
            open(f, 'w').close()

    count = 0
    batch_size = args.batch_size
    sleep_time = args.sleep_time

    for document in collection.find().batch_size(batch_size):
        doi = document.get('doi', '').strip()

        # Skip if already processed
        with open(scanned_file, 'r') as f:
            scanned_dois = [line.strip() for line in f.readlines()]
            if doi in scanned_dois:
                logger.info(f"{doi} has been extracted. Skip.")
                continue

        # Check for full text
        full_text = document.get('full_text', '')
        if not full_text:
            logger.info(f"{doi} has no full text. Skip.")
            continue

        # Get tables if available
        tables = document.get('table', [])
        table_text = "\n".join(tables) if tables else ''

        try:
            extractor.full_text_extract(
                doi=doi,
                text=full_text,
                table=table_text
            )
            time.sleep(sleep_time)

            # Mark as successfully processed
            with open(scanned_file, 'a') as f:
                f.write(doi + "\n")

            count += 1
            logger.info(f"{count} papers scanned. Last DOI: {doi}")

        except Exception as e:
            logger.error(f"Failed to process {doi}: {str(e)}")
            with open(missed_file, 'a') as f:
                f.write(doi + "\n")
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Mineral Knowledge Extraction Pipeline",
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
        "--output-folder",
        default="../../data/exp/full_ext",
        help="Output directory for extracted data"
    )
    parser.add_argument(
        "--keynote-path",
        default="../../data/basic_keynotes/GPT4/notebook.jsonl",
        help="Path to keynote JSONL file"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=15,
        help="Number of documents to process in each batch"
    )
    parser.add_argument(
        "--sleep-time",
        type=int,
        default=20,
        help="Sleep time between document processing (seconds)"
    )

    args = parser.parse_args()

    try:
        wv_client, mg_client = initialize_clients(
            openai_api_key=args.openai_api_key,
            mongo_uri=args.mongo_uri
        )
        process_documents(args, wv_client, mg_client)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
    finally:
        if 'wv_client' in locals():
            wv_client.close()
        if 'mg_client' in locals():
            mg_client.close()


if __name__ == "__main__":
    main()
