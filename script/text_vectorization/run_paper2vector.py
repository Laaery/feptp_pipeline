"""
Run a pipeline to preprocess, vectorize and store full text and tables from scientific papers to Weaviate Database
"""
import argparse
import os
from pathlib import Path

import weaviate
from pymongo import MongoClient
from weaviate import UnexpectedStatusCodeException
from weaviate.classes import Filter
from weaviate.util import generate_uuid5
from feptp_pipeline.paper2vector import si2vec, text2vec, table2vec
import re
import logging
import logging.config
import yaml


def setup_logger(log_config_path):
    log_config_path = Path(log_config_path).resolve()
    with log_config_path.open("r") as f:
        config = yaml.safe_load(f)
    project_root = log_config_path.parents[1]
    log_file_path = project_root / "logs" / "p2v.log"
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    config["handlers"]["file"]["filename"] = str(log_file_path)
    logging.config.dictConfig(config)

    return logging.getLogger("p2v")


def main(args):
    logger = setup_logger(args.log_config)
    logger.info("P2v initialize...")

    # Connect to Weaviate
    wv_client = weaviate.connect_to_local(port=8080, grpc_port=50051)
    logger.info("Weaviate client connected.")

    # Load collections
    metadata = wv_client.collections.get("Metadata")
    fulltext = wv_client.collections.get("FullText")
    table = wv_client.collections.get("Table")

    # Connect to MongoDB
    mg_client = MongoClient("localhost", 27017)

    if not args.db_name or not args.collection_name:
        raise ValueError("Database name and collection name must be provided.")
    db = mg_client[args.db_name]
    collection = db[args.collection_name]
    logger.info(f"MongoDB connected: DB = {args.db_name}, Collection = {args.collection_name}")

    # Load processed DOIs
    if os.path.exists(args.processed_dois):
        with open(args.processed_dois, "r") as f:
            processed_dois = set([line.strip() for line in f])
    else:
        processed_dois = set()

    # Initialize p2v
    si_pipeline = si2vec.Si2Vec(wv_client)
    ft_pipeline = text2vec.Text2Vec(wv_client)
    tbl_pipeline = table2vec.Table2Vec(wv_client)

    count = 0
    for document in collection.find():
        doi = document.get("doi")
        if doi in processed_dois:
            logger.info(f"{doi} already processed.")
            continue

        try:
            cleaned_abstract = re.sub(r'\s+', ' ', document.get('abstract', ''))
            meta_properties = {
                "doi": doi,
                "title": document.get("title", ""),
                "abstract": cleaned_abstract
            }

            response = metadata.query.fetch_objects(filters=Filter("doi").equal(str(doi)), limit=1)
            if len(response.objects) == 0:
                md_uuid = metadata.data.insert(properties=meta_properties,
                                               uuid=generate_uuid5(meta_properties))
                logger.info(f"Metadata of {doi} created.")
            else:
                logger.info(f"Metadata of {doi} already exists.")
                fulltext.data.delete_many(
                    where=Filter(["hasMetadata", "Metadata", "doi"]).equal(str(doi))
                )
                table.data.delete_many(
                    where=Filter(["hasMetadata", "Metadata", "doi"]).equal(str(doi))
                )
                md_uuid = response.objects[0].uuid

            # Full text
            if 'full_text' in document:
                ft_pipeline.run(
                    doi=doi,
                    full_text=document['full_text'],
                    full_text_type=document.get('full_text_type', ''),
                    md_uuid=md_uuid
                )

            # Tables
            if 'table' in document:
                tbl_pipeline.run(
                    doi=doi,
                    tables=document['table'],
                    md_uuid=md_uuid
                )

            # Supplementary Information
            if 'si' in document and args.si_root:
                si_pipeline.run(
                    doi=doi,
                    paths=document['si'],
                    root=args.si_root,
                    md_uuid=md_uuid,
                )

            # Mark as processed
            with open(args.processed_dois, "a+") as f:
                f.write(doi + "\n")
            processed_dois.add(doi)

            count += 1
            logger.info(f"{count} papers processed.")

        except UnexpectedStatusCodeException as e:
            if e.status_code == 500:
                logger.error("Connection timeout. Please retry.", exc_info=True)
        except Exception as e:
            logger.error(f"Error processing {doi}", exc_info=True)


if __name__ == "__main__":
    CURRENT_FILE = Path(__file__).resolve()
    PROJECT_ROOT = CURRENT_FILE.parents[2]
    config_path = PROJECT_ROOT / "config" / "logging_p2v.yaml"
    parser = argparse.ArgumentParser(description="Ingest scientific articles into Weaviate.")
    parser.add_argument("--db_name", type=str, default="fe_mineral",
                        help="MongoDB database name")
    parser.add_argument("--collection_name", type=str,
                        help="MongoDB collection name")
    parser.add_argument("--processed_dois", type=str, default="../../logs/processed_dois_p2v.txt",
                        help="Path to record processed dois")
    parser.add_argument("--log_config", type=str, default=config_path,
                        help="Path to logging_p2v.yaml config file")
    parser.add_argument("--si_root", type=str, default=None,
                        help="Root directory for supplementary materials (SI)")

    args = parser.parse_args()
    log_path = os.path.dirname(args.processed_dois)
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)
    if not os.path.isfile(args.processed_dois):
        open(args.processed_dois, 'a').close()  # Create if missing

    for path in [args.log_config, args.si_root]:
        if path is None:
            continue
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path does not exist: {path}")

    main(args)
