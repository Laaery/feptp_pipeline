"""
Convert table to vector, and store in vector database
"""
import os
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
import weaviate
from weaviate.classes import Filter
from weaviate.util import generate_uuid5
from pymongo import MongoClient
from tqdm import tqdm
import logging

logger = logging.getLogger("p2v.table2vec")


class Table2Vec:
    """
    Convert table to vector, and store in vector database.
    """
    def __init__(self, client: weaviate.Client):
        self.client = client

    def load_data(self, tables, md_uuid, batch_size):
        # Configure batch
        self.client.batch.configure(batch_size=batch_size,
                                    retry_failed_objects=True,
                                    dynamic=True)
        # Populate data into table collection
        with self.client.batch as batch:
            for table in tqdm(tables, desc="Loading tables to Weaviate", ncols=100, total=len(tables)):
                # Return uuid of the full text
                tbl_uuid = batch.add_object(properties=table,
                                            collection="Table",
                                            uuid=generate_uuid5(table))
                # Cross-reference metadata and table
                batch.add_reference(from_object_collection="Table",
                                    from_object_uuid=tbl_uuid,
                                    from_property_name="hasMetadata",
                                    to_object_uuid=md_uuid,
                                    to_object_collection="Metadata")
                batch.add_reference(from_object_collection="Metadata",
                                    from_object_uuid=md_uuid,
                                    from_property_name="hasTable",
                                    to_object_uuid=tbl_uuid,
                                    to_object_collection="Table")
            batch.flush()

    def run(self, doi, tables, md_uuid, batch_size=50):
        _tables = []
        logger.info("=" * 20 + f"Start processing tables of {doi}..." + "=" * 20)
        for table in tables:
            count = 0
            splits = table.split("\n\n")
            # Sometimes missing label or caption might happen
            # If a split starting with a "|", regard it as the main body of the table
            if splits[0].startswith("|"):
                label = "None"
                caption = "None"
                tbody = splits[0:]
            elif splits[1].startswith("|"):
                label = splits[0]
                caption = "None"
                tbody = splits[1:]
            else:
                label = splits[0]
                caption = splits[1]
                tbody = "\n".join(splits[2:])
            # Fill data into table properties
            _tables.append({
                "label": label,
                "caption": caption,
                "tbody": tbody
            })

        self.load_data(_tables, md_uuid, batch_size)
        logger.info("=" * 20 + f"Finish processing tables of {doi}" + "=" * 20)
