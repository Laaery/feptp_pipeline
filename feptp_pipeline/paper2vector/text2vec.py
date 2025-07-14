"""
Convert text to vector, and store in Weaviate vector database
"""
import os
import openai
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Weaviate
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.docstore.document import Document
import weaviate
from weaviate.util import generate_uuid5
from pymongo import MongoClient
import spacy
from tqdm import tqdm
import logging

logger = logging.getLogger("p2v.text2vec")


class Text2Vec:
    """
    Define a pipeline to split full text into meaningful chunks, vectorize and store them in Weaviate database.
    """

    def __init__(self, client: weaviate.Client):
        """
        Initialize the pipeline for full text.
        """
        self.client = client

    # Define function to identify headers and assign each paragraph with a header
    def organize_paragraphs(self, text_list, max_header_length=10):
        """
        Identify headers and assign a header to each text segments.

        Args:
            text_list(list): list of strings
            max_header_length(int): threshold to identify a header

        Returns:
            Dict of text segments with section headers.
        """
        # Initialize an empty dictionary to store headings and paragraphs
        result_dict = {'introduction': [], 'method': [], 'results and discussion': [], 'conclusion': []}

        # Variable to store the current heading
        current_heading = 'introduction'

        # Load spacy model
        spacy_en = spacy.load("en_core_web_sm")
        # Iterate through the list of strings
        for text in text_list:
            try:
                # Check if the text meets the standard of a heading
                if len([tok.text for tok in spacy_en.tokenizer(text)]) <= max_header_length:
                    # Figure out what heading it is
                    if any(keyword in text.lower() for keyword in ['introduction', 'background']):
                        current_heading = 'introduction'
                    elif any(
                            keyword in text.lower() for keyword in ['method', 'experiment', 'methodology', 'materials',
                                                                    'sample', 'procedure', 'approach']):
                        current_heading = 'method'
                    elif any(keyword in text.lower() for keyword in ['results', 'discussion']):
                        current_heading = 'results and discussion'
                    elif any(keyword in text.lower() for keyword in
                             ['conclusion', 'prospect', 'perspective', 'limitation', 'implication', 'outlook']):
                        current_heading = 'conclusion'
                    else:
                        # The RecursiveCharacterTextSplitter make sure that the text is a complete sentence at least,
                        # so if it is not a heading, it is most likely greater than 10 words
                        # The rest of the headings are non-standard, regards them as 'result and discussion'
                        current_heading = 'results and discussion'
                else:
                    # Add the paragraph to the list under the current heading
                    result_dict[current_heading].append(text)
            except TypeError:
                pass

        return result_dict

    def markdown_splitter(self, full_text: str) -> list[Document]:
        """
        Split Markdown text into meaningful text segments.

          Args:
              full_text(str): Markdown text

          Returns:
              List of Document objects.
        """
        # Split text into sections according to headers
        headers_to_split_on = [
            ("#", "header_1"),
            ("##", "header_2"),
            ("###", "header_3"),
            ("####", "header_4"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(full_text)
        # List of Document objects
        # print(md_header_splits)

        # Configure TextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512,
                                                       chunk_overlap=100,
                                                       separators=["\n\n", "\n", "(?<=\\. )", "(?<=\\, )"],
                                                       length_function=len)
        # Further split paragraphs in Document objects into smaller chunks using TextSplitter
        docs = text_splitter.split_documents(md_header_splits)  # List of Documents objects
        logger.info(f'{len(docs)} pieces of docs are loaded to Weaviate.')
        return docs

    def plaintext_splitter(self, full_text: str) -> list[Document]:
        """
        Split plain text into meaningful text segments.

          Args:
              full_text(str): Plain text

          Returns:
              List of Document objects.
        """
        # Configure TextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512,
                                                       chunk_overlap=100,
                                                       separators=["\n\n", "\n", ".", ","],
                                                       length_function=len)
        # Split text into chunks
        docs = text_splitter.split_text(full_text)
        # Create Document objects
        docs = text_splitter.create_documents(docs)
        logger.info(f'{len(docs)} pieces of docs are loaded to Weaviate.')
        return docs
    # docs = []

    # for p in paragraphs:
    #     doc = text_splitter.split_text(p)
    #     docs.extend(doc)

    # # Remove empty string
    # docs = [doc for doc in docs if doc != ""]
    # # Number of docs
    # num_docs = len(docs)
    # # Organize docs into sections
    # sections = organize_paragraphs(docs)
    def load_data(self, docs, md_uuid, batch_size):
        # Configure batch
        self.client.batch.configure(batch_size=batch_size,
                                    retry_failed_objects=True,
                                    dynamic=True)

        # Customize storage
        with self.client.batch as batch:
            # Add object to batch
            for doc in tqdm(docs, desc="Loading docs to Weaviate", ncols=100, total=len(docs)):
                properties = {
                    "text": doc.page_content,
                }
                # Adding section where the text belongs
                for key in doc.metadata.keys():
                    properties[key] = doc.metadata[key]
                # Return uuid of the full text
                ft_uuid = batch.add_object(properties=properties,
                                           collection="FullText",
                                           uuid=generate_uuid5(properties))
                # Cross-reference metadata and full text
                batch.add_reference(from_object_collection="FullText",
                                    from_object_uuid=ft_uuid,
                                    from_property_name="hasMetadata",
                                    to_object_uuid=md_uuid,
                                    to_object_collection="Metadata")
                batch.add_reference(from_object_collection="Metadata",
                                    from_object_uuid=md_uuid,
                                    from_property_name="hasFullText",
                                    to_object_uuid=ft_uuid,
                                    to_object_collection="FullText")

            batch.flush()

    def run(self, doi, full_text, full_text_type, md_uuid, batch_size=50):
        """
        Run the pipeline.

        Args:
            doi(str): DOI of the paper.
            full_text(str): Full text of the document.
            full_text_type(str): Type of full text, either 'markdown' or 'plain text'.
            md_uuid(str): uuid of the metadata.
            batch_size(int): Size of batch.

        Returns:
            None
        """
        docs = []
        logger.info("=" * 20 + f"Start processing full text of {doi}..." + "=" * 20)
        # Determine which way to split the full text
        try:
            if full_text_type == 'markdown':
                docs = self.markdown_splitter(full_text)
            elif full_text_type == 'plain text':
                docs = self.plaintext_splitter(full_text)
        except TypeError:
            logger.error("No full text found.", exc_info=True)
        else:
            # Load data to Weaviate
            self.load_data(docs, md_uuid, batch_size)
            logger.info("=" * 20 + f"Finish processing full text of {doi}" + "=" * 20)


