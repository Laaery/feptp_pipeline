"""
Core module for information extraction.
"""
import os
from importlib.resources import files
from typing import Any, List, Dict, Union, Sequence, Optional, Literal
# from langchain_community.chat_models import ChatOpenAI
# from langchain_community.llms import GPT4All
from langchain.globals import set_debug, set_verbose
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import HumanMessagePromptTemplate
from langchain.prompts import load_prompt
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field, ValidationError
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from feptp_pipeline.IE.ie_utils import MineralMatcher, Context, Queries, KeyNotes, wrap_with_wildcards
from operator import itemgetter
import weaviate.classes as wvc
import re
# from spacy.language import Language
import json
import logging
import logging.config

# set_debug(True)
# set_verbose(True)

# CURRENT_PATH = os.path.dirname(__file__)
# GRANDPARENT_PATH = os.path.abspath(os.path.join(CURRENT_PATH, "../.."))
logger = logging.getLogger("ie")

class HeadTailScanner:
    """
    A scanner that scans the abstract and conclusion(if any) of a paper to extract information about research object or
    factors for certain process to investigate.
    Another feature is to quickly catch the main ideas of a paper.
    """

    def __init__(self,
                 prompt: Optional[os.PathLike] = None,
                 wv_client: Any = None,
                 kb_path: Optional[os.PathLike] = None,
                 model_version: str = None,
                 ):
        """
        Args:
            prompt: Path of prompt template for information extraction, '/prompt/prompt_ie1.json' by default.
            wv_client: Weaviate client object.
            kb_path: Path to the mineral glossary file, '/data/kb/mineral_glossary.csv' by default.
            model_version: The version of language model to be used. Chatbased or instruction-based. Only
            OPENAI models are supported. If you want to customize the model, please modify the self.llm setup.
        """
        import weaviate
        if prompt is None:
            # Use default prompt file
            self.prompt = files("feptp_pipeline.resources.prompt").joinpath("prompt_extract_keynote.json")
        else:
            self.prompt = prompt
        if kb_path is None:
            # Default pattern file path
            self.kb_path = files("feptp_pipeline.resources.kb").joinpath("mineral_glossary.csv")
        else:
            self.kb_path = kb_path
        if not isinstance(wv_client, weaviate.WeaviateClient):
            raise TypeError("Please pass an weaviate client(v4) instance. Try again.")
        else:
            self.wv_client = wv_client
        if model_version is None:
            self.model_version = "gpt-3.5-turbo"
        else:
            self.model_version = model_version
        # Load hand-crafted prompt from json file
        self.prompt_template_scan = load_prompt(self.prompt)

        # Set up language model
        self.llm = ChatOpenAI(temperature=0,
                              model=self.model_version,
                              openai_api_key=os.environ["OPENAI_API_KEY"],
                              request_timeout=60,
                              model_kwargs={"seed": 42, "response_format": {"type": "json_object"}}
                              )
        # Logger
        self.logger = logging.getLogger("ie.headtailscan")


    def scan(self,
             doi: str,
             head_included: Optional[bool] = True,
             tail_included: Optional[bool] = False,
             storage_path: str = None,
             ) -> KeyNotes:
        """
        Scan the abstract and conclusion(if any) of a paper and extract information as a KeyNotes object.
        At least one of head_included and tail_included must be True.
        Args:
            doi: The doi of the paper to be scanned.
            head_included: Whether to scan the abstract of a paper.
            tail_included: Whether to scan the conclusion of a paper.
            storage_path: Path to store the scan results. Default to None.
        Returns:
            A KeyNotes object containing the information extracted from the paper.
        """
        self.logger.info(f"Start scanning paper {doi}...")
        # Set up a Pydantic parser and inject instructions into the prompt template.
        parser = PydanticOutputParser(pydantic_object=KeyNotes)

        # Prompt for stage 1 information extraction(IE)
        prompt_template = ChatPromptTemplate.from_messages(
            [("system", "You are an expert proficient in chemistry, environmental "
                        "science and mineralogy."),
             HumanMessagePromptTemplate(prompt=self.prompt_template_scan)
             ])

        # Check if at least one of head_included and tail_included is True
        if not (head_included or tail_included):
            raise ValueError("At least one of head_included and tail_included must be True.")
        text = ""
        if head_included:
            text += "abstract: " + self._get_abstract(doi)
        if tail_included:
            text += "conclusion: " + self._get_conclusion(doi)

        mineral_info = self.match_minerals(text)
        self.logger.info(f"Augmented mineral info from IMA knowledge base: {mineral_info}...")

        # Logging prompt
        prompt = prompt_template.format_messages(mineral_info=mineral_info,
                                                 format_instructions=parser.get_format_instructions(),
                                                 text=text)
        self.logger.info(f"Prompt: {prompt}")

        # Run IE
        try:
            response = self.llm(prompt)
        except Exception as e:
            self.logger.error(f"Error occurred during scanning: {e}")
        try:
            # Return parsed information
            keynotes = parser.parse(response.content)
            self.to_disk(doi, keynotes, storage_path)
        # Deal with validation errors
        except ValidationError as e:
            self.logger.error(f"Pydantic validation error: {e}")
            self.logger.info(f"Unformatted LLM response: {response.content.strip()}")

        return keynotes

    def _get_abstract(self, doi: str):
        """
        Fetch the abstract of a paper from Weaviate database.
        """
        metadata = self.wv_client.collections.get("Metadata")
        response = metadata.query.fetch_objects(
            filters=wvc.Filter("doi").equal(doi),
            limit=1
        )
        abstract = response.objects[0].properties['abstract']
        # Clean annoying empty spaces
        abstract = re.sub(r'\s+', ' ', abstract)
        return abstract

    # Get abstract throughout the database
    def _get_conclusion(self, doi: str):
        """
        Fetch the abstract of a paper from Weaviate database.
        """
        metadata = self.wv_client.collections.get("Metadata")
        fulltext = self.wv_client.collections.get("FullText")
        keyword_list = ["conclusion", "summary"]
        response = fulltext.query.fetch_objects(
            filters=wvc.Filter("header_1").contains_any(keyword_list) & wvc.Filter(
                ["hasMetadata", "Metadata", "doi"]).equal(doi),
        )
        conclusion = ""
        # if response not empty
        if response:
            for o in response.objects:
                conclusion += o.properties["text"] + " "
        else:
            print("No conclusion found.")
        # Clean annoying empty spaces
        conclusion = re.sub(r'\s+', ' ', conclusion)
        return conclusion

    def match_minerals(self, text: str):
        """
        Match minerals in the context via external data source.
        """
        mineral_matcher = MineralMatcher(kb_path=self.kb_path)

        return mineral_matcher.match(text)

    def to_disk(self, doi, keynotes, path: str = None):
        """
        Save the keynotes to disk.
        """
        if path is not None:
            # Save scan results to jsonl
            with open(path, 'a+') as f:
                # Convert keynote object to dictionary
                keynotes_dict = keynotes.model_dump()
                line = json.dumps({"doi": doi, "keynotes": keynotes_dict["keynotes"]}) + '\n'
                f.write(line)
                self.logger.info(f"{doi} is successfully scanned.")


class Extractor:
    """
    A class that extracts information from a paper.
    """

    def __init__(self,
                 wv_client: Any = None,
                 output_folder: Union[str, os.PathLike] = None,
                 model_version: str = None,
                 ):
        """

        Args:
            wv_client: Weaviate client
            output_folder: Output folder for extracted information
            model_version: The version of language model to be used. Currently OPENAI and Google models are supported.

        """
        import weaviate
        self.wv_client = wv_client
        if not isinstance(self.wv_client, weaviate.WeaviateClient):
            raise TypeError("Please pass an weaviate client(v4) instance. Try again.")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        self.output_folder = output_folder
        if model_version is None:
            self.model_version = "gpt-3.5-turbo"
        else:
            self.model_version = model_version
        self._init_model()

    def _init_model(self):
        if self.model_version.startswith("gpt"):
            self.llm = ChatOpenAI(temperature=0,
                                  model=self.model_version,
                                  openai_api_key=os.environ["OPENAI_API_KEY"],
                                  request_timeout=60,
                                  model_kwargs={"seed": 42, "response_format": {"type": "json_object"}}
                                  )
            self.llm_str = ChatOpenAI(temperature=0,
                                      model=self.model_version,
                                      openai_api_key=os.environ["OPENAI_API_KEY"],
                                      request_timeout=60,
                                      model_kwargs={"seed": 42}
                                      )  # Response format is not specified, so the model will return the raw text.

        elif self.model_version.startswith("gemini"):
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                from google.generativeai.types import HarmBlockThreshold
                from google.ai.generativelanguage_v1 import HarmCategory
                from google.generativeai.types import HarmBlockThreshold
            except ImportError:
                raise ImportError(
                    "This feature requires 'langchain_google_genai' and 'google-generativeai'.\n"
                    "Please install them with:\n\n"
                    "  pip install langchain-google-genai google-generativeai\n"
                )
            if os.getenv("GOOGLE_API_KEY") is None:
                raise ValueError("Please set up your Google API key in environment variables.")
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest",
                                              google_api_key=os.getenv("GOOGLE_API_KEY"),
                                              transport="rest",
                                              convert_system_message_to_human=True,
                                              safety_settings={
                                                  HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                                                  HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                                  HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                                                  HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                                              })
        else:
            raise ValueError("Model version not supported!")

    def _custom_retriever(self,
                          query: str,
                          doi: str,
                          ):
        """
        Retrieve the context-related sections from Weaviate.

        Args:
            query(str): The query to search for the context.
            doi(str): The doi of the paper to search for the context.

        Returns:
            str: Context most relevant in specific section.
        """
        import re
        # Get collections
        fulltext = self.wv_client.collections.get("FullText")
        # Choose the most relevant section based on the query, return a dict object
        context = self.choose_context(doi, query)
        print("Most relevant sections in this paper:", context)
        retrieved_text = []
        if context == "No available full text":
            print("No available full text")
            return retrieved_text
        elif context == "No headers information":
            response = fulltext.query.near_text(
                query=query,
                filters=wvc.Filter(["hasMetadata", "Metadata", "doi"]).equal(doi),
                limit=3
            )
            if response.objects:
                for i, obj in enumerate(response.objects):
                    retrieved_text.append(obj.properties["text"].strip())
            elif not response.objects or all([not o.properties for o in response.objects]):
                print("No relevant context found")
        else:
            # Context header might be wrong because of the limitation of the model
            try:
                responses = []
                for context_item in context["context"]:
                    k, v = next(iter(context_item.items()))
                    v_clean = wrap_with_wildcards(re.sub(r"^\d+\s*", "", v))

                    # Try original and adjacent headers
                    headers_to_try = [k]
                    if k.startswith('header_'):
                        header_num = int(k.split('_')[1])
                        if header_num > 1:
                            headers_to_try.append(f"header_{header_num - 1}")  # Higher level
                        headers_to_try.append(f"header_{header_num + 1}")  # Lower level
                    for header in headers_to_try:
                        response = fulltext.query.near_text(
                            query=query,
                            filters=wvc.Filter(header).like(v_clean) &
                                    wvc.Filter(["hasMetadata", "Metadata", "doi"]).equal(doi),
                            limit=1
                        )
                        if response.objects:
                            responses.append(response)
                            break
                        else: # Fallback: Try without header filter
                            response = fulltext.query.near_text(
                                query=query,
                                filters=wvc.Filter(["hasMetadata", "Metadata", "doi"]).equal(doi),
                                limit=1
                            )
                            if response.objects:
                                responses.append(response)

                # Sometimes objects returned by Weaviate are empty
                for response in responses:
                    if response.objects:
                        for i, obj in enumerate(response.objects):
                            retrieved_text.append(obj.properties["text"].strip())
            # If the context header is wrong, search for the query in the whole text
            except Exception as e:
                response = fulltext.query.near_text(
                    query=query,
                    filters=wvc.Filter(["hasMetadata", "Metadata", "doi"]).equal(doi),
                    limit=3
                )
                if response.objects:
                    for i, obj in enumerate(response.objects):
                        retrieved_text.append(obj.properties["text"].strip())
                elif not response.objects or all([not o.properties for o in response.objects]):
                    print("No relevant context found")
        return retrieved_text

    def custom_retriever(self, _dict):
        """
        A wrapper function that accepts a single input and unpacks it into multiple argument.

        Args:
            _dict(dict): A dictionary containing the query and doi

        Returns:
            Result of private function _custom_retriever
        """
        return self._custom_retriever(_dict["query"], _dict["doi"])

    def choose_context(self,
                       doi: str,
                       query: str = None
                       ):
        """
        Determine the context based on the user's intention and query.
        Args:
            doi: The doi of the paper to be scanned.
            query: The query to search for.
        Returns:
            A list of headers or sub-headers.
        """
        default_prompt = files("feptp_pipeline.resources.prompt").joinpath("prompt_choose_context.json")
        prompt = load_prompt(default_prompt)
        # Only text with headers and not supplementary material works
        fulltext = self.wv_client.collections.get("FullText")
        # List out all headers and sub-headers
        response = fulltext.query.fetch_objects(
            filters=wvc.Filter(["hasMetadata", "Metadata", "doi"]).equal(doi),
            return_properties=["header_1", "header_2", "header_3"],
            limit=1000
        )
        # Retain tree hierarchy of headers as dictionary
        # tree_dict = {
        #     1: {
        #         2: {
        #             4: {},
        #         },
        #         3: {},
        #     }
        # }
        content = {}
        if not response.objects:
            return "No available full text"
        elif all([not o.properties for o in response.objects]):
            return "No headers information"
        else:
            for obj in response.objects:
                # Not every object has headers
                if "header_1" in obj.properties:
                    header_1 = obj.properties["header_1"]
                    if header_1 not in content and header_1 != "supplementary material":
                        content[header_1] = {}
                else:
                    continue
                if "header_2" in obj.properties:
                    header_2 = obj.properties["header_2"]
                    if header_2 not in content[header_1]:
                        content[header_1][header_2] = {}
                else:
                    continue
                if "header_3" in obj.properties:
                    header_3 = obj.properties["header_3"]
                    if header_3 not in content[header_1][header_2]:
                        content[header_1][header_2][header_3] = {}
        # Convert the content to LLM readable format
        # Reconstruct the content to hierarchical format organized by /n
        content = "\n".join([f"{k1}\n" + "\n".join([f"  {k2}\n" + "\n".join([f"    {k3}" for k3 in v2.keys()])
                                                    for k2, v2 in v1.items()]) for k1, v1 in content.items()])
        # print("Content: " + content)
        # Prompt
        format_instructions = PydanticOutputParser(pydantic_object=Context).get_format_instructions()
        prompt = prompt.partial(format_instructions=format_instructions)
        prompt_template = ChatPromptTemplate.from_messages(
            [("system", "You are an expert proficient in chemistry, environmental "
                        "science and mineralogy."),
             HumanMessagePromptTemplate(prompt=prompt)
             ])

        # Set up language model
        # model = ChatOpenAI(temperature=0,
        #                    model="gpt-3.5-turbo",
        #                    openai_api_key=os.environ["OPENAI_API_KEY"],
        #                    request_timeout=60,
        #                    model_kwargs={"seed": 42, "response_format": {"type": "json_object"}}
        #                    )
        model = self.llm
        # Most likely context window given by LLM (Default to 3)
        choose_context_chain = ({"query": itemgetter("query"), "content": itemgetter("content")}
                                | prompt_template
                                | model
                                | JsonOutputParser())
        response = choose_context_chain.invoke({"query": query, "content": content})
        return response

    def query_constructor(self,
                          pathways: List[dict] = None):
        """
        Construct queries for RAG based on the information from pathways or raw text.

        Args:
            pathways(list[dict]): pathways from extracted data or raw text.

        Returns:
            A list of queries.
        """
        default_prompt = files("feptp_pipeline.resources.prompt").joinpath("prompt_query_constructor.json")
        prompt = load_prompt(default_prompt)
        # model = ChatOpenAI(temperature=0,
        #                    model="gpt-3.5-turbo",
        #                    openai_api_key=os.environ["OPENAI_API_KEY"],
        #                    request_timeout=30,
        #                    model_kwargs={"seed": 42, "response_format": {"type": "json_object"}}
        #                    )
        model = self.llm
        format_instructions = PydanticOutputParser(pydantic_object=Queries).get_format_instructions()
        chain = prompt | model | JsonOutputParser()
        response = chain.invoke({"pathways": pathways, "format_instructions": format_instructions})
        # Convert json to a python list
        try:
            queries = response["query"]
        except KeyError:
            # try again with different model
            model = ChatOpenAI(temperature=0,
                               model="gpt-4o",
                               openai_api_key=os.environ["OPENAI_API_KEY"],
                               request_timeout=30,
                               model_kwargs={"seed": 42, "response_format": {"type": "json_object"}}
                               )
            chain = prompt | model | JsonOutputParser()
            response = chain.invoke({"pathways": pathways, "format_instructions": format_instructions})
            queries = response["query"]
        return queries

    def full_text_extract(self,
                          doi: str,
                          text: str,
                          table: str = None,
                          prompt: Union[os.PathLike, str] = None,
                          keynote_path: Optional[Union[str, os.PathLike]] = None
                          ):
        """
            Extract detailed information from the full text of a paper.
        Args:
            doi: The doi of the paper to be extracted.
            text: The full text of the paper.
            table: The table content of the paper.
            prompt: The path of the prompt template for information extraction.
            keynote_path: The path of the keynotes from the head-tail scanner module.

        Returns:
            JSON file containing the details.
        """
        if keynote_path is not None and os.path.exists(keynote_path):
            with open(keynote_path, 'r', encoding='utf-8') as f:
                keynotes = f.readlines()
            try:
                keynotes = [json.loads(d) for d in keynotes if json.loads(d)["doi"] == doi]
            # Keynote should not be empty
            except IndexError:
                raise IndexError(f"The keynote is empty for paper {doi}")
        else:
            keynotes = [{"keynotes": []}]

        logger.info(f"Execute extracting details from paper {doi}...")
        model = self.llm
        if isinstance(prompt, (str, os.PathLike)) and os.path.isfile(prompt):
            prompt = load_prompt(prompt)
        else:
            # Use default prompt file
            print("Using default prompt file for extraction.")
            prompt = files("feptp_pipeline.resources.prompt").joinpath("prompt_extract_full.json")
        parser = JsonOutputParser()
        chain = prompt | model | parser
        response = chain.invoke({'keynote': keynotes[0]['keynotes'], 'text': text, 'table': table})
        # Save as csv file name by doi
        with open(f'{self.output_folder}/ext_info.jsonl', 'a+', encoding='utf-8') as f:
            f.write(json.dumps({"doi": doi, "ext_info": response}) + '\n')
        return response

    def extract(self,
                doi: str,
                query: Optional[Union[str, os.PathLike]],
                prompt: os.PathLike
                ) -> str:
        """
        This function performs a flexible, LLM-driven extraction task. It accepts a free-form natural language query
        and returns an answer derived from the most relevant section(s) of the input paper. Internally, it uses
        large language models (LLMs) or vector search (if provided) to locate, interpret, and synthesize answers.

        Args:
            doi: The doi of the paper to be scanned.
            query: The query to search for the context, or the json file path
            of the query file. The json file should contain a dictionary with the following key-value pairs: {"query": "query text"}.
            prompt: path of prompt template for information extraction.

        Returns:
            A string containing the details the user asks for.
        """
        # Require a query to search for the context
        if query is None:
            raise ValueError("Please provide a query to search for the context.")
        elif os.path.isfile(query):
            # Load query from json file
            with open(query, 'r') as f:
                content = json.load(f)
                query = content["query"]
        elif isinstance(query, str):
            pass

        if isinstance(prompt, (str, os.PathLike)) and os.path.isfile(prompt):
            prompt = load_prompt(prompt)
        else:
            raise ValueError("Prompt must be a valid file path.")
        model = self.llm_str

        # Prompt for information extraction(IE)
        prompt_template = ChatPromptTemplate.from_messages(
            [("system", "You are an expert proficient in chemistry, environmental "
                        "science and mineralogy."),
             HumanMessagePromptTemplate(prompt=prompt),
             ])

        def aggregate_texts(texts: List[str]) -> str:
            # Concatenate all texts into a single string
            agg_texts = "\n".join(texts)
            logger.info(f"Retrieved texts: {agg_texts[:1000]}...")  # Log first 1000 characters
            return agg_texts

        retrieval_chain = ({"query": itemgetter("query"), "text": {"query": itemgetter("query"), "doi": itemgetter("doi")} | RunnableLambda(
                self.custom_retriever) | RunnableLambda(aggregate_texts)} | prompt_template | model | StrOutputParser())

        response = retrieval_chain.invoke({"query": query, "doi": doi})
        return response

    def extract_table2json(self,
                           doi: str,
                           prompt: Union[List[str], List[os.PathLike]],
                           schema: str = None,
                           save_path: str = None):
        """
        Extract experimental conditions and results from tables to a json file. First, LLM is instructed to choose the
        proper table based on the prompt. Then, the table is extracted based on the schema and the second prompt.

        Args:
            doi: The doi of the paper.
            prompt: The prompt to obtain key information from table.
            schema: The schema of the json file to be saved.
            save_path: The path to store the json file.

        Returns:
            A json file containing the experimental conditions and results from tables.
        """

        prompt = [load_prompt(i) for i in prompt]
        # load schema as a string from json file
        with open(schema, 'r') as f:
            schema = f.read()

        tables = self.wv_client.collections.get("Table")
        # Convert table to json
        table_content = tables.query.fetch_objects(
            filters=wvc.Filter(["hasMetadata", "Metadata", "doi"]).equal(doi),
            return_properties=["caption", "tbody"],
        )
        # Set up language model
        model = self.llm_str
        # First, use the first prompt to determine which table to extract
        prompt_1 = ChatPromptTemplate.from_messages(
            [("system", "You are an expert proficient in chemistry, environmental "
                        "science and mineralogy."),
             HumanMessagePromptTemplate(prompt=prompt[0]),
             ])
        table_list = []
        # Generate a list of table captions and tbody
        for i, obj in enumerate(table_content.objects):
            table_list.append(
                "caption:" + obj.properties["caption"] + "\n" + "main body of table:" + obj.properties["tbody"])
        chain_1 = ({"table": RunnablePassthrough()} | prompt_1 | model | StrOutputParser())
        response_1 = chain_1.batch(table_list)
        # Convert scores to numbers
        response_1 = [float(score) for score in response_1]
        print(response_1)
        # Select the table with the highest score
        table_index = response_1.index(max(response_1))
        print(table_index)

        # Then, use the second prompt template with schema to extract the table
        prompt_2 = ChatPromptTemplate.from_messages(
            [("system", "You are an expert proficient in chemistry, environmental "
                        "science and mineralogy."),
             HumanMessagePromptTemplate(prompt=prompt[1]),
             ])
        # According to the response from the first prompt, fetch the body of table
        tbody = table_content.objects[table_index].properties["tbody"]
        prompt_2 = prompt_2.partial(schema=schema)
        chain_2 = ({"tbody": RunnablePassthrough()} | prompt_2 | model | StrOutputParser())
        response_2 = chain_2.invoke(tbody)

        # Save as json file
        if save_path is not None:
            with open(save_path, 'a+', encoding='utf-8') as f:
                # add doi to the json file
                f.write(response_2 + '\n')

        return response_2
