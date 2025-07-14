"""
A LLM agent that curates the data extracted from literature
"""
import os
import time
import pandas as pd
from typing import List, Union, Optional, Any
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.callbacks import FileCallbackHandler
# Langchain
from langchain_community.tools import tool
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser, JsonOutputParser
from langchain_core.prompts import load_prompt, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.tools import ToolException
from feptp_pipeline.IE.ie_core import Extractor
from feptp_pipeline.IE.ie_utils import MineralMatcher
from langchain_openai import ChatOpenAI
from langchain_community.tools import ShellTool
import json
import weaviate
import yaml
import logging

# CURRENT_PATH = os.path.dirname(__file__)
# GRANDPARENT_PATH = os.path.abspath(os.path.join(CURRENT_PATH, "../.."))



###################################
# Define output for curated data  #
###################################
class Temperature(BaseModel):
    value: float = Field(default="25")
    greater_than: Optional[bool] = Field(default=False)
    min_value: float = Field(default="25")
    max_value: float = Field(default="25")
    unit: str = Field(default="°C")


class HeatingRate(BaseModel):
    value: float = Field(default="")
    unit: str = Field(default="°C/min")


class Time(BaseModel):
    value: float = Field(default="")
    greater_than: Optional[bool] = Field(default=False)
    min_value: float = Field(default="")
    max_value: float = Field(default="")
    unit: str = Field(default="h")


class Concentration(BaseModel):
    value: float = Field(default="")
    unit: str = Field(default="M")


class PH(BaseModel):
    initial_value: float = Field(default="7")
    final_value: float = Field(default="7")
    min_value: float = Field(default="7", description="Minimum pH value if the pH is a range")
    max_value: float = Field(default="7", description="Maximum pH value if the pH is a range")


class EH(BaseModel):
    value: float = Field(default="", description="Redox potential")
    greater_than: Optional[bool] = Field(default=False)
    unit: str = Field(default="V")


class Pressure(BaseModel):
    value: float = Field(default="0.1")
    unit: str = Field(default="MPa")


class Speed(BaseModel):
    value: float = Field(default="")
    unit: str = Field(default="rpm")


class HeatingCondition(BaseModel):
    temperature: Temperature = Field(default_factory=Temperature)
    time: Time = Field(default_factory=Time)
    heating_rate: HeatingRate = Field(default_factory=HeatingRate)
    solvent: str = Field(default="")
    reactant_or_reagent: List[str] = Field(default_factory=list)
    reactant_or_reagent_concentration: List[Concentration] = Field(default_factory=list,
                                                                   description="concentration of corresponding reactant or reagent")
    atmosphere: str = Field(default="")
    pressure: Pressure = Field(default_factory=Pressure)
    pH: PH = Field(default_factory=PH)
    Eh: EH = Field(default_factory=EH)


class AgingCondition(BaseModel):
    temperature: Temperature = Field(default_factory=Temperature)
    time: Time = Field(default_factory=Time)
    solvent: str = Field(default="water")
    reactant_or_reagent: List[str] = Field(default_factory=list)
    reactant_or_reagent_concentration: List[Concentration] = Field(default_factory=list,
                                                                   description="concentration of corresponding reactant or reagent")
    atmosphere: str = Field(default="")
    pressure: Pressure = Field(default_factory=Pressure)
    pH: PH = Field(default_factory=PH)
    Eh: EH = Field(default_factory=EH)


class GrindingCondition(BaseModel):
    time: Time = Field(default_factory=Time)
    atmosphere: str = Field(default="")
    pressure: Pressure = Field(default_factory=Pressure)
    speed: Speed = Field(default_factory=Speed)
    ball_to_powder_ratio: str = Field(default="")
    reactant_or_reagent: List[str] = Field(default_factory=list)
    reactant_or_reagent_concentration: List[Concentration] = Field(default_factory=list)


class CoolingCondition(BaseModel):
    temperature: Temperature = Field(default_factory=Temperature)
    time: Time = Field(default_factory=Time)
    atmosphere: str = Field(default="")
    pressure: Pressure = Field(default_factory=Pressure)
    quench: bool = Field(default="", description="Whether the sample is quenched")
    quenching_medium: str = Field(default="water")


class DryingCondition(BaseModel):
    temperature: Temperature = Field(default_factory=Temperature)
    time: Time = Field(default_factory=Time)
    atmosphere: str = Field(default="")
    pressure: Pressure = Field(default_factory=Pressure)


class OtherCondition(BaseModel):
    description: str = Field(default="")
    temperature: Temperature = Field(default_factory=Temperature)
    time: Time = Field(default_factory=Time)
    solvent: str = Field(default="water")
    atmosphere: str = Field(default="")
    pressure: Pressure = Field(default_factory=Pressure)
    reactant_or_reagent: List[str] = Field(default_factory=list)
    reactant_or_reagent_concentration: List[Concentration] = Field(default_factory=list)
    pH: PH = Field(default_factory=PH)
    Eh: EH = Field(default_factory=EH)
    ultrasonic: bool = Field(default=False)
    light: bool = Field(default=False)
    microwave: bool = Field(default=False)
    microbio: bool = Field(default=False)
    microbio_species: List[str] = Field(default_factory=list)


class HeatingProcedure(BaseModel):
    type: str = Field(default="heating")
    original_description: str = Field(default="")
    condition: HeatingCondition = Field(default_factory=HeatingCondition)


class AgingProcedure(BaseModel):
    type: str = Field(default="aging")
    original_description: str = Field(default="")
    condition: AgingCondition = Field(default_factory=AgingCondition)


class GrindingProcedure(BaseModel):
    type: str = Field(default="grinding")
    original_description: str = Field(default="")
    condition: GrindingCondition = Field(default_factory=GrindingCondition)


class CoolingProcedure(BaseModel):
    type: str = Field(default="cooling")
    original_description: str = Field(default="")
    condition: CoolingCondition = Field(default_factory=CoolingCondition)


class DryingProcedure(BaseModel):
    type: str = Field(default="drying")
    original_description: str = Field(default="")
    condition: DryingCondition = Field(default_factory=DryingCondition)


class OtherProcedure(BaseModel):
    type: str = Field(default="other")
    original_description: str = Field(default="")
    condition: OtherCondition = Field(default_factory=OtherCondition)


ProcedureType = Union[
    HeatingProcedure, AgingProcedure, GrindingProcedure, CoolingProcedure, DryingProcedure, OtherProcedure]


class MineralPhase(BaseModel):
    name: str = Field(default="", required=True)
    formula: str = Field(default="", required=True)
    original_description: str = Field(default="", required=True)
    defect: bool = Field(default=False)
    doping_species: List[str] = Field(default_factory=list)
    doping_amount: List[str] = Field(default_factory=list, description="Doping percentage or ratio")
    solid_solution: bool = Field(default=False)
    amorphous: bool = Field(default=False, required=True)
    hydrate: bool = Field(default=False)
    adsorbed_species: List[str] = Field(default_factory=list)
    concentration_before_adsorption: List[Concentration] = Field(default_factory=list)


class Reason(BaseModel):
    category: List[str] = Field(default_factory=list, required=True)
    summary: str = Field(default="", description="A brief summary of the reason from original text")


class Pathway(BaseModel):
    precursor_phase: List[MineralPhase] = Field(default_factory=list, description="iron-mineral precursor phase")
    product_phase: List[MineralPhase] = Field(default_factory=list, description="iron-mineral product phase")
    reason: Reason = Field(default_factory=Reason)
    procedure: List[ProcedureType] = Field(default_factory=list)
    changed: bool = Field(default=True, description="Whether the transformation reaction occurs")
    extent_of_transformation: str = Field(default="")
    reaction_equation: str = Field(default="")
    confidence: str = Field(default="")


class CuratedDataOutput(BaseModel):
    doi: str = Field(default="")
    pathway: List[Pathway] = Field(default_factory=list)


class Curator:
    def __init__(self,
                 input_path,
                 output_path,
                 curated_doi_path,
                 missed_doi_path,
                 keynote_path,
                 prompt_path,
                 log_config_path):

        self.input_path = input_path
        self.output_path = output_path
        self.curated_doi_path = curated_doi_path
        self.missed_doi_path = missed_doi_path
        self.prompt_path = prompt_path

        # Set up logging
        with open(log_config_path, "r") as f:
            config = yaml.safe_load(f)
        project_root = log_config_path.parent.parent
        log_file_path = project_root / "logs" / "curator.log"
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        config["handlers"]["file"]["filename"] = str(log_file_path)
        logging.config.dictConfig(config)
        self.logger = logging.getLogger("curator")

        # Initialize LLM
        self.llm = ChatOpenAI(
            temperature=0,
            model='gpt-4o',
            api_key=os.environ["OPENAI_API_KEY"],
            request_timeout=120,
            model_kwargs={"seed": 42, "response_format": {"type": "json_object"}}
        )

        # Initialize extractor
        self.extractor = Extractor(keynote_path=keynote_path)
        self.mineral_matcher = MineralMatcher()

    def run(self):
        input_data = pd.read_json(self.input_path, lines=True)
        input_data = input_data[input_data["exp_info"].apply(lambda x: len(x) > 0)]
        self.logger.info(f"Total records to process: {len(input_data)}")

        if os.path.exists(self.curated_doi_path):
            with open(self.curated_doi_path, 'r') as f:
                curated_dois = [line.strip() for line in f.readlines()]
        else:
            curated_dois = []
            open(self.curated_doi_path, 'w').close()

        count = len(curated_dois)
        prompt_template = load_prompt(self.prompt_path)
        for _, row in input_data.iterrows():
            doi = row['doi']
            if doi in curated_dois:
                self.logger.info(f"{doi} already curated. Skipping.")
                continue
            try:
                curated_data = self.curate_one(row, prompt_template)
                with open(self.output_path, 'a+', encoding='utf-8') as f:
                    f.write(json.dumps(curated_data) + "\n")
                with open(self.curated_doi_path, 'a+') as f:
                    f.write(doi + "\n")
                count += 1
                self.logger.info(f"{count} papers curated.")
            except Exception as e:
                self.logger.error(f"Failed to curate {doi}: {str(e)}")
                with open(self.missed_doi_path, 'a+') as f:
                    f.write(doi + "\n")

    def curate_one(self, row, prompt_template):
        doi = row["doi"]
        pathways = row["exp_info"]
        queries = [q for q in self.extractor.query_constructor(pathways) if q]
        retrieved_texts = []
        for query in queries:
            retrieved_texts.extend(self.extractor._custom_retriever(query, doi))
        retrieved_text = "\n".join(f"{i+1}. {t}" for i, t in enumerate(set(retrieved_texts)))

        mineral_info = str(self.mineral_matcher.match(retrieved_text + str(pathways)))
        parser = PydanticOutputParser(pydantic_object=CuratedDataOutput)
        format_instructions = parser.get_format_instructions()
        prompt = prompt_template.partial(
            original_text=retrieved_text,
            format_instructions=format_instructions,
            mineral_info=mineral_info
        )
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a cautious and responsible curator to curate the data extracted from scientific literature."),
            HumanMessagePromptTemplate(prompt=prompt),
        ])
        chain = chat_prompt | self.llm | JsonOutputParser()
        return chain.invoke({"original_data": {"doi": doi, "exp_info": pathways}})

