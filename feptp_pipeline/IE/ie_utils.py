"""
Utility functions for Information Extraction (IE) tasks.
"""
import os
from typing import Any, List, Dict, Union, Sequence, Optional, Literal
from fuzzywuzzy import process, fuzz
import pandas as pd
from importlib.resources import files
from pydantic import BaseModel, Field, ValidationError


class MineralMatcher:
    """
    A similarity-based name entity matcher that identifies minerals in a paragraph using glossary.
    """

    def __init__(self,
                 kb_path: Optional[os.PathLike] = None,
                 ):
        """
            Normally, entity name should be consistent with the column name of the knowledge base.
        Args:
            kb_path: data path of the mineral glossary.
        """
        if kb_path is None:
            # Default knowledge base path
            self.kb_path = files("feptp_pipeline.resources.kb").joinpath("mineral_glossary.csv")
        else:
            self.kb_path = kb_path

    def match(self,
              paragraph: str,
              match_properties: List[str] = None,
              similarity_threshold: int = 90  # Define a similarity threshold for matching
              ) -> List[Dict[str, Any]]:
        """
            Match minerals in a paragraph and return the most similar results using fuzzy matching.
        Args:
            paragraph: A paragraph of text.
            match_properties: Properties to be matched, default to ['Mineral_Name', 'IMA_Formula', 'IMA_Formula_Simple'].
            similarity_threshold: The minimum similarity score to consider a match.
        Returns:
            A list of dictionaries containing the information of the most similar matched minerals.
        """
        # Split the paragraph into words or phrases
        STOP_WORDS = {"the", "an", "is", "in", "on", "at", "and", "or", "but", "a", "of", "for", "with", "by", "to",
                      "as",
                      "from", "this", "that", "which", "it", "its", "their", "there", "has", "have", "was", "were",
                      "be",
                      "been"}
        words = [word for word in paragraph.lower().split() if word not in STOP_WORDS]
        # Default properties to be matched
        DEFAULT_PROPERTIES = ['Mineral_Name', 'IMA_Formula', 'IMA_Formula_Simple',
                              'Mineral_Alias_1', 'Mineral_Alias_2', 'Mineral_Alias_3', 'Mineral_Alias_4']
        if match_properties is None:
            match_properties = DEFAULT_PROPERTIES

        # Load knowledge base
        mineral_kb = pd.read_csv(self.kb_path)

        # Create a list to store the best matches
        results = []

        for word in words:
            best_match = None
            best_score = 0
            for prop in match_properties:
                if prop in mineral_kb.columns:
                    # Get the best match based on the similarity score
                    matches = process.extractOne(word, mineral_kb[prop].str.lower().tolist())
                    if matches:
                        match, score = matches
                        if score > similarity_threshold and score > best_score:
                            best_match = mineral_kb[mineral_kb[prop].str.lower() == match].to_dict(orient='records')[0]
                            best_score = score
            if best_match:
                results.append(best_match)

        # Remove items with NaN
        results = [{k: v for k, v in d.items() if pd.notna(v)} for d in results]
        # Remove duplicates
        results = [dict(t) for t in {tuple(d.items()) for d in results}]

        return results


# Pydantic data class for information extraction from the head or tail of a paper
class KeyNote(BaseModel):
    """
    A Pydantic data class to validate the extracted mineral phase transformation pathway.
    """
    precursor_phase_name: List[str]
    product_phase_name: List[str]
    precursor_phase_formulas: List[str]
    product_phase_formulas: List[str]
    reason: Optional[Union[List[str], str]] = Field(
        description="brief reason causing the mineral compositional or structural "
                    "changes, use short terminology phrase, choose from phrases below unless none of them fit, "
                    "aqueous reaction, electrochemical,"
                    "roasting, melting, mechanochemistry, hydrothermal, bio-mediated"
                    "transformation")
    conditions: Optional[Union[List[dict], dict, str]] = Field(description="specific conditions when the "
                                                                           "transformation reaction occurred. For example, "
                                                                           "[{'temperature': '80°C','time': '2h', "
                                                                           "'reagent': 'H2SO4', 'concentration': '0.1 M'}, "
                                                                           "{temperature': '80°C', 'atmosphere': 'air', "
                                                                           "'pH': '2', 'doping agent': 'Cu'}] are two set "
                                                                           "of independent condition. Try to use"
                                                                           "conditions mentioned in the text as keys.")


class KeyNotes(BaseModel):
    """
    Collection of all the keynotes of mineral phase transformation pathway in a paper.
    """
    keynotes: Sequence[KeyNote]


class Queries(BaseModel):
    """
    A Pydantic data class to validate the queries generated from the reaction pathways or raw text.
    """
    query: List[str]


class Context(BaseModel):
    """
    A Pydantic data class to validate the generated section titles from LLM.
    """
    context: List[Dict[str, str]] = Field(
        examples=[{"header_2": "Preparation of Hematite"}, {"header_3": "Effect of pH"}])


def wrap_with_wildcards(s):
    s = s.strip()
    if not s.startswith("*"):
        s = "*" + s
    if not s.endswith("*"):
        s = s + "*"
    return s
