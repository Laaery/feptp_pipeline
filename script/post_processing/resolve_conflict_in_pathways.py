#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time: 2024/7/8 20:48
# @Author: LL
# @File：resolve_conflict_in_pathways.py
"""
Resolve conflicts in the transformation pathways.
"""
import os
import pandas as pd
from typing import List, Any, Tuple
from tqdm import tqdm
import pubchempy as pcp
import copy
from feptp_pipeline.IE.ie_core import MineralMatcher
import argparse


CURRENT_PATH = os.path.dirname(__file__)
GRANDPARENT_PATH = os.path.abspath(os.path.join(CURRENT_PATH, "../.."))


def normalize_name_and_formula(name: str, formula: str) -> Tuple[str, str]:
    """
    Normalize the name and formula of the phase.
    Args:
        name(str): The name of the phase.
        formula(str): The formula of the phase.
    Returns:
        output(Tuple[str, str]): The normalized name and formula of the phase.
    """
    ion_2 = ["Fe2+", "Fe(2+)", "Fe(II)", "FeII", "Fe(II)aq", "Fe(II)(aq)", "Fe^2+", "Fe^(2+)", "Fe^{2+}"]
    ion_3 = ["Fe3+", "Fe(3+)", "Fe(III)", 'FeIII', "Fe(III)aq", "Fe(III)(aq)", "Fe^3+", "Fe^(3+)", "Fe^{3+}"]
    matcher = MineralMatcher()
    if name in ion_2:
        norm_name = "iron(II) ions"
        norm_formula = "Fe2+"
    elif name in ion_3:
        norm_name = "iron(III) ions"
        norm_formula = "Fe3+"
    elif name == "Iron" and formula != "Fe":
        # Use pubchempy to get the IUPAC name of the formula
        # Replace /u00b7 with . in the formula
        if matcher.match(formula):
            norm_name = matcher.match(formula)[0]["Mineral_Name"]
            norm_formula = formula
        elif pcp.get_compounds(formula.replace("\u00b7", "."), 'name'):
            iupac_name = pcp.get_compounds(formula.replace("\u00b7", "."), 'name')[0].iupac_name
            if iupac_name:
                norm_name = iupac_name
                norm_formula = formula
        else:
            norm_name = formula
            norm_formula = formula
    else:
        norm_name = name
        if matcher.match(norm_name):
            norm_formula = matcher.match(norm_name)[0]["IMA_Formula_Simple"]
        else:
            norm_formula = formula
    return norm_name, norm_formula


def resolve_conflict_in_pathways(data: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve conflict in the transformation pathways.
    Args:
        data(DataFrame): The DataFrame of transformation pathways.
    Returns:
        output(DataFrame): The DataFrame of resolved transformation pathways.
    """
    output = []
    conflict_count = {'type1': 0, 'type2': 0, 'type3': 0, 'type4': 0, 'type5': 0, 'type6': 0, 'type7': 0}
    pathways = data["pathway"]
    for i, pathway in enumerate(tqdm(pathways)):
        precursor_name = [phase['name'] for phase in pathway['precursor_phase']]
        precursor_formula = [phase['formula'] for phase in pathway['precursor_phase']]
        product_name = [phase['name'] for phase in pathway['product_phase']]
        product_formula = [phase['formula'] for phase in pathway['product_phase']]
        precursor_original_description = [phase['original_description'] for phase in pathway['precursor_phase']]
        product_original_description = [phase['original_description'] for phase in pathway['product_phase']]
        # New pathway
        new_pathways = []
        # TYPE 1
        if pathway['changed']:
            # Check if the name and formula of precursors and products are the same
            # For now, only significant structural and composition differences are considered. Minor variations on
            # defects or morphology are not considered.
            if precursor_name == product_name and precursor_formula == product_formula:
                conflict_count['type1'] += 1
                print(f"### 1 ### Pathway has the same precursor and product phase.")
                print(f"Precursor: {precursor_name}, {precursor_formula}, {precursor_original_description}")
                print(f"Product: {product_name}, {product_formula}, {product_original_description}")
                # Decision to drop the pathway or revise. If the user wants to revise, input the corrected pathway,
                # otherwise, just press enter to drop the pathway.
                corrected_pathway = input("Please input the corrected pathway if you want to revise else press enter"
                                          "to drop the pathway: ")
                if corrected_pathway:
                    # Update the pathway as dictionary
                    pathway = eval(corrected_pathway)
                else:
                    continue
        # TYPE 2
        # Check if the same procedures repeat within 1 pathway
        procedures = [procedure['type'] for procedure in pathway['procedure']]
        if len(procedures) != len(set(procedures)):  # Specific procedures repeat
            # Check if the procedure type is the same and only minimum variation exists
            # Consult to human expert for further resolution
            print(f"### 2 ### Pathway has repeated procedures.")
            print(f"Procedures: {procedures}")
            conflict_count['type2'] += 1
            # Human input
            # print("Please input the corrected procedures:")
            # corrected_procedures = input()
            # pathway['procedures'] = corrected_procedures
            # Drop the pathway
            continue
        # TYPE 3
        # Resolve conflicts in temperature based on common knowledge of reason and procedure. For example,
        # the temperature for heating procedure during melting is normally higher than 25 degree Celsius,
        # while the temperature for heating procedure during biotransformation is normally equal to or lower than 100
        # degree Celsius.
        if "heating" in procedures:
            for procedure in pathway['procedure']:
                if procedure['type'] == "heating" and 'condition' in procedure:
                    condition = procedure['condition']
                    # Check if the reason is related to melting
                    if pathway['reason']['category'] == "Melting":
                        if 'temperature' in condition:
                            temperature = condition['temperature']
                            # If temperature value is none, pass
                            if temperature.get('value') is None and temperature.get('max_value') is None:
                                continue
                            value = temperature.get('value', 26)
                            max_value = temperature.get('max_value', 26)
                            if (value is not None and value <= 25) or (max_value is not None and max_value <= 25):
                                print(f"### 3 ### Melting pathway has a heating temperature lower than 25 degree Celsius.")
                                print(f"Pathway: {pathway}")
                                conflict_count['type3'] += 1
                                corrected_pathway = input("Please input the corrected pathway: ")
                                # If the user does not want to correct the pathway, just press enter
                                if corrected_pathway:
                                    # Update the pathway as dictionary
                                    pathway = eval(corrected_pathway)
                                else:
                                    continue
                    # Check if the reason is related to biotransformation
                    elif pathway['reason']['category'] == "Biotransformation":
                        if 'temperature' in condition:
                            temperature = condition['temperature']
                            if temperature.get('value') is None:
                                continue
                            value = temperature.get('value', 99)
                            if value is not None and value > 100:
                                print(
                                    f"### 3 ### Biotransformation pathway has a heating temperature higher than 100 degree Celsius.")
                                conflict_count['type3'] += 1
                                print(f"Pathway: {pathway}")
                                corrected_pathway = input("Please input the corrected pathway: ")
                                # if the user does not want to correct the pathway, just press enter
                                if corrected_pathway:
                                    # Update the pathway as dictionary
                                    pathway = eval(corrected_pathway)
                                else:
                                    continue
        # TYPE 4
        # Conflict between formula and name of the phases, particularly for those named with "Iron" while the formula is
        # not "Fe". For example, {"name": "Iron", "formula": "Fe3(NO3)3"} should be corrected to {"name":
        # " Iron(III) nitrate", "formula": "Fe(NO3)3"}.
        for phase_name, formula in zip(precursor_name + product_name, precursor_formula + product_formula):
            norm_name, norm_formula = normalize_name_and_formula(phase_name, formula)
            if phase_name != norm_name or formula != norm_formula:
                print(f"### 4 ### Pathway has conflict between formula and name of the phases.")
                print(f"Original: {phase_name}, {formula}")
                print(f"Normalized: {norm_name}, {norm_formula}")
                conflict_count['type4'] += 1
                # Replace the original name and formula in pathway with the normalized name and formula
                for phase in pathway['precursor_phase']:
                    if phase['name'] == phase_name:
                        phase['name'] = norm_name
                        phase['formula'] = norm_formula
                for phase in pathway['product_phase']:
                    if phase['name'] == phase_name:
                        phase['name'] = norm_name
                        phase['formula'] = norm_formula
        # TYPE 5
        # Remove pH in the pathway if the reason is solid-state reaction
        if pathway['reason']['category'] == "Solid-State Reaction":
            for procedure in pathway['procedure']:
                if 'condition' in procedure and 'pH' in procedure['condition']:
                    print(f"### 5 ### Solid-state transformation pathway has pH.")
                    conflict_count['type5'] += 1
                    # Remove pH in the pathway
                    for operation in pathway['procedure']:
                        if 'condition' in operation:
                            condition = operation['condition']
                            try:
                                if (condition['pH']["initial_value"] == 7
                                        and condition['pH']["final_value"] == 7):
                                    del condition['pH']
                                else:
                                    pathway['reason']['category'] = "Dissolution-Precipitation Reaction"
                                    print(f"Pathway (SSR -> DPR): {pathway}")
                            except KeyError:
                                print(f"Pathway: {pathway}")
        # TYPE 6
        # Different length of reactant_or_reagent and reactant_or_reagent_concentration
        condition = [operation['condition'] for operation in pathway['procedure'] if 'condition' in operation]
        for c in condition:
            if 'reactant_or_reagent' in c and 'reactant_or_reagent_concentration' in c:
                if len(c['reactant_or_reagent']) > len(c['reactant_or_reagent_concentration']) and c['reactant_or_reagent_concentration']:
                    print(f"### 6 ### Pathway has different length of reactant_or_reagent and reactant_or_reagent_concentration.")
                    print(f"Lack of concentration: {pathway}")
                    conflict_count['type6'] += 1
                    continue
                elif len(c['reactant_or_reagent']) < len(c['reactant_or_reagent_concentration']):
                    # Normally, it is probably a preference of LLM generation to merge 2 pathways into 1 pathway Try
                    # to split the original pathway into 2 pathways For example, {"reactant_or_reagent": ["NaCO3"],
                    # "reactant_or_reagent_concentration": [{"value": 0.5, "unit": "M"}, {"value": 1, "unit": "M"}]}
                    # can be split into 2 pathways: {"reactant_or_reagent": ["NaCO3"],
                    # "reactant_or_reagent_concentration": [{"value": 0.5, "unit": "M"}]} and {"reactant_or_reagent":
                    # ["NaCO3"], "reactant_or_reagent_concentration": [{"value": 1, "unit": "M"}]}
                    print(f"### 6 ### Pathway has different length of reactant_or_reagent and reactant_or_reagent_concentration.")
                    print(f"Splitting...")
                    conflict_count['type7'] += 1
                    for j, conc in enumerate(c['reactant_or_reagent_concentration']):
                        pathway_copy = copy.deepcopy(pathway)
                        pathway_copy['reactant_or_reagent_concentration'] = [conc]
                        pathway_copy['reactant_or_reagent'] = [c['reactant_or_reagent'][j]] if j < len(
                            c['reactant_or_reagent']) else []
                        for operation in pathway_copy['procedure']:
                            if 'condition' in operation:
                                condition = operation['condition']
                                if 'reactant_or_reagent_concentration' in condition:
                                    condition['reactant_or_reagent_concentration'] = pathway_copy[
                                        'reactant_or_reagent_concentration']
                        print(f"Split pathway: {pathway_copy}")
                        new_pathways.append(pathway_copy)
            if new_pathways:
                for new_pathway in new_pathways:
                    output.append(
                        {"doi": data["doi"][i], "pathway": new_pathway, "published_date": data["published_date"][i]})
            else:
                output.append(
                {"doi": data["doi"][i], "pathway": pathway, "published_date": data["published_date"][i]})
    print(f"Conflict resolved.")
    print(f"Type 1 conflict: {conflict_count['type1']}")
    print(f"Type 2 conflict: {conflict_count['type2']}")
    print(f"Type 3 conflict: {conflict_count['type3']}")
    print(f"Type 4 conflict: {conflict_count['type4']}")
    print(f"Type 5 conflict: {conflict_count['type5']}")
    print(f"Type 6 conflict: {conflict_count['type6']}")

    return output


def main():
    arg_parser = argparse.ArgumentParser(description="Resolve conflict in transformation pathways.")
    arg_parser.add_argument('--input_path', type=str,
                            default=os.path.join(GRANDPARENT_PATH, 'data/curation/pathways_final.jsonl'),
                            help="Path to the input pathways JSONL file.")
    arg_parser.add_argument('--output_path', type=str,
                            default=os.path.join(GRANDPARENT_PATH, 'data/curation/pathways_final.jsonl'),
                            help="Path to save the resolved pathways JSONL file.")
    args = arg_parser.parse_args()
    # Load the transformation pathways
    data = pd.read_json(args.input_path, lines=True)
    resolved_pathways = resolve_conflict_in_pathways(data)
    # Length of resolved data
    print("Number of resolved pathways: ", len(resolved_pathways))
    # Save the resolved data
    pd.DataFrame(resolved_pathways).to_json(args.output_path, lines=True, orient='records')


if __name__ == '__main__':
    main()

