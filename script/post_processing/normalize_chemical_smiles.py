"""
This script is used to add and normalize the SMILES of the reactants and solvents in the pathway file. Another feature is
to supplement the missing reactants or reagents in the pathway based on the reaction equation.
"""
import pandas as pd
import os
from typing import List, Any
import pubchempy as pcp
import re
from tqdm import tqdm
from rdkit import Chem
import argparse

CURRENT_PATH = os.path.dirname(__file__)
GRANDPARENT_PATH = os.path.abspath(os.path.join(CURRENT_PATH, "../.."))


def chemical_name2smiles(chemical_names) -> List[dict]:
    """
    Find the SMILES of the chemical_name from PubChem database and update the entry.
    """
    new_entries = []
    if not chemical_names:
        return new_entries
    if isinstance(chemical_names, str):
        chemical_names = [chemical_names.replace("solution", "").strip()]
        # chemical_names = [chemical_names.split(" ").strip()]
        # # Remove "solution" from the chemical name
        # chemical_names = [chemical_name for chemical_name in chemical_names if chemical_name != "solution"]
    for chemical_name in chemical_names:
        # Find the SMILES of the chemical_name from PubChem database
        entry = {}
        try:
            compound = pcp.get_compounds(chemical_name, 'name')[0]
            entry["name"] = chemical_name
            # entry["formula"] = compound.molecular_formula
            entry["smiles"] = compound.canonical_smiles
        except:
            entry["name"] = chemical_name
            # entry["formula"] = None
            entry["smiles"] = None
        new_entries.append(entry)
    return new_entries


def add_missing_reactants_reagents(reaction_equation: str, reactants_reagents: List[dict]) -> List[dict]:
    """
    Add the missing reactants or reagents in the pathway based on the reaction equation.
    """
    # Split the input into individual equations
    equations = re.split(r'\s*;\s*', reaction_equation)

    for eq in equations:
        # Remove state of phase in parentheses and other unwanted characters
        eq_normalized = re.sub(r'\s?\(g\)|\(aq\)|\(l\)|\(s\)|\(solid\)|\(liquid\)|\(gas\)', '', eq)
        # Ensure there's a space between every term and symbol
        eq_normalized = re.sub(r'([+-]=>)', r' \1 ', eq_normalized)
        # Split the equation into reactants and products by '->'. Set the maximum split to 1 to deal with the case
        # where the reactant contains many '->' to show the intermediate steps, for example, 2Fe(OH)3 -> 2Fe3+ + 6OH- ->
        # 2FeO(OH) + 2H2O
        try:
            reactants, _ = re.split(r'\s*->\s*', eq_normalized, 1)
        except ValueError:
            print(f"Invalid reaction equation: {eq_normalized}")
            continue

        # Split reactants by ' + '
        reactant_list = re.split(r'\s\+\s', reactants)

        # Filter out terms containing Fe or e- except for Fe ions and remove the coefficients in front of the reactants
        filtered_reactants = []
        for r in reactant_list:
            # Remove all the number in front of the reactant
            r_no_coeff = re.sub(r'^\d+(\.\d+)?(/\d+(\.\d+)?)?\s*', '', r)
            # Remove another form of coefficients with undetermined number of digits, for instance, (1-4z), (2+2x), 2x...
            r_no_coeff = re.sub(r'^\(\d+([+-]\d+)?[xyz]\)|^(\d+)?[xyz]|^(\d+)?[xyz]([+-]\d+)?', '', r_no_coeff)
            # Remove the water molecule after \u00b7, CaFe(PO4)\u00b7H2O -> CaFe(PO4)
            r_no_coeff = re.sub(r'\u00b7.*', '', r_no_coeff)
            # Check for Fe ions and exclude 'e-'
            if ('Fe' in r_no_coeff and re.search(r'Fe\d+\+|Fe(II)|Fe(III)', r_no_coeff)) or (
                    'Fe' not in r_no_coeff and 'e-' not in r_no_coeff):
                filtered_reactants.append(r_no_coeff)

        # Find the missing reactants or reagents
        missing_reactants_reagents = [reactant for reactant in filtered_reactants if
                                      all(reactant != reactant_reagent["name"] for reactant_reagent in
                                          reactants_reagents)]
        new_reactants_reagents = []
        # Add the missing reactants or reagents
        for missing_reactant_reagent in missing_reactants_reagents:
            smiles = chemical_name2smiles(missing_reactant_reagent)
            new_reactants_reagents.extend(smiles)
        # Deduplicate the reactants or reagents based on the smiles.
        old_smiles = [reactant_reagent["smiles"] for reactant_reagent in reactants_reagents]
        for new_reactant_reagent in new_reactants_reagents:
            if new_reactant_reagent["smiles"] not in old_smiles:
                reactants_reagents.append(new_reactant_reagent)

    print(f"Updated reactants or reagents: {reactants_reagents}")
    return reactants_reagents


def remove_redundant_reactants_reagents(reactants_reagents: List[dict]) -> List[dict]:
    """
    Remove the redundant reactants or reagents in the pathway. Since some of the reactants or reagents are in the form
    of ions, and their corresponding compounds are already included in the pathway, we need to remove the redundant
    reactants or reagents. For example, the smiles "O=S([O-])([O-])=S.[Na+].[Na+]" is composed of "O=S([O-])([O-])=S",
    "Na+" and "Na+", therefore, if "[Na+]" is already included in the pathway, we need to remove it from the list of
    reactants or reagents.
    """
    new_reactants_reagents = reactants_reagents.copy()
    index_list = []
    # Deduplicate from the last element to the first element in the reactants_reagents list
    for reactant_reagent in reactants_reagents[::-1]:
        # Find the index of the reactant or reagent in the list
        index = new_reactants_reagents.index(reactant_reagent)
        temp_reactants_reagents = new_reactants_reagents[:index]
        # Create a list for redundancy check and split the reactant or reagent by '.'
        redundancy_check_list = [component
                                 for r in temp_reactants_reagents
                                 if r != reactant_reagent and r["smiles"] is not None
                                 for component in r["smiles"].split('.')]
        print(f"Redundancy check: {redundancy_check_list}")
        if reactant_reagent["smiles"] in redundancy_check_list:
            new_reactants_reagents.remove(reactant_reagent)
            print(f"Removed redundant reactant or reagent: {reactant_reagent}")
            index_list.append(index)
    return new_reactants_reagents, index_list


def isomeric_smiles2canonical_smiles(isomeric_smiles: str) -> str:
    """
    Convert isomeric SMILES to canonical SMILES using RDKit.
    """
    if isomeric_smiles:
        mol = Chem.MolFromSmiles(isomeric_smiles)
    else:
        return isomeric_smiles
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {isomeric_smiles}")

    canonical_smiles = Chem.MolToSmiles(mol)

    return canonical_smiles


def main():
    parser = argparse.ArgumentParser(description="Normalize chemical names and SMILES in pathway data.")
    parser.add_argument('--input_path', type=str,
                        default=os.path.join(GRANDPARENT_PATH, 'data/curation/pathways_final.jsonl'),
                        help='Path to the input JSONL file containing pathway data.')
    parser.add_argument('--output_path', type=str,
                        default=os.path.join(GRANDPARENT_PATH, 'data/curation/pathways_final.jsonl'),
                        help='Path to save the updated JSONL file with normalized chemical names and SMILES.')
    args = parser.parse_args()
    # Read the chemical_name from the csv file
    data = pd.read_json(args.input_path, lines=True)
    df = pd.DataFrame()
    for index, row in tqdm(data.iterrows(), total=data.shape[0]):
        print(f"########## Processing pathway {index} ##########")
        # Doping species and adsorbed species
        for phase in row["pathway"]["precursor_phase"] + row["pathway"]["product_phase"]:
            if "doping_species" in phase:
                phase["doping_species"] = chemical_name2smiles(phase["doping_species"])
            if "adsorbed_species" in phase:
                phase["adsorbed_species"] = chemical_name2smiles(phase["adsorbed_species"])
        if "reaction_equation" in row["pathway"] and row["pathway"]["reaction_equation"] != "uncertain":
            for operation in row["pathway"]["procedure"]:
                try:
                    if "condition" in operation and "reactant_or_reagent" in operation["condition"]:
                        # Uncomment the following line if you want to add missing reactants or reagents
                        # operation["condition"]["reactant_or_reagent"] = add_missing_reactants_reagents(
                        #     row["pathway"]["reaction_equation"],
                        #     operation["condition"]["reactant_or_reagent"])
                        operation["condition"]["reactant_or_reagent"], index_list = remove_redundant_reactants_reagents(
                            operation["condition"]["reactant_or_reagent"])
                    else:
                        # Uncomment the following line if you want to add missing reactants or reagents
                        # operation["condition"]["reactant_or_reagent"] = add_missing_reactants_reagents(
                        #     row["pathway"]["reaction_equation"],
                        #     [])
                        index_list = []
                    # Align the concentration to the list of reactant or reagent
                    if "condition" in operation and "reactant_or_reagent_concentration" in operation["condition"]:
                        num_of_reactant_reagent = len(operation["condition"]["reactant_or_reagent"])
                        num_of_concentration = len(operation["condition"]["reactant_or_reagent_concentration"])
                        if num_of_reactant_reagent > num_of_concentration:
                            operation["condition"]["reactant_or_reagent_concentration"] = (
                                    operation["condition"]["reactant_or_reagent_concentration"]
                                    + [{"value": None, "unit": "M"}] * (num_of_reactant_reagent - num_of_concentration))
                        elif num_of_reactant_reagent < num_of_concentration:
                            # Remove the redundant concentration according to the index_list
                            operation["condition"]["reactant_or_reagent_concentration"] = [
                                operation["condition"]["reactant_or_reagent_concentration"][i] for i in
                                range(num_of_concentration) if i not in index_list]
                    else:
                        operation["condition"]["reactant_or_reagent_concentration"] = [{"value": None,
                                                                                        "unit": "M"}] * len(
                            operation["condition"]["reactant_or_reagent"])
                except KeyError:
                    print(f"Missing key in pathway {index}")
        # Add the updated row to the dataframe
        df = df._append(row, ignore_index=True)
    # Save the updated dataframe to a json file
    df.to_json(args.output_path, orient='records', lines=True, force_ascii=False)


if __name__ == '__main__':
    main()
