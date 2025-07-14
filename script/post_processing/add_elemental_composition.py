"""
Add the elemental composition of the mineral phases in the curated data.
"""
import os
import re
import pandas as pd
from chempy import Substance
from chempy.util import periodic
from chempy import balance_stoichiometry
from numpy import nan
import argparse

CURRENT_PATH = os.path.dirname(__file__)
GRANDPARENT_PATH = os.path.abspath(os.path.join(CURRENT_PATH, "../.."))


# Function to get molecular weight using chempy
def get_molecular_weight(element):
    substance = Substance.from_formula(element)
    return substance.mass


# Function to convert weight percentage to molar ratios
def weight_to_molar_ratios(weight_percentages, elements):
    # Calculate the number of moles of each element
    moles = [weight_percentages[i] / get_molecular_weight(elements[i]) for i in range(len(elements))]
    # Calculate the total moles
    total_moles = sum(moles)
    # Calculate molar ratio for each element
    molar_ratios = {elements[i]: moles[i] / total_moles for i in range(len(elements))}
    return molar_ratios


def calculate_elemental_composition(phase):
    try:
        # If formula is an iron ions, for example, Fe2+ Fe2+ ,Fe3+, Fe(II), Fe(III), FeII, FeIII
        ions_charge_2 = ["Fe2+", "Fe(2+)", "Fe(II)", "FeII", "Fe(II)aq", "Fe(II)(aq)", "Fe^2+", "Fe^(2+)", "Fe^{2+}"]
        ions_charge_3 = ["Fe3+", "Fe(3+)", "Fe(III)", 'FeIII', "Fe(III)aq", "Fe(III)(aq)", "Fe^3+", "Fe^(3+)",
                         "Fe^{3+}"]
        if phase['formula'] in ions_charge_2:
            formula = 'Fe2+'
            return {"Fe": 1, "net charge": 2}
        elif phase['formula'] in ions_charge_3:
            formula = 'Fe3+'
            return {"Fe": 1, "net charge": 3}
        elif "alloy" in phase['name'].lower():
            # Pattern to match the alloy formula
            alloy_pattern = re.compile(
                r'Fe-(\d+)([A-Z][a-z]*)-(\d+)([A-Z][a-z]*)-?(\d*.\d*)?([A-Z][a-z]*)?-?(\d*.\d*)?([A-Z][a-z]*)?')
            # Search for the pattern in the phase name
            match = alloy_pattern.search(phase['name'])
            if match:
                elements = ['Fe']
                weight_percentages = [100]
                for i in range(1, len(match.groups()), 2):
                    if match.group(i) and match.group(i + 1):
                        weight_percentages.append(float(match.group(i)))
                        elements.append(match.group(i + 1))
                # Adjust Fe weight percentage
                weight_percentages[0] -= sum(weight_percentages[1:])
                # Convert weight percentages to molar ratios
                elemental_composition = weight_to_molar_ratios(weight_percentages, elements)
                return elemental_composition
            else:
                return {}
        else:
            # Replace Fe2+ ,Fe3+, Fe(II), Fe(III), FeII, FeIII with Fe
            formula = re.sub(r'Fe2\+|Fe3\+|Fe\(II\)|Fe\(III\)|FeIII|FeII', 'Fe', phase['formula'])
            # Normalize the · symbol
            formula = formula.replace('\u22c5', '\u00b7').replace('\u2219', '\u00b7').replace('\u2022', '\u00b7')
            # Remove unicode characters except for /u00b7
            formula = re.sub(r'[^\x00-\x7F\u00b7]', '', formula)
            # Replace • with unicode character /u00b7
            formula = re.sub(r'•', r'·', formula)
            # Remove valance using regex. Valance is behind the element symbol and in front of the +/- symbol. Sometimes
            # when the value of valance is 1, it is not written in the formula, so we need to remove the +/- symbol
            formula = re.sub(r'([A-Z][a-z]*)(\d+)?[+-]', r'\1', formula)
            # Remove deficiency of elements. Normally denotes as a space followed by a decimal number/integer or nothing
            formula = re.sub(r'\s+\d*\.?\d*', '', formula)
            # Handle when stoichiometric number for H2O is written as a decimal number, such as Fe2O3·0.5H2O
            # Calculate the stoichiometry for H and O separately
            prefix_num_h2o = re.search(r'(\d+\.\d+)H2O', formula)
            if prefix_num_h2o:
                stoichiometry = float(prefix_num_h2o.group(1))
                formula = re.sub(r'\d+\.\d+H2O', f'H{stoichiometry * 2}O{stoichiometry}', formula)
            formula = re.sub(r'[+-]', '', formula)
            print(f'Formula before cleaning: {phase["formula"]}', f'Formula after cleaning: {formula}')
            composition = Substance.from_formula(formula).composition
            elemental_composition = {periodic.symbols[atomic_number - 1]: amount for atomic_number, amount in composition.items()}
            return elemental_composition
    except Exception as e:
        mineral_data = pd.read_csv(os.path.join(GRANDPARENT_PATH, 'data/kb/IMA_Fe_minerals_all.csv'))
        # Dealing with some special cases
        # Find elemental composition by name when name is in the mineral_data and empirical formula is not None
        if phase['name'] in mineral_data['Mineral Name (plain)'].values and \
                mineral_data[mineral_data['Mineral Name (plain)'] == phase['name']]['Empirical Formula'].values[
                    0] is not nan:
            mineral = mineral_data[mineral_data['Mineral Name (plain)'] == phase['name']]
            formula = mineral['Empirical Formula'].values[0]
            formula = re.sub(r'([A-Z][a-z]*)(\d+)?[+-]', r'\1', formula)
            try:
                composition = Substance.from_formula(formula).composition
                elemental_composition = {periodic.symbols[atomic_number - 1]: amount for atomic_number, amount in composition.items()}
                return elemental_composition
            except:
                return {}
        else:
            print("Bad formula: ", phase['formula'])
            cleaned_formula = re.sub(r'Fe2\+|Fe3\+|Fe\(II\)|Fe\(III\)|FeIII|FeII', 'Fe', phase['formula'])
            cleaned_formula = re.sub(r'([A-Z][a-z]*)(\d+)?[+-]', r'\1', cleaned_formula)
            cleaned_formula = re.sub(r'⋅.*H2O', '', cleaned_formula)
            # If formula is not a typical mineral, which contains x, n , (...,...), .../u00b72.6H2O, etc.
            # Using regex to clean the formula. Note that the stoichiometric numbers are removed.
            # Remove stoichiometric numbers and valences within parentheses, then remove the parentheses
            cleaned_formula = re.sub(r'\((\w+),\s*(\w+)\)[\d.x-]*', r'\1\2', cleaned_formula)
            # Remove any remaining stoichiometric numbers
            cleaned_formula = re.sub(r'\d[-+]x', '', cleaned_formula)
            cleaned_formula = re.sub(r'\d[-+]y', '', cleaned_formula)
            # Remove deficiency of elements. Normally denotes as a space followed by a number or nothing
            cleaned_formula = re.sub(r'\s\d*', '', cleaned_formula)
            # Remove all unicode characters
            cleaned_formula = re.sub(r'[^\x00-\x7F]', '', cleaned_formula)
            cleaned_formula = re.sub(r'[+-]', '', cleaned_formula)
            print(f'Formula before cleaning: {phase["formula"]}', f'Formula after cleaning: {cleaned_formula}')
            try:
                # Elemental composition of the cleaned formula with uncertain stoichiometry
                composition = Substance.from_formula(cleaned_formula).composition
                return {periodic.symbols[atomic_number - 1]: "uncertain" for atomic_number, amount in
                        composition.items()}
            except:
                return {}


def get_id_composition(pathway_data):
    """
    args:
        data: dict, the data of pathways

    Return pair of id and composition from the pathway with no duplicates
    """
    id_composition = []
    for _, row in pathway_data.iterrows():
        for phase in row['pathway']['precursor_phase'] + row['pathway']['product_phase']:
            if 'id' in phase:
                id_composition.append({'id': phase['id'], 'composition': phase['elemental_composition']})
    # Remove duplicate ids
    unique_dict = {}
    seen_ids = set()
    for d in id_composition:
        if d["id"] in seen_ids:
            continue
        seen_ids.add(d["id"])
        unique_dict[d["id"]] = d
    # Sort the dictionary by id
    id_composition = [unique_dict[id] for id in sorted(unique_dict)]

    return id_composition


def main():
    parser = argparse.ArgumentParser(description="Add elemental composition to the pathways data")
    parser.add_argument('--input_path', type=str,
                        default=os.path.join(GRANDPARENT_PATH, 'data/curation/pathways_final.jsonl'),
                        help='Path to the input pathways data file')
    parser.add_argument('--output_path', type=str,
                        default=os.path.join(GRANDPARENT_PATH, 'data/curation/pathways_final.jsonl'),
                        help='Path to the output pathways data file')
    parser.add_argument('--id_composition_path', type=str,
                        default=os.path.join(GRANDPARENT_PATH, 'data/kb/id_composition.csv'),
                        help='Path to the output id composition file')
    args = parser.parse_args()
    # Load the curated data
    data = pd.read_json(args.input_path, lines=True)

    # Remove empty pathways
    data = data.dropna(subset=['pathway'])
    # Remove the pathways without precursor_phase or product_phase keys
    data = data[data['pathway'].apply(lambda x: 'precursor_phase' in x and 'product_phase' in x)]
    # Remove the pathways with empty precursor phases or product phases
    data = data[data['pathway'].apply(lambda x: x['precursor_phase'] != [] and x['product_phase'] != [])]
    # Remove the pathways whose formula is empty string
    data = data[data['pathway'].apply(
        lambda x: all(['formula' in phase and phase['formula'] != '' for phase in x['precursor_phase']]))]
    data = data[data['pathway'].apply(
        lambda x: all(['formula' in phase and phase['formula'] != '' for phase in x['product_phase']]))]
    # Drop incomplete pathways, which do not have the "changed" and "reason" keys
    data = data[data['pathway'].apply(lambda x: 'changed' in x and 'reason' in x)]
    # Drop the pathways with empty procedure list
    data = data[data['pathway'].apply(lambda x: 'procedure' in x and x['procedure'] != [])]
    print(f'The number of pathways is {data.shape[0]}')
    # Reset the index
    data = data.reset_index(drop=True)
    print(data.head())
    for index, row in data.iterrows():
        precursor_phases = row['pathway']['precursor_phase']
        product_phases = row['pathway']['product_phase']
        print(f'Progress: {index + 1}/{data.shape[0]}')
        # Add the elemental composition of each precursor and product phases
        for precursor_phase in precursor_phases:
            print(precursor_phase['formula'])
            if precursor_phase['formula'] != 'uncertain':
                precursor_phase['elemental_composition'] = calculate_elemental_composition(precursor_phase)
                # If the elemental composition has key 'net charge', rename the precursor phase name to Fe(II) ion or Fe(III) ion
                if 'net charge' in precursor_phase['elemental_composition']:
                    if precursor_phase['elemental_composition']['net charge'] == 2:
                        precursor_phase['formula'] = 'Fe2+'
                        precursor_phase['name'] = 'Fe(II) ion'
                    elif precursor_phase['elemental_composition']['net charge'] == 3:
                        precursor_phase['formula'] = 'Fe3+'
                        precursor_phase['name'] = 'Fe(III) ion'
            else:
                precursor_phase['elemental_composition'] = {}
            print(precursor_phase['elemental_composition'])
        for product_phase in product_phases:
            print(product_phase['formula'])
            if product_phase['formula'] != 'uncertain':
                product_phase['elemental_composition'] = calculate_elemental_composition(product_phase)
                # If the elemental composition has key 'net charge', rename the product phase name to Fe(II) ion or Fe(III) ion
                if 'net charge' in product_phase['elemental_composition']:
                    if product_phase['elemental_composition']['net charge'] == 2:
                        product_phase['formula'] = 'Fe2+'
                        product_phase['name'] = 'Fe(II) ion'
                    elif product_phase['elemental_composition']['net charge'] == 3:
                        product_phase['formula'] = 'Fe3+'
                        product_phase['name'] = 'Fe(III) ion'
            else:
                product_phase['elemental_composition'] = {}
            print(product_phase['elemental_composition'])
    # # Eliminate the pathways without 'Fe' in the elemental composition
    # if 'elemental_composition' in data['pathway'][0]['precursor_phase'][0]:
    #     data = data[data['pathway'].apply(lambda x: any([any([list(element.keys())[0] == 'Fe' for element in phase['elemental_composition']]) for phase in x['precursor_phase']]))]
    #     data = data[data['pathway'].apply(lambda x: any([any([list(element.keys())[0] == 'Fe' for element in phase['elemental_composition']]) for phase in x['product_phase']]))]
    # Remove precursor and product phases without 'Fe' in the elemental composition
    for index, row in data.iterrows():
        precursor_phases = row['pathway']['precursor_phase']
        product_phases = row['pathway']['product_phase']
        precursor_phases = [precursor_phase for precursor_phase in precursor_phases if
                            any(['Fe' in element for element in
                                 precursor_phase['elemental_composition']] or
                                ['Fe' in precursor_phase['formula']])]
        product_phases = [product_phase for product_phase in product_phases if
                          any(['Fe' in element for element in product_phase['elemental_composition']]
                              or ['Fe' in product_phase['formula']])]
        data.at[index, 'pathway']['precursor_phase'] = precursor_phases
        data.at[index, 'pathway']['product_phase'] = product_phases
    # Save the data, if the file exists, append the data to the file
    print(f'The number of pathways is {data.shape[0]}')
    data.to_json(args.output_path, orient='records', lines=True)
    # Save id and composition to a csv file
    id_composition = get_id_composition(data)
    id_composition = pd.DataFrame(id_composition)
    id_composition.to_csv(args.id_composition_path, index=False)


if __name__ == '__main__':
    main()
