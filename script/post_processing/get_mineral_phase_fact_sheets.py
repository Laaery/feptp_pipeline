"""
Reorganize a data fact sheet for mineral phases in the sequence of mineral_cif_mapping.csv, and add corresponding mp_id
and other properties if possible. To obtain unique mp_id, first we need to seek for restriction of the mineral phase,
such as crystal system, space group, and chemical formula. Then we can use the obtained restriction to search for
the mp_id in the Materials Project database.
"""
import os
import pandas as pd
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.periodic_table import Element
from pymatgen.core.periodic_table import Specie
import numpy as np
from mp_api.client import MPRester
from mp_api.client.routes.materials.chemenv import ChemenvRester
from tqdm import tqdm
import argparse

CURRENT_PATH = os.path.dirname(__file__)
GRANDPARENT_PATH = os.path.abspath(os.path.join(CURRENT_PATH, "../.."))


def get_structure_detail(id, mineral_phase, database):
    """
    Get the structure details of a mineral phase.
    Args:
        id: (int) the id of the mineral phase cif
        mineral_phase: (str) the name of the mineral phase
        database: (dataframe) the database of mineral phases
    Returns:
        mineral_phase_details: (dict) the details of the mineral phase
    """
    mineral_phase_details = {}
    # Get the details of the mineral phase from the database
    mineral_phase_details['name'] = mineral_phase
    if mineral_phase in database['name'].values:
        mineral_phase_details['empirical_formula_db'] = \
            database.loc[database['name'] == mineral_phase, 'Chemical Formula'].values[0]
        mineral_phase_details['crystal_system_db'] = database.loc[database['name'] == mineral_phase, 'csystem'].values[
            0].capitalize()
        mineral_phase_details['space_group_num_db'] = \
            database.loc[database['name'] == mineral_phase, 'spacegroup'].values[0]
        mineral_phase_details['space_group_db'] = \
            database.loc[database['name'] == mineral_phase, 'spacegroupset'].values[0]
        mineral_phase_details['a_db'] = database.loc[database['name'] == mineral_phase, 'a'].values[0]
        mineral_phase_details['b_db'] = database.loc[database['name'] == mineral_phase, 'b'].values[0]
        mineral_phase_details['c_db'] = database.loc[database['name'] == mineral_phase, 'c'].values[0]
        mineral_phase_details['alpha_db'] = database.loc[database['name'] == mineral_phase, 'alpha'].values[0]
        mineral_phase_details['beta_db'] = database.loc[database['name'] == mineral_phase, 'beta'].values[0]
        mineral_phase_details['gamma_db'] = database.loc[database['name'] == mineral_phase, 'gamma'].values[0]
        mineral_phase_details['va3_db'] = database.loc[database['name'] == mineral_phase, 'va3'].values[0]
        # Get the Fe content, FeO content, and Fe2O3 content
        fe_content = database.loc[database['name'] == mineral_phase, 'Fe_content'].values[0]
        feo_content = database.loc[database['name'] == mineral_phase, 'FeO_content'].values[0]
        fe2o3_content = database.loc[database['name'] == mineral_phase, 'Fe2O3_content'].values[0]
        # All the values of contents are not 0 or np.nan
        if any([fe_content == 0, fe_content == np.nan, feo_content == 0, feo_content == np.nan, fe2o3_content == 0,
                fe2o3_content == np.nan]):
            print(f'The Fe content, FeO content, and Fe2O3 content of {mineral_phase} are missing.')
            fe2, fe3 = None, None, None
        else:
            fe2, fe3 = calculate_fe_species_percentage(fe_content, feo_content, fe2o3_content)
        mineral_phase_details['total_fe_wt'] = fe_content
        mineral_phase_details['fe2_at_ratio'] = fe2
        mineral_phase_details['fe3_at_ratio'] = fe3

    # Get another series of mineral details from CIF
    mineral_phase_details.update(get_structure_details_from_cif(id, mineral_phase))
    # Compare the details from database and CIF. Focus on crystal system.
    if "crystal_system_db" in mineral_phase_details and "crystal_system_cif" in mineral_phase_details:
        # Normalize the crystal system symbols. For crystal system from database, replace "Isometric" with "cubic".
        mineral_phase_details['crystal_system_db'] = mineral_phase_details['crystal_system_db'].replace("Isometric",
                                                                                                        "cubic")
        if mineral_phase_details['crystal_system_db'] != mineral_phase_details['crystal_system_cif']:
            print(f'{mineral_phase} has different crystal system in database and CIF.')
            print(f'cystal system in database: {mineral_phase_details["crystal_system_db"]}')
            print(f'crystal system in CIF: {mineral_phase_details["crystal_system_cif"]}')
    # Final merge with a unified criterion. The information from database is prioritized except for Parascorodite and
    # Ferrihydrite. For these two mineral phases, the information from CIF is prioritized. If the information is
    # missing in the database, use the information from CIF automatically.
    # Check if any information is missing (0 or np.nan) or no key ended with '_db' in the database
    missing_flag = False
    for key in mineral_phase_details.keys():
        if ((key.endswith('_db') and (mineral_phase_details[key] == 0 or mineral_phase_details[key] == np.nan))
                or not key.endswith('_db')):
            missing_flag = True
            break
    print(mineral_phase_details)
    if mineral_phase == "Parascorodite" or mineral_phase == "Ferrihydrite" or missing_flag:
        mineral_phase_details['crystal_system'] = mineral_phase_details['crystal_system_cif']
        mineral_phase_details['space_group_num'] = mineral_phase_details['space_group_num_cif']
        mineral_phase_details['space_group'] = mineral_phase_details['space_group_cif']
        mineral_phase_details['a'] = mineral_phase_details['a_cif']
        mineral_phase_details['b'] = mineral_phase_details['b_cif']
        mineral_phase_details['c'] = mineral_phase_details['c_cif']
        mineral_phase_details['alpha'] = mineral_phase_details['alpha_cif']
        mineral_phase_details['beta'] = mineral_phase_details['beta_cif']
        mineral_phase_details['gamma'] = mineral_phase_details['gamma_cif']
        mineral_phase_details['va3'] = mineral_phase_details['va3_cif']
    else:
        mineral_phase_details['crystal_system'] = mineral_phase_details['crystal_system_db']
        mineral_phase_details['space_group_num'] = mineral_phase_details['space_group_num_db']
        mineral_phase_details['space_group'] = mineral_phase_details['space_group_db']
        mineral_phase_details['a'] = mineral_phase_details['a_db']
        mineral_phase_details['b'] = mineral_phase_details['b_db']
        mineral_phase_details['c'] = mineral_phase_details['c_db']
        mineral_phase_details['alpha'] = mineral_phase_details['alpha_db']
        mineral_phase_details['beta'] = mineral_phase_details['beta_db']
        mineral_phase_details['gamma'] = mineral_phase_details['gamma_db']
        mineral_phase_details['va3'] = mineral_phase_details['va3_db']
    return mineral_phase_details


def get_structure_details_from_cif(id, mineral_phase):
    """
    Get the structure details of a mineral phase from the cif file.
    Args:
        id: (int) the id of the mineral phase cif
        mineral_phase: (str) the name of the mineral phase
    Returns:
        mineral_phase_details: (dict) the details of the mineral phase
    """
    mineral_phase_details = {}
    cif_path = os.path.join(GRANDPARENT_PATH, f'data/kb/cif_std/{id}.cif')
    # Read the cif file
    parser = CifParser(cif_path, occupancy_tolerance=1.1)
    structure = parser.get_structures()[0]
    # Get the details of the mineral phase
    mineral_phase_details['empirical_formula_cif'] = structure.composition.reduced_formula
    mineral_phase_details['crystal_system_cif'] = SpacegroupAnalyzer(structure).get_crystal_system().capitalize()
    mineral_phase_details['space_group_num_cif'] = SpacegroupAnalyzer(structure).get_space_group_number()
    mineral_phase_details['space_group_cif'] = SpacegroupAnalyzer(structure).get_space_group_symbol()
    mineral_phase_details['a_cif'] = structure.lattice.abc[0]
    mineral_phase_details['b_cif'] = structure.lattice.abc[1]
    mineral_phase_details['c_cif'] = structure.lattice.abc[2]
    mineral_phase_details['alpha_cif'] = structure.lattice.angles[0]
    mineral_phase_details['beta_cif'] = structure.lattice.angles[1]
    mineral_phase_details['gamma_cif'] = structure.lattice.angles[2]
    mineral_phase_details['va3_cif'] = structure.volume
    return mineral_phase_details


def calculate_fe_species_percentage(fe_content, feo_content, fe2o3_content):
    """
    Calculate the atomic percentage of Fe(II) and Fe(III) species in the mineral phase given the Fe content,
    FeO content, and Fe2O3 content, all in wt%.
    Args:
        fe_content: (float) the Fe content in wt%
        feo_content: (float) the FeO content in wt%
        fe2o3_content: (float) the Fe2O3 content in wt%
    Returns:
        fe2: (float) the atomic percentage of Fe(II) in all Fe species (at%)
        fe3: (float) the atomic percentage of Fe(III) in all Fe species (at%)
    """
    molar_mass_fe = 55.845
    molar_mass_feo = 71.844
    molar_mass_fe2o3 = 159.688
    fe2 = (feo_content / molar_mass_feo) / (fe_content / molar_mass_fe)
    fe3 = (fe2o3_content / molar_mass_fe2o3) * 2 / (fe_content / molar_mass_fe)
    return fe2, fe3


def main():
    parser = argparse.ArgumentParser(
        description="Generate mineral phase fact sheets with mp_id and other properties.")
    parser.add_argument('--input_path', type=str,
                           default=os.path.join(GRANDPARENT_PATH, 'data/kb/cif_std/mineral_cif_mapping.csv'),
                           help='Path to the input mineral phase mapping CSV file.')
    parser.add_argument('--composition_path', type=str,
                           default=os.path.join(GRANDPARENT_PATH, 'data/kb/id_composition.csv'),
                           help='Path to the input composition CSV file.')
    parser.add_argument('--database_path', type=str,
                           default=os.path.join(GRANDPARENT_PATH, 'data/kb/IMA_Fe_minerals_all.csv'),
                           help='Path to the input mineral database CSV file.')
    parser.add_argument('--output_path', type=str,
                           default=os.path.join(GRANDPARENT_PATH, 'data/kb/mineral_phase_fact_sheets.csv'),
                           help='Path to the output mineral phase fact sheets CSV file.')
    args = parser.parse_args()
    # Read the database of mineral phases
    data = pd.read_csv(args.input_path)
    composition_df = pd.read_csv(args.composition_path)

    database = pd.read_csv(args.database_path)
    details_list = []
    # Get the mp_id of each mineral phase tqdm(range(len(data)))
    for i in tqdm(range(len(data)), desc='Processing mineral phases'):
        mineral_phase = data.loc[i, 'Name']
        # Get the details of the mineral phase
        mineral_phase_details = get_structure_detail(i + 1, mineral_phase, database)
        # Get the mp_id of the mineral phase based on the composition and crystal system from materials project
        # The composition of the mineral phase
        composition = eval(composition_df.loc[composition_df['id'] == i + 1, 'composition'].values[0])
        comp = ''.join([key + str(value) for key, value in composition.items()])
        with MPRester(api_key=os.environ['MP_API_KEY_NEW']) as mpr:
            # Search for the corresponding mp_id in the Materials Project database. Try to find the structure with the
            # lowest energy per atom and not theoretical.
            filters = {
                'formula': comp,
                'crystal_system': mineral_phase_details['crystal_system'],
                'theoretical': False
            }
            docs = mpr.materials.summary.search(**filters)
            if docs:
                docs = sorted(docs, key=lambda x: x.energy_per_atom if x.energy_per_atom is not None else float('inf'))
            else:
                filters.pop('theoretical')
                docs = mpr.materials.summary.search(**filters)
                if docs:
                    docs = sorted(docs,
                                  key=lambda x: x.energy_per_atom if x.energy_per_atom is not None else float('inf'))
            if len(docs) == 0:
                print(f'The mp_id of {mineral_phase} is not found.')
                mineral_phase_details['mp_id'] = None
            else:
                mineral_phase_details['mp_id'] = docs[0].material_id
                mineral_phase_details['formation_energy'] = docs[0].formation_energy_per_atom
                mineral_phase_details['theoretical'] = docs[0].theoretical
                mineral_phase_details['is_stable'] = docs[0].is_stable
            # If fe_percentage not in database, try to use the mp_id to get the structure and calculate the Fe percentage
            if 'total_fe_wt' not in mineral_phase_details and mineral_phase_details['mp_id'] is not None:
                chemenv_rester = ChemenvRester(api_key=os.environ["MP_API_KEY_NEW"])
                total_fe_count, fe2_count, fe3_count = 0, 0, 0
                try:
                    structure = chemenv_rester.search(material_ids=docs[0].material_id)[0].structure
                    for site in structure:
                        if site.specie.symbol == "Fe":
                            total_fe_count += 1
                            if site.specie.oxi_state == 2:
                                fe2_count += 1
                            elif site.specie.oxi_state == 3:
                                fe3_count += 1
                except Exception as e:
                    print(f"{mineral_phase} has no similar computed cystal structure with oxidation state assigned.")
                    fe3_count = None
                    fe2_count = None
                # Calculate atomic ratios for Fe2+ and Fe3+
                if total_fe_count > 0 and fe2_count is not None and fe3_count is not None:
                    fe2_at_ratio = fe2_count / total_fe_count
                    fe3_at_ratio = fe3_count / total_fe_count
                # Calculate the total Fe weight percent in the structure
                molar_mass_fe = 55.845
                total_mass = sum(
                    [Element(specie.symbol).atomic_mass * count for specie, count in structure.composition.items()])
                fe_mass = sum([molar_mass_fe for site in structure if site.specie.symbol == "Fe"])
                fe_wt_percent = (fe_mass / total_mass) * 100 if total_mass else 0
                mineral_phase_details['total_fe_wt'] = fe_wt_percent
                mineral_phase_details['fe2_at_ratio'] = fe2_at_ratio
                mineral_phase_details['fe3_at_ratio'] = fe3_at_ratio

        details_list.append(mineral_phase_details)
    details_df = pd.DataFrame(details_list)
    # Add a column of id for indexing as the first column
    details_df.insert(0, 'id', range(1, len(details_df) + 1))
    details_df.to_csv(args.output_path, index=False)


if __name__ == '__main__':
    main()
