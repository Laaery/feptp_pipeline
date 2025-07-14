"""
Convert the units of the transformation pathways to standard units.
"""
import os
import pandas as pd
from typing import List, Any, Optional
from tqdm import tqdm
import chempy
import argparse


CURRENT_PATH = os.path.dirname(__file__)
GRANDPARENT_PATH = os.path.abspath(os.path.join(CURRENT_PATH, "../.."))


def temperature_convertor(temp: float, unit: str) -> float:
    """
    Convert the temperature units of the transformation pathways to standard units.
    Args:
        temp(float): The temperature value.
        unit(str): The unit of temperature.
    Returns:
        temp(float): The temperature value in standard units.
    """
    if unit in ('\u00b0C', 'C', 'celsius'):
        return temp
    elif unit in ('K', 'kelvin'):
        return temp - 273.15
    elif unit in ('\u00b0F', 'F', 'fahrenheit'):
        return (temp - 32) * 5 / 9
    else:
        print(f"Unit {unit} is not supported.")
        return None


def time_convertor(time: float, unit: str) -> float:
    """
    Convert the time units of the transformation pathways to standard units.
    Args:
        time(float): The time value.
        unit(str): The unit of time.
    Returns:
        time(float): The time value in standard units.
    """
    if unit in ('h', 'hour', 'hours'):
        return time
    elif unit in ('min', 'minute', 'minutes'):
        return time / 60
    elif unit in ('s', 'sec', 'second', 'seconds'):
        return time / 3600
    elif unit in ('day', 'days', 'd'):
        return time * 24
    elif unit in ('week', 'weeks'):
        return time * 24 * 7
    elif unit in ('month', 'months'):
        return time * 24 * 30
    elif unit in ('year', 'years'):
        return time * 24 * 365
    elif unit in ('Myrs', 'Myr', 'Ma', 'million years'):
        return time * 24 * 365 * 1e6
    elif unit in ('Gyrs', 'Gyr', 'Ga', 'billion years'):
        return time * 24 * 365 * 1e9
    else:
        print(f"Unit {unit} is not supported.")
        return None


def pressure_convertor(pressure: float, unit: str) -> float:
    """
    Convert the pressure units of the transformation pathways to standard units.
    Args:
        pressure(float): The pressure value.
        unit(str): The unit of pressure.
    Returns:
        pressure(float): The pressure value in standard units.
    """
    if unit == 'MPa':
        return pressure
    elif unit in ('Pa', 'pascal'):
        return pressure / 1e6
    elif unit in ('kPa', 'kilopascal'):
        return pressure / 1e3
    elif unit == 'bar':
        return pressure / 10
    elif unit == 'atm':
        return pressure / 101.325
    elif unit == 'psi':
        return pressure / 145.038
    elif unit == 'mbar':
        return pressure / 1e4
    elif unit == 'GPa':
        return pressure * 1e3
    elif unit == 'kbar':
        return pressure * 100
    elif unit in ('torr', 'mmHg', 'Torr'):
        return pressure / 7500.61685
    else:
        print(f"Unit {unit} is not supported.")
        return None


def redox_potential_convertor(redox_potential: float, unit: str) -> float:
    """
    Convert the redox potential units of the transformation pathways to standard units.
    Args:
        redox_potential(float): The redox potential value.
        unit(str): The unit of redox potential.
    Returns:
        redox_potential(float): The redox potential value in standard units.
    """
    if unit in ('V', 'volt'):
        return redox_potential
    elif unit in ('mV', 'millivolt'):
        return redox_potential / 1e3
    else:
        print(f"Unit {unit} is not supported.")
        return None


def concentration_convertor(concentration: float, unit: str, reactant_or_reagent: Optional[str] = None) -> float:
    """
    Convert the concentration units of the transformation pathways to standard units.
    Args:
        concentration(float): The concentration value.
        unit(str): The unit of concentration.
        reactant_or_reagent(str): Optional. The reactant or reagent name.
    Returns:
        concentration(float): The concentration value in standard units.
    """
    if unit in ('M', 'mol/L', 'mol L-1'):
        return concentration
    elif unit in ('mM', 'millimolar'):
        return concentration / 1e3
    elif unit == 'ppm':
        # Get the molecular weight of the reactant or reagent
        mw = chempy.Substance.from_formula(reactant_or_reagent).mass
        # Convert the concentration to molarity
        molarity = concentration / mw * 1000
        return molarity
    else:
        print(f"Unit {unit} is not supported.")
        return None


def main():
    parser = argparse.ArgumentParser(description="Convert units in transformation pathways.")
    parser.add_argument('--input_path', type=str,
                        default=os.path.join(GRANDPARENT_PATH, 'data/curation/pathways_original.jsonl'),
                        help='Path to the input JSONL file with pathways data.')
    parser.add_argument('--output_path', type=str,
                        default=os.path.join(GRANDPARENT_PATH, 'data/curation/pathways_standard_units.jsonl'),
                        help='Path to save the output JSONL file with converted units.')

    args = parser.parse_args()
    # Load data
    data = pd.read_json(args.input_path, lines=True, orient='records')
    # Convert the units
    for i, record in tqdm(data.iterrows()):
        pathway = record['pathway']
        for operation in pathway['procedure']:
            try:
                condition = operation['condition']
            except KeyError:
                print(f"KeyError: no condition in operation {operation} in pathway {pathway}")
                continue
            if 'temperature' in condition:
                if 'value' in condition['temperature'] and condition['temperature']['value'] is not None:
                    avg_value = temperature_convertor(condition['temperature']['value'],
                                                      condition['temperature']['unit'])
                    condition['temperature']['value'] = avg_value
                    condition['temperature']['unit'] = '\u00b0C'
                if 'min_value' in condition['temperature'] and condition['temperature']['min_value'] is not None:
                    min_value = temperature_convertor(condition['temperature']['min_value'],
                                                      condition['temperature']['unit'])
                    condition['temperature']['min_value'] = min_value
                    condition['temperature']['unit'] = '\u00b0C'
                if 'max_value' in condition['temperature'] and condition['temperature']['max_value'] is not None:
                    max_value = temperature_convertor(condition['temperature']['max_value'],
                                                      condition['temperature']['unit'])
                    condition['temperature']['max_value'] = max_value
                    condition['temperature']['unit'] = '\u00b0C'
            if 'time' in condition:
                if 'value' in condition['time'] and condition['time']['value'] is not None:
                    avg_value = time_convertor(condition['time']['value'], condition['time']['unit'])
                    # Set 0 h to None
                    if avg_value == 0:
                        avg_value = None
                    condition['time']['value'] = avg_value
                    condition['time']['unit'] = 'h'
                if 'min_value' in condition['time'] and condition['time']['min_value'] is not None:
                    min_value = time_convertor(condition['time']['min_value'], condition['time']['unit'])
                    condition['time']['min_value'] = min_value
                    condition['time']['unit'] = 'h'
                if 'max_value' in condition['time'] and condition['time']['max_value'] is not None:
                    max_value = time_convertor(condition['time']['max_value'], condition['time']['unit'])
                    condition['time']['max_value'] = max_value
                    condition['time']['unit'] = 'h'
            if 'pressure' in condition:
                if 'value' in condition['pressure'] and condition['pressure']['value'] is not None:
                    value = pressure_convertor(condition['pressure']['value'], condition['pressure']['unit'])
                    condition['pressure']['value'] = value
                    condition['pressure']['unit'] = 'MPa'
            if 'Eh' in condition:
                if 'value' in condition['Eh'] and condition['Eh']['value'] is not None:
                    value = redox_potential_convertor(condition['Eh']['value'], condition['Eh']['unit'])
                    condition['Eh']['value'] = value
                    condition['Eh']['unit'] = 'V'
            if ('reactant_or_reagent_concentration' in condition and 'reactant_or_reagent' in condition and
                    condition['reactant_or_reagent'] is not None and condition[
                        'reactant_or_reagent_concentration'] is not None):
                # Map the concentration to the reactant or reagent
                reactant_or_reagent = condition['reactant_or_reagent']
                concentration = condition['reactant_or_reagent_concentration']
                for i in range(len(concentration)):
                    if 'value' in concentration[i] and concentration[i]['unit'] != 'ppm':
                        value = concentration_convertor(concentration[i]['value'], concentration[i]['unit'])
                        if value == 0:
                            value = None
                        concentration[i]['value'] = value
                        concentration[i]['unit'] = 'M'

                    # if 'value' in concentration[i] and concentration[i]['unit'] == 'ppm':
                    #     value = concentration_convertor(concentration[i]['value'], concentration[i]['unit'], reactant_or_reagent[i])
                    #     concentration[i]['value'] = value
                    #     concentration[i]['unit'] = 'M'

            # except KeyError:
            #     print(f"KeyError: {operation} in pathway {pathway}")

    # Save the converted data
    data.to_json(args.output_path, orient='records', lines=True, force_ascii=False)


if __name__ == '__main__':
    main()
