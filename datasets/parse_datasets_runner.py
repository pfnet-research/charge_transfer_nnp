import json
import os
import re
from typing import List

import click
import numpy as np
from ase.units import Bohr, Hartree
from tqdm import tqdm

from estorch.datasets import CHARGES_KEY, SYMBOLS_KEY, TOTAL_CHARGE_KEY
from nequip.data import AtomicDataDict

HA_TO_EV = Hartree  # 1 Ha = 27.211382543519 eV
BOHR_TO_AA = Bohr  # 1 Bohr = 0.529177249 AA


def split_by_spaces(line: str) -> List[str]:
    return re.split("\s+", line.strip("\n"))  # NOQA


def parse_dataset(path: str, output_dir: str, pbc: bool, atomic: bool = False):
    """
    Parse dataset of non-periodic system in RuNNer format

    Parsed Format
    -------------
    begin
    atom [X] [Y] [Z] [element:str] [charge] 0.0 [Fx] [Fy] [Fz]
    <repeat by num_atoms>
    energy [energy]
    charge [charge]
    end
    <repeat by num_datum>

    Returned JSON
    -------------
    positions, symbols, cell, charges: compatible with ASE's atoms object
    total_energy, forces, total_charge: target quantities
    These names should be consistent with nequip/data/_keys.py

    Units: eV for energy, angstrom for distance, elemental charge for charges, eV/AA for forces.
    """
    with open(path, "r") as f:
        lines = f.readlines()
    num_lines = len(lines)

    all_atoms = []
    pos = 0
    while pos < num_lines:
        # consume begin
        assert lines[pos].strip("\n") == "begin"
        pos += 1

        cell = []
        if pbc:
            for _ in range(3):
                parsed = split_by_spaces(lines[pos])
                assert parsed[0] == "lattice"
                lx, ly, lz = map(float, parsed[1:4])
                if atomic:
                    cell.append(list(np.array([lx, ly, lz])))
                else:
                    cell.append(list(np.array([lx, ly, lz]) * BOHR_TO_AA))
                pos += 1

        positions = []
        symbols = []
        charges = []
        forces = []

        while True:
            parsed = split_by_spaces(lines[pos])
            if len(parsed) != 10:
                break
            assert parsed[0] == "atom"
            x, y, z = map(float, parsed[1:4])
            element = parsed[4]
            charge = float(parsed[5])
            fx, fy, fz = map(float, parsed[7:10])

            if atomic:
                positions.append(list(np.array([x, y, z])))
            else:
                positions.append(list(np.array([x, y, z]) * BOHR_TO_AA))
            symbols.append(element)
            charges.append(charge)
            if atomic:
                forces.append(list(np.array([fx, fy, fz])))
            else:
                forces.append(list(np.array([fx, fy, fz]) * HA_TO_EV / BOHR_TO_AA))

            pos += 1

        # total energy
        parsed = split_by_spaces(lines[pos])
        assert parsed[0] == "energy"
        if atomic:
            energy = float(parsed[1])
        else:
            energy = float(parsed[1]) * HA_TO_EV
        pos += 1

        # total charge
        parsed = split_by_spaces(lines[pos])
        assert parsed[0] == "charge"
        total_charge = float(parsed[1])
        pos += 1

        # consume end
        assert lines[pos].strip("\n") == "end"
        pos += 1

        data = {
            AtomicDataDict.POSITIONS_KEY: positions,
            SYMBOLS_KEY: symbols,
            CHARGES_KEY: charges,
            AtomicDataDict.TOTAL_ENERGY_KEY: energy,
            AtomicDataDict.FORCE_KEY: forces,
            TOTAL_CHARGE_KEY: total_charge,
        }
        if pbc:
            data["cell"] = cell

        all_atoms.append(data)

    os.makedirs(output_dir, exist_ok=True)
    for i, atoms_i in tqdm(enumerate(all_atoms)):
        filename = os.path.join(output_dir, f"{i}.json")
        with open(filename, "w") as f:
            json.dump(atoms_i, f, indent=4)


@click.command()
@click.option("--atomic", default=False, is_flag=True, type=bool, help="Use atomic unit")
@click.option("--output-suffix", default="", type=str)
def main(atomic, output_suffix):
    datasets_root = "datasets_runner"

    nonperiodic_systems = ["Ag_cluster", "Carbon_chain", "NaCl"]
    periodic_systems = ["AuMgO"]

    for system in nonperiodic_systems:
        print(f"Convert {system} dataset")
        output_dir = system + output_suffix
        parse_dataset(
            os.path.join(datasets_root, system, "input.data"), output_dir, pbc=False, atomic=atomic
        )

    for system in periodic_systems:
        print(f"Convert {system} dataset")
        output_dir = system + output_suffix
        parse_dataset(
            os.path.join(datasets_root, system, "input.data"), output_dir, pbc=True, atomic=atomic
        )


if __name__ == "__main__":
    main()
