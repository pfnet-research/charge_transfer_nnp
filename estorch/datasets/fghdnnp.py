import json
import os
from glob import glob
from os.path import abspath, basename
from typing import List

import numpy as np
from ase import Atoms

from estorch.data import AtomicData, AtomicInMemoryDataset
from estorch.datasets._keys import CHARGES_KEY, SYMBOLS_KEY, TOTAL_CHARGE_KEY
from nequip.data import AtomicDataDict


class FGHDNNPDataset(AtomicInMemoryDataset):
    """
    Dataset in Fourth-Generation High-Dimensional Neural Network Potential (4G-HDNNP)
    """

    @property
    def raw_file_names(self):
        return [basename(self.file_name)]

    @property
    def raw_dir(self):
        return abspath(self.file_name)

    def _get_atomicdata(self, atoms_dict) -> AtomicData:
        # ref: nequip.data.AtomicData.from_ase
        add_fields = {}

        add_fields[AtomicDataDict.FORCE_KEY] = np.array(atoms_dict[AtomicDataDict.FORCE_KEY])
        add_fields[CHARGES_KEY] = np.expand_dims(np.array(atoms_dict[CHARGES_KEY]), axis=1)
        add_fields[AtomicDataDict.TOTAL_ENERGY_KEY] = float(
            atoms_dict[AtomicDataDict.TOTAL_ENERGY_KEY]
        )
        add_fields[TOTAL_CHARGE_KEY] = float(atoms_dict[TOTAL_CHARGE_KEY])

        # atomic numbers
        atoms = Atoms(symbols=atoms_dict[SYMBOLS_KEY])
        add_fields[AtomicDataDict.ATOMIC_NUMBERS_KEY] = atoms.get_atomic_numbers().tolist()

        # cartesian positions
        positions = atoms_dict[AtomicDataDict.POSITIONS_KEY]

        # cell (optional)
        cell = atoms_dict.get(AtomicDataDict.CELL_KEY, None)
        pbc = cell is not None

        return AtomicData.from_points(
            pos=positions,
            cell=cell,
            pbc=pbc,
            **self.extra_fixed_fields,  # have to contain "r_max"
            **add_fields,
        )

    def get_data(self) -> List[AtomicData]:
        atoms_list = []
        for path in glob(os.path.join(self.raw_dir, "*.json")):
            with open(path, "r") as f:
                atoms_dict = json.load(f)
            atoms_list.append(atoms_dict)

        if self.include_frames is None:
            return ([self._get_atomicdata(atoms_dict) for atoms_dict in atoms_list],)
        else:
            return ([self._get_atomicdata(atoms_list[i]) for i in self.include_frames],)
