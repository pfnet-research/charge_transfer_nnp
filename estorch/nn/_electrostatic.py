import math
from typing import List

import numpy as np
import torch
from ase.data import covalent_radii
from e3nn.o3 import Irreps, Linear
from torch_scatter import scatter

from estorch.datasets import CHARGES_KEY, ELECTROSTATIC_ENERGY_KEY, TOTAL_CHARGE_KEY
from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin

"""
Coulomb constant
ke = e^{2}/(4 * pi * epsilon_{0}) = 2.3070775523417355e-28 J.m = 14.399645478425668 eV.ang
    = e*e/(4*pi*epsilon_0) / eV / angstrom
"""
COULOMB_FACTOR = 14.399645478425668  # eV.ang


class ElectrostaticCorrection(GraphModuleMixin, torch.nn.Module):
    """
    Add electrostatic term to total energy:
        E_ele = ke * sum_{i < j} Q_{i} Q_{j} / r_{ij}
    Be careful this module takes O(N^2) time with N atoms!
    """

    def __init__(
        self,
        out_field: str = ELECTROSTATIC_ENERGY_KEY,
        irreps_in=None,
        pbc: bool = False,
        energy_scale: float = 1.0,  # std of energy in dataset, eV unit
    ):
        super().__init__()

        self.pbc = pbc
        if self.pbc:
            raise NotImplementedError(
                "Electrostatic correction for periodic systems is not implemented yet!"
            )

        self.energy_scale = energy_scale

        self.out_field = out_field
        irreps_out = {self.out_field: Irreps("1x0e")}
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[AtomicDataDict.POSITIONS_KEY, CHARGES_KEY],
            irreps_out=irreps_out,
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data[self.out_field] = self._calc_electrostatic(data)
        return data

    def _calc_electrostatic(self, data: AtomicDataDict.Type) -> torch.TensorType:
        data = AtomicDataDict.with_batch(data)

        # batch-wise pair indices of atoms
        pos = data[AtomicDataDict.POSITIONS_KEY]  # (num_atoms, 3)
        pair_indices, batch_indices = get_pair_indices_within_batch(data["ptr"], pos)

        # electrostatic term for each i-j pair within batch
        dists = torch.pairwise_distance(
            pos[pair_indices[0]], pos[pair_indices[1]], eps=1e-6, keepdim=True
        )
        charges = data[CHARGES_KEY]  # (num_atoms, 1)
        ele_pair = charges[pair_indices[0]] * charges[pair_indices[1]] / dists

        # sum over each batch
        ele = scatter(ele_pair, batch_indices, dim=0, reduce="sum")  # (batch_size, 1)
        ele = COULOMB_FACTOR * ele / self.energy_scale  # dimensionless

        return ele


class Qeq(GraphModuleMixin, torch.nn.Module):
    """
    Add electrostatic term in Qeq
    """

    def __init__(
        self,
        allowed_species: List[int],  # automatically passed from shared_params
        out_field: str = ELECTROSTATIC_ENERGY_KEY,
        irreps_in=None,
        pbc: bool = False,
        energy_scale: float = 1.0,  # std of energy in dataset, eV unit
    ):
        super().__init__()

        self.pbc = pbc
        if self.pbc:
            raise NotImplementedError(
                "Electrostatic correction for periodic systems is not implemented yet!"
            )

        self.energy_scale = energy_scale
        self.scaled_coulomb_factor = COULOMB_FACTOR / self.energy_scale  # dimensionless

        self.out_field = out_field
        irreps_out = {
            self.out_field: Irreps("1x0e"),
            CHARGES_KEY: Irreps("1x0e"),
        }
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[AtomicDataDict.POSITIONS_KEY, AtomicDataDict.NODE_FEATURES_KEY],
            irreps_out=irreps_out,
        )

        # sigma: species_index (0-indexed) -> covalent radius
        self.sigma = torch.from_numpy(np.array(covalent_radii[allowed_species]))
        if len(allowed_species) == 1:
            self.sigma = torch.unsqueeze(self.sigma, 0)

        self.to_chi = Linear(
            irreps_in=irreps_in[AtomicDataDict.NODE_FEATURES_KEY],
            irreps_out=Irreps("1x0e"),
        )
        hardness_params = torch.ones(len(allowed_species))
        self.to_hardness = torch.nn.Parameter(data=hardness_params)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        chi = self.to_chi(data[AtomicDataDict.NODE_FEATURES_KEY])  # (num_atoms, 1)
        species_idx = data[AtomicDataDict.SPECIES_INDEX_KEY]
        # square here to restrit hardness to be positive!
        hardness = torch.square(self.to_hardness[species_idx])  # (num_atoms, )

        # batch-wise pair indices of atoms
        pos = data[AtomicDataDict.POSITIONS_KEY]  # (num_atoms, 3)
        pair_indices, pair_batch_indices = get_pair_indices_within_batch(data["ptr"], pos)
        dists = torch.pairwise_distance(
            pos[pair_indices[0]], pos[pair_indices[1]], eps=1e-6, keepdim=False
        )
        device = pos.device

        num_atoms = pos.shape[0]
        coeffs = torch.zeros((num_atoms, num_atoms), device=device)
        sigmas = self.sigma[species_idx].to(device)  # (num_pairs, )
        gammas = torch.sqrt(
            sigmas[pair_indices[0]] ** 2 + sigmas[pair_indices[1]] ** 2
        )
        coeffs[pair_indices[0], pair_indices[1]] += (
            self.scaled_coulomb_factor * torch.erf(dists / math.sqrt(2.0) / gammas) / dists
        )
        coeffs += torch.transpose(coeffs, 0, 1)
        coeffs += torch.diag(hardness + self.scaled_coulomb_factor / (math.sqrt(math.pi) * sigmas))

        ptr = data["ptr"]
        batch_size = ptr.shape[0] - 1
        charges = []
        # TODO: avoid for-loop
        for bi in range(batch_size):
            num_atoms_bi = ptr[bi + 1] - ptr[bi]
            coeffs_bi = torch.ones((num_atoms_bi + 1, num_atoms_bi + 1), device=device)
            coeffs_bi[:num_atoms_bi, :num_atoms_bi] = coeffs[
                ptr[bi] : ptr[bi + 1], ptr[bi] : ptr[bi + 1]
            ]
            coeffs_bi[-1, -1] = 0.0

            total_charge_bi = torch.Tensor([[data[TOTAL_CHARGE_KEY][bi]]]).to(device)  # (1, 1)
            rhs_bi = torch.cat([-chi[ptr[bi] : ptr[bi + 1]], total_charge_bi])

            # solve Qeq
            # for small (n, n)-matrix (n < 2048), batched DGESV is faster than usual DGESV in MAGMA
            charges_and_lambda = torch.linalg.solve(
                torch.unsqueeze(coeffs_bi, dim=0), torch.unsqueeze(rhs_bi, dim=0)
            )
            charges_bi = torch.squeeze(charges_and_lambda, dim=0)[:-1]  # (num_atoms_bi, 1)
            charges.append(charges_bi)

        charges = torch.cat(charges)  # (num_atoms, 1)
        data[CHARGES_KEY] = charges

        # energy expression
        e_qeq = self._calc_qeq_energy(
            coeffs, chi, charges, pair_indices, pair_batch_indices, data[AtomicDataDict.BATCH_KEY]
        )
        data[self.out_field] = e_qeq

        return data

    def _calc_qeq_energy(self, coeffs, chi, charges, pair_indices, pair_batch_indices, batch):
        """
        Calculate Qeq-electrostatic energy

        Parameters
        ----------
        coeffs: (num_atoms, num_atoms) coefficient matrix of Qeq
        chi: (num_atoms, 1) electronegativity of atoms
        charges: (num_atoms, 1) atomic charges
        pair_indices:
        pair_batch_indices:
        batch: data['batch']

        Returns
        -------
        e_qeq: (batch_size, 1)
        """
        e_qeq_pair = (
            coeffs[pair_indices[0], pair_indices[1]][:, None]
            * charges[pair_indices[0]]
            * charges[pair_indices[1]]
        )
        e_qeq = scatter(e_qeq_pair, pair_batch_indices, dim=0, reduce="sum")  # (batch_size, 1)
        e_qeq_self = chi * charges + 0.5 * torch.diagonal(coeffs)[:, None] * torch.square(charges)
        e_qeq += scatter(e_qeq_self, batch, dim=0, reduce="sum")
        return e_qeq


class SumEnergies(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        input_fields: List[str],
        out_field: str = AtomicDataDict.TOTAL_ENERGY_KEY,
        irreps_in=None,
    ):
        super().__init__()

        self.input_fields = input_fields
        self.out_field = out_field
        irreps_out = {self.out_field: irreps_in[self.out_field]}
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[
                self.out_field,
            ]
            + input_fields,
            irreps_out=irreps_out,
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        total = torch.zeros_like(data[self.out_field])
        for field in self.input_fields:
            total += data[field]
        data[self.out_field] = total

        return data


class ChargeSkipConnection(GraphModuleMixin, torch.nn.Module):
    """
    add charges info. to output-hidden features in atomwise
        f_i <- f_i + Block(charge_i)
    """

    def __init__(
        self,
        irreps_in=None,
        field: str = CHARGES_KEY,
        out_field: str = AtomicDataDict.NODE_FEATURES_KEY,
    ):
        super().__init__()
        self.field = field
        self.out_field = out_field
        self.irreps_out = {
            self.out_field: irreps_in[self.out_field],
        }
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[self.field, self.out_field],
            irreps_out=self.irreps_out,
        )

        self.linear = Linear(
            irreps_in=self.irreps_in[self.field], irreps_out=self.irreps_out[self.out_field]
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        prev = data[self.out_field]
        residual = self.linear(data[self.field])
        data[self.out_field] = prev + residual
        return data


def get_pair_indices_within_batch(ptr, pos):
    device = pos.device
    batch_size = ptr.shape[0] - 1
    pair_indices_list = []  # node indices for each pair
    batch_indices_list = []  # pair index to batch index
    for bi in range(batch_size):
        num_atoms_bi = ptr[bi + 1] - ptr[bi]
        # off-diagonal triangular indices
        indices_bi = torch.triu_indices(
            num_atoms_bi, num_atoms_bi, offset=1, device=device
        ) + torch.tensor(ptr[bi])
        pair_indices_list.append(indices_bi)
        batch_indices_list.append(torch.tensor([bi]).repeat(indices_bi.shape[1]))
    pair_indices = torch.cat(pair_indices_list, dim=1)
    batch_indices = torch.cat(batch_indices_list, dim=0).to(device)
    return pair_indices, batch_indices
