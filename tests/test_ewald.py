import math

import torch

from estorch.nn import EwaldAuxiliary


def test_ewald_sum_NaCl():
    # test Ewald summation by reproducing Madelung constant of NaCl structure
    L = 2.0
    cell = L * torch.eye(3)
    pos = L * torch.tensor(
        [
            # Na
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            # Cl
            [0.5, 0.5, 0.5],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
        ],
        requires_grad=True,
    )
    charges = torch.tensor([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0], requires_grad=True)

    ewald = EwaldAuxiliary(cell, pos, point_charge=True, accuracy=1e-8)
    energy_actual = ewald.calc_energy(charges)

    madelung_nacl = -1.747565
    r0 = L * 0.5
    energy_expect = pos.shape[0] / 2 * ewald.COULOMB_FACTOR * madelung_nacl / r0

    print(energy_actual, energy_expect)
    assert abs((energy_actual.item() - energy_expect) / energy_expect) < 1e-4


def test_ewald_sum_wurtzite():
    # test Ewald summation by reproducing Madelung constant of Wurtzite (ZnS) structure
    # See https://www.atomic-scale-physics.de/lattice/struk/b4.html
    L = 2.0
    c_by_a = math.sqrt(8 / 3)
    cell = L * torch.tensor(
        [
            [0.5, -0.5 * math.sqrt(3), 0.0],
            [0.5, 0.5 * math.sqrt(3), 0.0],
            [0.0, 0.0, c_by_a],
        ]
    )
    pos = L * torch.tensor(
        [
            # Zn
            [0.5, 0.5 / math.sqrt(3), 0.0],
            [0.5, -0.5 / math.sqrt(3), 0.5 * c_by_a],
            # Sn
            [0.5, 0.5 / math.sqrt(3), 0.375 * c_by_a],
            [0.5, -0.5 / math.sqrt(3), 0.875 * c_by_a],
        ],
        requires_grad=True,
    )
    charges = torch.tensor([2.0, 2.0, -2.0, -2.0], requires_grad=True)

    ewald = EwaldAuxiliary(cell, pos, point_charge=True, accuracy=1e-16)
    energy_actual = ewald.calc_energy(charges)

    madelung_wurtzite = -3.28264
    r0 = L * 0.375 * c_by_a
    energy_expect = charges[0] * pos.shape[0] / 2 * ewald.COULOMB_FACTOR * madelung_wurtzite / r0

    print(energy_actual, energy_expect)
    assert abs((energy_actual.item() - energy_expect) / energy_expect) < 1e-4
