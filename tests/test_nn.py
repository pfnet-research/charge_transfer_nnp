import torch
from e3nn import o3
from e3nn.o3 import Irreps
from torch_scatter import scatter

from estorch.datasets import CHARGES_KEY, TOTAL_CHARGE_KEY
from estorch.nn import AttentionBlock, Qeq, TotalChargeEmbedding
from nequip.data import AtomicDataDict, Collater
from nequip.nn.embedding import OneHotAtomEncoding


def get_batches(dataset):
    b = []
    c = Collater.for_dataset(dataset, exclude_keys=[])
    for idx in [[0], [1], [0, 1]]:
        b.append(c.collate(dataset[idx]))
    return b


def test_total_charge_embedding(nonperiodic_dataset):
    dataset, config = nonperiodic_dataset
    node_features_irreps = Irreps("3x0e")
    irreps_in = {
        TOTAL_CHARGE_KEY: Irreps("1x0e"),  # total charge is scalar
        AtomicDataDict.NODE_FEATURES_KEY: node_features_irreps,  # temporal
    }
    obj = TotalChargeEmbedding(irreps_in=irreps_in)

    data = dataset[0]
    data[AtomicDataDict.NODE_FEATURES_KEY] = node_features_irreps.randn(-1)
    obj.forward(data)


def test_qeq(nonperiodic_dataset):
    dataset, config = nonperiodic_dataset
    batches = get_batches(dataset)
    node_features_irreps = Irreps("1x0e")
    irreps_in = {
        TOTAL_CHARGE_KEY: Irreps("1x0e"),  # total charge is scalar
        AtomicDataDict.NODE_FEATURES_KEY: node_features_irreps,  # temporal
    }
    allowed_species = torch.tensor(
        [
            47,
        ],
        dtype=torch.long,
    )  # Ag

    embedding = OneHotAtomEncoding(allowed_species=allowed_species, set_features=True)
    nn = Qeq(irreps_in=irreps_in, allowed_species=allowed_species)

    for data in batches:
        data = embedding(data)
        data = nn.forward(data)

        # check total charges
        total_charges_actual = scatter(
            data[CHARGES_KEY], data[AtomicDataDict.BATCH_KEY], dim=0, reduce="sum"
        )
        total_charges_expect = data[TOTAL_CHARGE_KEY]
        assert torch.allclose(total_charges_actual, total_charges_expect)


def test_attention_block(nonperiodic_dataset):
    dataset, config = nonperiodic_dataset
    batches = get_batches(dataset)
    node_features_irreps = Irreps("3x0e + 5x1o")
    irreps_in = {
        AtomicDataDict.NODE_FEATURES_KEY: node_features_irreps,  # temporal
    }

    rot = o3.rand_matrix()

    nn = AttentionBlock(irreps_in=irreps_in)
    D_in = nn.irreps_in[AtomicDataDict.NODE_FEATURES_KEY].D_from_matrix(rot)
    D_out = nn.irreps_out[AtomicDataDict.NODE_FEATURES_KEY].D_from_matrix(rot)

    for data in batches:
        num_atoms = data[AtomicDataDict.POSITIONS_KEY].shape[0]

        features = node_features_irreps.randn(num_atoms, -1)
        data[AtomicDataDict.NODE_FEATURES_KEY] = features @ D_in.T
        data = nn(data)
        features_before = data[AtomicDataDict.NODE_FEATURES_KEY]

        data[AtomicDataDict.NODE_FEATURES_KEY] = features
        data = nn(data)
        features_after = data[AtomicDataDict.NODE_FEATURES_KEY] @ D_out.T

        assert torch.allclose(features_before, features_after, atol=1e-3, rtol=1e-3)
