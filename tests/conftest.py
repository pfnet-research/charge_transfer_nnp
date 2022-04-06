import os
from shutil import rmtree

import pytest

from nequip.utils.auto_init import dataset_from_config
from nequip.utils.config import Config


def create_config(name):
    # ref: https://github.com/mir-group/nequip/blob/main/tests/datasets/test_simplesets.py
    file_name = os.path.join(os.path.dirname(__file__), name)
    root = os.path.join(os.path.dirname(__file__), "results", name)

    config = Config(
        dict(
            # instanciated class
            dataset="estorch.datasets.fghdnnp.FGHDNNPDataset",
            # root directory to save dataset
            root=root,
            # file name of data source
            file_name=file_name,
            # keys to move from AtomicData to fixed_fields directory
            force_fixed_keys=[],
            # extra key that are not stored in data but needed for AtomicData initialization
            extra_fixed_fields={"r_max": 5},
            # frames to process with the constructor
            include_frames=None,
            batch_size=2,
        )
    )

    return config


def delete_processed_dataset(name):
    root = os.path.join(os.path.dirname(__file__), "results", name)
    rmtree(os.path.join(root, "processed"))


@pytest.fixture
def nonperiodic_dataset():
    name = "Ag_cluster_small"
    config = create_config(name)
    dataset = dataset_from_config(config)
    yield dataset, config
    delete_processed_dataset(name)


@pytest.fixture
def periodic_dataset():
    name = "AuMgO_small"
    config = create_config(name)
    dataset = dataset_from_config(config)
    yield dataset, config
    delete_processed_dataset(name)
