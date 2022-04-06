# Charge transfer modeling with neural network potential

## Installation
1. Install [PyTorch](https://pytorch.org/get-started/locally/). This package is tested on
    - CUDA==11.1
    - Python==3.8.11
    - PyTorch==1.8.2
2. Install [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). **Note that this package does not work with the latest PyG's major version. Please install `torch_geometric<=1.7.2`.** For example,
```shell
export CUDA=cu111
export TORCH=1.8.0
python -m pip install torch-scatter==2.0.8 torch-sparse==0.6.11 torch-geometric==1.7.2 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
```
3. Install other dependencies
```shell
python -m pip install -r requirements.txt
```
4. Install this package
```shell
python -m pip install .
```

## Usage

### Prepare 4G-HDNNP datasets [1, 2]
Datasets are taken from 4G-HDNNPs [1, 2].
```shell
cd datasets
# If you fail to download the below url, try to download `datasets.tar.gz` directly
# from https://archive.materialscloud.org/record/2020.137
wget "https://archive.materialscloud.org/record/file?filename=datasets.tar.gz&record_id=629"
mv file\?filename\=datasets.tar.gz\&record_id\=629 datasets_runner.tar.gz
tar xzvf datasets_runner.tar.gz
python parse_datasets_runner.py 
```

Now you have the following file structure:
```
+
|- datasets
|   |- parse_datasets_runner.py
|   |- datasets_runner.tar.gz
|   |- datasets_runner
|   |- Ag_cluster
|   |   |- 0.json
|   |   |- ...
|   |- AuMgO
|   |- Carbon_chain
|   |- NaCl
-...
```
The units of the processed datasets are angstrom for distance, eV for total energies, eV/angstrom for forces, and elemental charge for charges.

### Train network
All settings for training are described with a YAML file. `estorch-train` command start to train a network.
```shell
estorch-train configs/minimal.yaml
```
Note that **`estorch-train` is assumed to be executed at the top of this reposity,** because a directory path for a dataset, `dataset_file_name` in the YAML file, may be relative.
The result are stored under `root` directory specified in the YAML file.

Example configurations are provided in [NequIP](https://github.com/mir-group/nequip#basic-network-training) [3-4], which this package is developed on the top of.
There are a few additional options for this package
```yaml
# in YAML file
use_charge: true  # iff true, use total_charge and predict atomic charges
use_ele: false  # iff true, calculate electrostatic term
use_qeq: true  # iff true, 
pbc: false  # iff true, is periodic system
```

Training can be automatically started and restarted by using `estorch-requeue` command
```shell
estorch-requeue configs/minimal_requeue.yaml
```

We provide some configurations for reproducing experiments.
- `configs/baseline/{system}_baseline.yaml`: Baseline model (NequIP) trained with 4G-HDNNP dataset
- `configs/charge/{system}_charge.yaml`: Predict directly atomic charges and add electrostatic energy
- `configs/qeq/{system}_qeq.yaml`: Predict atomic charges via charge equilibration scheme (Qeq) and add electrostatic energy

|               | YAML files              | use_charge | use_ele | use_qeq |
|---------------|-------------------------|------------|---------|---------|
| Base (NequIP) | configs/baseline/*.yaml | false      | false   | false   |
| Base w/ E_ele | configs/charge/*.yaml   | true       | true    | false   |
| Base w/ Qeq   | configs/qeq/*.yaml      | true       | true    | true    |

## How to load custom dataset
A loaded dataset is controlled by `dataset` and `dataset_file_name` keywords in the YAML file.
```yaml
# Example: in configs/minimal.yaml
dataset: estorch.datasets.fghdnnp.FGHDNNPDataset
dataset_file_name: datasets/Carbon_chain
```
`dataset` keyword specifies a module for creating datasets, which inherit `torch_geometric.data.Dataset`.

`dataset_file_name` keyword specifies a directory path for a raw dataset.
When we set `dataset: estorch.datasets.fghdnnp.FGHDNNPDataset`, this directory contains JSON files for structures.
Each JSON file has the following keys:
```
{
    "pos": ...,  // (num_atoms, 3) float array, positions of atoms in cartesian coordinates
    "symbols": ...,  // (num_atoms, ) str array, atomic species
    "charges": ...,  // (num_atoms, ) float array, atomic charges
    "total_energy": ...,  // float
    "forces": ...,  // (num_atoms, 3) float array, forces acting on atoms
    "total_charge": ...  // float
}
```

For more details, please read [NequIP's developer tutorial](https://github.com/mir-group/nequip#developers-tutorial).

## References
1. Tsz Wai Ko, Jonas A. Finkler, Stefan Goedecker, Jörg Behler, A fourth-generation high-dimensional neural network potential with accurate electrostatics including non-local charge transfer, [Nat. Commun. 12, 398 (2021)](https://www.nature.com/articles/s41467-020-20427-2).
1. https://archive.materialscloud.org/record/2020.137
1. S. Batzner et al., [arxiv:2101.03164](https://arxiv.org/abs/2101.03164)
1. https://github.com/mir-group/nequip
