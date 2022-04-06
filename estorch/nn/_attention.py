import math

import torch
from e3nn.nn import NormActivation
from e3nn.o3 import FullyConnectedTensorProduct, Linear
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin
from nequip.nn.nonlinearities import ShiftedSoftPlus


class AttentionBlock(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        irreps_in,
    ):
        super().__init__()

        self.feature_irreps = irreps_in[AtomicDataDict.NODE_FEATURES_KEY]
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={AtomicDataDict.NODE_FEATURES_KEY: self.feature_irreps},
            required_irreps_in=[AtomicDataDict.NODE_FEATURES_KEY],
        )

        self.linear_q = Linear(
            irreps_in=self.feature_irreps,
            irreps_out=self.feature_irreps,
        )
        self.linear_k = Linear(
            irreps_in=self.feature_irreps,
            irreps_out=self.feature_irreps,
        )
        self.linear_v = Linear(
            irreps_in=self.feature_irreps,
            irreps_out=self.feature_irreps,
        )
        self.dot_qk = FullyConnectedTensorProduct(self.feature_irreps, self.feature_irreps, "0e")

        self.equivariant_nonlin = NormActivation(
            irreps_in=self.feature_irreps,
            scalar_nonlinearity=ShiftedSoftPlus,  # https://paperswithcode.com/method/ssp
            normalize=True,
            epsilon=1e-8,
            bias=False,
        )

        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # Ref. https://docs.e3nn.org/en/stable/guide/transformer.html
        # save old features for resnet
        old_x = data[AtomicDataDict.NODE_FEATURES_KEY]

        query = self.equivariant_nonlin(self.linear_q(old_x))
        key = self.equivariant_nonlin(self.linear_k(old_x))
        value = self.equivariant_nonlin(self.linear_v(old_x))  # (num_atoms, feature_size)

        feature_size = query.shape[1]
        alpha = self.softmax(self.dot_qk(query, key) / math.sqrt(feature_size))

        edges = []
        ptr = data["ptr"]
        batch_size = ptr.shape[0] - 1
        device = data[AtomicDataDict.POSITIONS_KEY].device
        for bi in range(batch_size):
            num_atoms_bi = ptr[bi + 1] - ptr[bi]
            grid = torch.cartesian_prod(
                torch.arange(num_atoms_bi, device=device),
                torch.arange(num_atoms_bi, device=device),
            )
            grid += ptr[bi]
            edges.append(grid)
        edges = torch.cat(edges)
        edges_src = edges[:, 0]
        edges_dst = edges[:, 1]

        # `qk` is scalar array
        qk = self.dot_qk(query[edges_src], key[edges_dst]) / math.sqrt(feature_size)
        alpha = scatter_softmax(qk, torch.unsqueeze(edges_dst, dim=1), dim=0)

        out = scatter(alpha * value[edges_dst], edges_dst, dim=0, reduce="sum")
        data[AtomicDataDict.NODE_FEATURES_KEY] = old_x + out

        return data
