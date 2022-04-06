import torch
from e3nn.o3 import Linear

from estorch.datasets import TOTAL_CHARGE_KEY
from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin


class TotalChargeEmbedding(GraphModuleMixin, torch.nn.Module):
    """
    Embed total charge and add it into chemical embedding
    """

    def __init__(
        self,
        irreps_in=None,
    ):
        super().__init__()
        self.field = TOTAL_CHARGE_KEY
        self.out_field = AtomicDataDict.NODE_FEATURES_KEY
        self.irreps_out = {
            self.out_field: irreps_in[self.out_field],
        }
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[self.field, AtomicDataDict.NODE_FEATURES_KEY],
            irreps_out=self.irreps_out,
        )

        # take total_charge and return embedding with the same size with node_features
        self.linear = Linear(
            irreps_in=self.irreps_in[self.field], irreps_out=self.irreps_out[self.out_field]
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        total_charge_embedding = self.linear(
            data[self.field]
        )  # (batch_size, chemical_embedding_irreps_out)
        data[self.out_field] = (
            data[self.out_field] + total_charge_embedding[data[AtomicDataDict.BATCH_KEY], :]
        )
        return data
