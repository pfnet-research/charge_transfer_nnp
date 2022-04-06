import logging

from e3nn.o3 import Irreps

from estorch.datasets import CHARGES_KEY, ELECTROSTATIC_ENERGY_KEY, TOTAL_CHARGE_KEY
from estorch.nn import (
    AttentionBlock,
    ChargeSkipConnection,
    ElectrostaticCorrection,
    Ewald,
    EwaldQeq,
    Qeq,
    SumEnergies,
    TotalChargeEmbedding,
)
from nequip.data import AtomicDataDict
from nequip.nn import (
    AtomwiseLinear,
    AtomwiseReduce,
    ConvNetLayer,
    ForceOutput,
    GraphModuleMixin,
    PerSpeciesScaleShift,
    SequentialGraphNetwork,
)
from nequip.nn.embedding import (
    OneHotAtomEncoding,
    RadialBasisEdgeEncoding,
    SphericalHarmonicEdgeAttrs,
)


def EnergyChargeModel(**shared_params) -> SequentialGraphNetwork:
    """
    Energy-and-charge model architecture based on nequip.models._eng.EnergyModel
    """
    logging.debug("Start building the network model")

    num_layers = shared_params.pop("num_layers", 3)
    add_per_species_shift = shared_params.pop("PerSpeciesScaleShift_enable", False)
    pbc = shared_params.pop("pbc", False)

    use_charge = shared_params.pop("use_charge", False)
    use_qtot = shared_params.pop("use_qtot", False)  # enable if only use Q_tot
    use_ele = shared_params.pop("use_ele", False)
    use_qeq = shared_params.pop("use_qeq", False)
    if use_ele and (not use_charge):
        logging.info("Use ground-truth charges. Be careful what you did!")
    if use_qeq and (not use_charge):
        raise ValueError("Set use_charge: true to enable use_qeq: true !")
    if use_ele and use_qeq:
        raise ValueError("Set use_ele xor use_qeq for electrostatic correction!")
    use_nonlocal = shared_params.pop("use_nonlocal", False)
    energy_scale = shared_params.pop("_global_scale", 1.0)

    layers = {
        # -- Encode --
        "one_hot": OneHotAtomEncoding,
        "spharm_edges": SphericalHarmonicEdgeAttrs,
        "radial_basis": RadialBasisEdgeEncoding,
        # -- Embed features --
        "chemical_embedding": AtomwiseLinear,
    }

    if use_qtot or use_ele or use_qeq:
        # place TotalChargeEmbedding after chemical_embedding layer
        layers["total_charge_embedding"] = TotalChargeEmbedding

    # TODO: nonlocal-interaction block in ConvNetLayer
    # add convnet layers
    # insertion preserves order
    for layer_i in range(num_layers):
        layers[f"layer{layer_i}_convnet"] = ConvNetLayer
        if use_nonlocal:
            layers[f"layer{layer_i}_attention"] = AttentionBlock

    layers["conv_to_output_hidden"] = AtomwiseLinear

    # charge term
    if use_charge and use_ele:
        layers["atomic_charges"] = (
            AtomwiseLinear,
            dict(
                irreps_out="1x0e",
                field=AtomicDataDict.NODE_FEATURES_KEY,  # "node_features"
                out_field=CHARGES_KEY,
            ),
        )

    if use_ele:
        if pbc:
            # for periodic system, calculate electrostatic energy via Ewald summation
            layers["total_energy_with_ele"] = (
                Ewald,
                dict(scale=energy_scale),
            )
        else:
            # for nonperiodic system, calculate electrostatic energy directly
            layers["total_energy_with_ele"] = (
                ElectrostaticCorrection,
                dict(pbc=pbc, energy_scale=energy_scale),
            )
    elif use_qeq:
        # atomic charges are also generated in Qeq block
        if pbc:
            layers["total_energy_with_qeq"] = (
                EwaldQeq,
                dict(scale=energy_scale),
            )
        else:
            layers["total_energy_with_qeq"] = (
                Qeq,
                dict(pbc=pbc, energy_scale=energy_scale),
            )

        # add charges to output-hidden
        layers["add_charges_to_output_hidden"] = ChargeSkipConnection

    # short-range atomic energy
    layers["output_hidden_to_scalar"] = (
        AtomwiseLinear,
        dict(irreps_out="1x0e", out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY),
    )

    if add_per_species_shift:
        layers["per_species_scale_shift"] = (
            PerSpeciesScaleShift,
            dict(
                field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
                out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
            ),
        )

    layers["total_energy_sum"] = (
        AtomwiseReduce,
        dict(
            reduce="sum",
            field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
            out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
        ),
    )

    # add E_ele term into AtomicDataDict.TOTAL_ENERGY_KEY
    if use_ele or use_qeq:
        layers["sum_energy_terms"] = (
            SumEnergies,
            dict(
                input_fields=[AtomicDataDict.TOTAL_ENERGY_KEY, ELECTROSTATIC_ENERGY_KEY],
            ),
        )

    # additional irreps_in for charges
    irreps_in = {
        TOTAL_CHARGE_KEY: Irreps("1x0e"),  # total charge is scalar
    }
    if (not use_charge) and use_ele:
        # debug option
        irreps_in[CHARGES_KEY] = Irreps("1x0e")

    return SequentialGraphNetwork.from_parameters(
        shared_params=shared_params,
        layers=layers,
        irreps_in=irreps_in,
    )


def ForceChargeModel(**shared_params) -> GraphModuleMixin:
    """
    energy-charge-force model architecture based on nequip.models._eng.ForceModel
    """
    energy_charge_model = EnergyChargeModel(**shared_params)
    return ForceOutput(energy_model=energy_charge_model)
