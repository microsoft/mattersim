# -*- coding: utf-8 -*-
import logging
import pathlib
import sys
from typing import Dict, Tuple, Union

if sys.version_info[1] >= 8:
    from typing import Final
else:
    from typing_extensions import Final

# This is a weird hack to avoid Intel MKL issues on the cluster
import ase.data

# when this is called as a subprocess of a process that has itself initialized
# PyTorch. Since numpy gets imported later anyway for dataset stuff,
# this shouldn't affect performance.
import numpy as np  # noqa: F401
import torch

from .jit import script

# Denote meta_data_keys
TWO_BODY_CUTOFF: Final[str] = "two_body_cutoff"
HAS_THREE_BODY: Final[str] = "has_three_body"
THREE_BODY_CUTOFF: Final[str] = "three_body_cutoff"
N_SPECIES_KEY: Final[str] = "n_species"
TYPE_NAMES_KEY: Final[str] = "type_names"
JIT_BAILOUT_KEY: Final[str] = "_jit_bailout_depth"
JIT_FUSION_STRATEGY: Final[str] = "_jit_fusion_strategy"
TF32_KEY: Final[str] = "allow_tf32"

_ALL_METADATA_KEYS = [
    TWO_BODY_CUTOFF,
    HAS_THREE_BODY,
    THREE_BODY_CUTOFF,
    N_SPECIES_KEY,
    TYPE_NAMES_KEY,
    JIT_BAILOUT_KEY,
    JIT_FUSION_STRATEGY,
    TF32_KEY,
]


def _compile_for_deploy(model):
    model.eval()

    if not isinstance(model, torch.jit.ScriptModule):
        print("Non TorchScript model detected,JIT  compiling the model ....")
        model = script(model)
    else:
        print(
            "Model provided is already a TorchScript model, "
            "return as it is."  # noqa: E501
        )
    return model


def load_deployed_model(
    model_path: Union[pathlib.Path, str],
    device: Union[str, torch.device] = "cpu",
    freeze: bool = True,
) -> Tuple[torch.jit.ScriptModule, Dict[str, str]]:
    r"""Load a deployed model.
    Args:
        model_path: the path to the deployed model's ``.pth`` file.
    Returns:
        model, metadata dictionary
    """
    metadata = {k: "" for k in _ALL_METADATA_KEYS}
    try:
        model = torch.jit.load(
            model_path, map_location=device, _extra_files=metadata
        )  # noqa: E501
    except RuntimeError as e:
        raise ValueError(
            f"{model_path} does not seem to be a deployed RL4CSP model file. "
            f"Did you forget to deploy it? \n\n(Underlying error: {e})"
        )

    # Confirm its TorchScript
    assert isinstance(model, torch.jit.ScriptModule)

    # Make sure we're in eval mode
    model.eval()
    # Freeze on load:
    if freeze and hasattr(model, "training"):
        # hasattr is how torch checks whether model is unfrozen
        # only freeze if already unfrozen
        model = torch.jit.freeze(model)

    # Everything we store right now is ASCII, so decode for printing
    metadata = {k: v.decode("ascii") for k, v in metadata.items()}

    # JIT strategy
    strategy = metadata.get(JIT_FUSION_STRATEGY, "")

    if strategy != "":
        strategy = [e.split(",") for e in strategy.split(";")]
        strategy = [(e[0], int(e[1])) for e in strategy]
    else:
        print(
            "Missing information: JIT strategy, "
            "loading deployed model fails !"  # noqa: E501
        )
        exit()

    # JIT bailout
    jit_bailout: int = metadata.get(JIT_BAILOUT_KEY, "")
    if jit_bailout == "":
        print(
            "Missing information: JIT_BAILOUT_KEY, "
            "loading deployed model fails !"  # noqa: E501
        )
        exit()

    # JIT allow_tf32
    jit_allow_tf32: int = metadata.get(TF32_KEY, "")
    if jit_allow_tf32 == "":
        print("Missing information: TF32_KEY, loading deployed model fails !")
        exit()

    return model, metadata


def deploy(
    model,
    is_m3gnet_pretrained=False,
    is_m3gnet_multi_head_pretrained=False,
    metadata=None,
    deployed_model_name="deployed.pth",
    device="cpu",
):
    # Compile model
    complied_model = _compile_for_deploy(model)

    # Use default metadata dictionary for pretrained models
    if is_m3gnet_pretrained:
        metadata = {}

        # Do set differences get atomic numbers
        full_atomic_numbers = set(np.arange(1, 95, 1))
        discard_atomic_numbers = set(np.arange(84, 89, 1))
        covered_atomic_numbers = list(
            full_atomic_numbers.difference(discard_atomic_numbers)
        )
        type_names = []
        for atomic_num in covered_atomic_numbers:
            type_names.append(ase.data.chemical_symbols[atomic_num])
        metadata[TWO_BODY_CUTOFF] = str(5.0)
        metadata[HAS_THREE_BODY] = str(True)
        metadata[THREE_BODY_CUTOFF] = str(4.0)
        metadata[N_SPECIES_KEY] = str(89)
        metadata[TYPE_NAMES_KEY] = " ".join(type_names)
        metadata[JIT_BAILOUT_KEY] = str(2)
        metadata[JIT_FUSION_STRATEGY] = ";".join(
            "%s,%i" % e for e in [("DYNAMIC", 3)]  # noqa: E501
        )
        metadata[TF32_KEY] = str(int(0))

    # TODO: Add default meta keys for m3gent_multi_head models
    # elif is_m3gnet_multi_head_pretrained:

    else:
        # Missing fields in meta data triggers failing compilation
        metadata_keys = metadata.keys
        for _ALL_METADATA_KEY in _ALL_METADATA_KEYS:
            if _ALL_METADATA_KEY not in metadata_keys:
                logging.info(
                    "Miss metadata key: "
                    + _ALL_METADATA_KEY
                    + " model deploying fails!"
                )
                exit()
        # Missing metadata values, other than JIT compile information,
        # triggers failing compilation
        for i in range(len(metadata_keys) - 3):
            if metadata[metadata_keys[i]].empty():
                logging.info(
                    "metadata with key "
                    + metadata_keys
                    + "not set, model deploying fails!"
                )
                exit()
        # Set default JIT compile information is values are not set
        if (
            metadata["JIT_BAILOUT_KEY"].empty()
            or metadata[JIT_FUSION_STRATEGY].empty()
            or metadata[TF32_KEY].empty()
        ):
            metadata[JIT_BAILOUT_KEY] = str(2)
            metadata[JIT_FUSION_STRATEGY] = ";".join(
                "%s,%i" % e for e in [("DYNAMIC", 3)]
            )
            metadata[TF32_KEY] = str(int(0))

    # Deploy model with full information
    # Confirm its TorchScript
    assert isinstance(complied_model, torch.jit.ScriptModule)
    if device != "cuda":
        complied_model = complied_model.cpu()

    torch.jit.save(complied_model, deployed_model_name, _extra_files=metadata)

    return complied_model, metadata
