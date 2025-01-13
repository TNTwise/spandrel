from __future__ import annotations

from typing_extensions import override

from spandrel.util import KeyCondition, get_pixelshuffle_params, get_seq_len

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .__arch.RCAN import RCAN


def get_scale(state):
    scale = get_seq_len(state, "tail.0") + 1
    try:
        if state["tail.0.0.weight"].shape[0] == 576:  # if scale is 3
            scale += 1
    except Exception:
        pass
    return scale  # this detection method only works up to 4


def get_n_resgroups(state) -> int:
    return get_seq_len(state, "body") - 1


def get_n_resblocks(state) -> int:
    return get_seq_len(state, "body.0.body") - 1


def get_n_reduction(state) -> int:
    return int(64 / state["body.0.body.0.body.3.conv_du.0.weight"].shape[0])


def is_unshuffle(state) -> bool:
    try:
        unshuffle = (
            state["head.1.weight"].shape[1] == 12
        )  # this means 12 channels, which is the number of channels for unshuffle
    except Exception:
        unshuffle = False
    return unshuffle


def is_norm(state) -> bool:
    return "sub_mean.weight" in state


def get_n_feats(state) -> int:
    n_feats = state["body.0.body.0.body.0.weight"].shape[0]
    return n_feats


def get_kernel_size(state) -> int:
    k_s = state["body.0.body.0.body.0.weight"].shape[2]
    return k_s


class RCANArch(Architecture[RCAN]):
    def __init__(self) -> None:
        super().__init__(
            id="RCAN",
            detect=KeyCondition.has_all(
                "body.0.body.0.body.0.weight",
                "body.0.body.0.body.3.conv_du.0.weight",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[RCAN]:
        in_channels = 3
        out_channels = 3

        n_resblocks = get_n_resblocks(state_dict)
        n_resgroups = get_n_resgroups(state_dict)
        reduction = get_n_reduction(state_dict)
        n_feats = get_n_feats(state_dict)
        kernel_size = get_kernel_size(state_dict)
        norm = is_norm(state_dict)
        unshuffle_mod = is_unshuffle(state_dict)
        scale, n_feats = get_pixelshuffle_params(state_dict, "tail.0")
        if unshuffle_mod:
            scale = 2
            size_requirements = SizeRequirements(multiple_of=2)
        else:
            size_requirements = SizeRequirements(multiple_of=1)

        model = RCAN(
            scale=scale,
            n_resgroups=n_resgroups,
            n_resblocks=n_resblocks,
            n_feats=n_feats,
            n_colors=3,
            rgb_range=255,
            norm=norm,
            kernel_size=kernel_size,
            reduction=reduction,
            res_scale=1,
            act_mode="relu",
            unshuffle_mod=unshuffle_mod,
        )
        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="SR",
            tags=[f"{n_feats}dim", f"{n_resblocks}nb", f"{kernel_size}ks"],
            supports_half=True,
            supports_bfloat16=True,
            scale=scale,
            input_channels=in_channels,
            output_channels=out_channels,
            size_requirements=size_requirements,
        )


__all__ = ["RCANArch", "RCAN"]
