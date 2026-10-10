"""Stub — TokenVideoTransformer is only used with objective=token_ce."""
import torch.nn as nn


class TokenVideoTransformer(nn.Module):
    def __init__(self, **kwargs):
        raise NotImplementedError("TokenVideoTransformer is not available in this installation. "
                                  "Use objective=mae or objective=diffusion instead.")
