

import math

import torch
from torch import nn

from utilities.misc import NestedTensor


class PositionEncodingSine1DRelative(nn.Module):
    """
    relative sine encoding 1D, partially inspired by DETR (https://github.com/facebookresearch/detr)
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    @torch.no_grad()
    def forward(self, inputs: NestedTensor):
        """
        :param inputs: NestedTensor
        :return: pos encoding [2W-1,C]
        """
        x = inputs.left
        bs, _, h, w = x.size()
        #1/4
        w = math.ceil(w / 4)
        h = math.ceil(h / 4)
        #1/8
        w1 = math.ceil(w / 2)
        h1 = math.ceil(h / 2)
        # populate all possible relative distances
        x_embed = torch.linspace(w - 1, -w + 1, 2 * w - 1, dtype=torch.float32, device=x.device)
        y_embed = torch.linspace(h - 1, -h + 1, 2 * h - 1, dtype=torch.float32, device=x.device)
        x_embed1 = torch.linspace(w1 - 1, -w1 + 1, 2 * w1 - 1, dtype=torch.float32, device=x.device)

        if self.normalize:
            x_embed = x_embed * self.scale
            y_embed = y_embed * self.scale
            x_embed1 = x_embed1 * self.scale


        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t  # 2W-1xC
        pos_y = y_embed[:, None] / dim_t  # 2H-1xC
        pos_x1 = x_embed1[:, None] / dim_t  # 2W-1xC

        # interleave cos and sin instead of concatenate
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)  # 2W-1xC
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)   # 2H-1xC
        pos_x1 = torch.stack((pos_x1[:, 0::2].sin(), pos_x1[:, 1::2].cos()), dim=2).flatten(1)  # 2W-1xC
        pos_y1=pos_y

        return pos_x, pos_y,pos_x1, pos_y1


def no_pos_encoding(x):
    return None


def build_position_encoding(args):
    mode = args.position_encoding
    channel_dim = args.channel_dim
    nheads = args.nheads
    if mode == 'sine1d_rel':
        n_steps = channel_dim
        position_encoding= PositionEncodingSine1DRelative(n_steps, normalize=False)
        
    elif mode == 'none':
        position_encoding_x,position_encoding_y = no_pos_encoding,no_pos_encoding
    else:
        raise ValueError(f"not supported {mode}")

    return position_encoding
