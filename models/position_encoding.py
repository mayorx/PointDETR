# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from util.misc import NestedTensor

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from util.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
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

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def calc_emb(self, normed_coord):
        normed_coord = normed_coord.clamp(0., 1.) * self.scale
        device = normed_coord.device

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = normed_coord[:, 0, None] / dim_t  # NxC
        pos_y = normed_coord[:, 1, None] / dim_t  # NxC

        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        pos = torch.cat((pos_y, pos_x), dim=1)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos

class PositionEmbeddingRelativeLearned(nn.Module):
    '''
    Relative Pos embedding, learned.
    '''

    def __init__(self, num_pos_feats=256, num_emb=51):
        # 0, 1, 2, .... num_emb
        # i / (num_emb - 1) , i = 0, 1, 2, ... num_emb - 1

        super().__init__()
        self.row_embed = nn.Embedding(num_emb, num_pos_feats)
        self.col_embed = nn.Embedding(num_emb, num_pos_feats)
        self.num_emb = num_emb
        self.reset_parameters()

        self.each_piece = 1 / (num_emb - 1)
        self.slices = torch.tensor([_ * self.each_piece for _ in range(num_emb)])
        # print(self.slices)

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        # NxHxW
        eps = 1e-6

        # norm
        y_embed = y_embed / (y_embed[:, -1:, :] + eps)
        x_embed = x_embed / (x_embed[:, :, -1:] + eps)

        weighted_col_emb = self.calc_weighted_emb_2d(x_embed, self.slices.to(x.device), self.col_embed.weight)
        weighted_row_emb = self.calc_weighted_emb_2d(y_embed, self.slices.to(x.device), self.row_embed.weight)

        return torch.cat([weighted_col_emb, weighted_row_emb], dim=3).permute(0, 3, 1, 2)

    """
    input:
        coord , NxHxW , norm to 0~1
        slices, num_emb
        embed, num_emb x num_pos_feats
        
    output:
        NxHxWxnum_pos_feats
    
    """
    def calc_weighted_emb_2d(self, coord, slices, embed):
        dis = abs(coord[:, :, :, None] - slices[None, :])
        weight = (-dis + self.each_piece) * (dis < self.each_piece) / self.each_piece  # NxHxWxnum_pos_feats
        return torch.matmul(weight, embed)

    def calc_emb(self, normed_coord):
        normed_coord = normed_coord.clamp(0., 1.)
        device = normed_coord.device

        # normed_coord[:, 0] #x
        # normed_coord[:, 1] #y

        weighted_col_emb = self.calc_weighted_emb(normed_coord[:, 0], self.slices.to(device), self.col_embed.weight)
        weighted_row_emb = self.calc_weighted_emb(normed_coord[:, 1], self.slices.to(device), self.row_embed.weight)
        return torch.cat([weighted_col_emb, weighted_row_emb], dim=1)  # Nx2

    def calc_weighted_emb(self, coord, slices, embed):
        # print('....  coord     slices', coord.size() , slices.size())
        dis = abs(coord[:, None] - slices[None, :])  # Nx(2num+1)
        weight = (self.each_piece - dis) * (dis < self.each_piece) / self.each_piece  # Nx(2num+1)
        return weight @ embed

def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    elif args.position_embedding in ('v4', 'relative-learned'):
        position_embedding = PositionEmbeddingRelativeLearned(N_steps, args.pos_emb_relative_dim)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
