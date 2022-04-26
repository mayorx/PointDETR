
import torch
from torch import nn

class LabelEncoder(nn.Module):

    def __init__(self, num_feats, num_classes):
        super().__init__()

        self.label_embed = nn.Embedding(num_classes, num_feats)
        # self.no_label_encoder = no_label_encoder
        self.num_feats = num_feats
        nn.init.uniform_(self.label_embed.weight)

    def forward(self, labels):
        emb = self.label_embed.weight[labels]
        return emb

    def calc_emb(self, labels):
        return self.forward(labels)

class SelfAttnLabelEncoder(LabelEncoder):

    def __init__(self, num_feats, num_classes):
        super().__init__(num_feats, num_classes)
        self.self_attn = nn.MultiheadAttention(num_feats, num_heads=8, dropout=0.1)

    def forward(self, labels):
        w = self.label_embed.weight.unsqueeze(dim=1)
        emb = self.self_attn(w, w, w)[0].squeeze(dim=1)[labels]
        return emb

def build_label_encoder(args, num_classes):
    if args.self_attn_label_encoder:
        return SelfAttnLabelEncoder(args.hidden_dim, num_classes)
    return LabelEncoder(args.hidden_dim, num_classes)




