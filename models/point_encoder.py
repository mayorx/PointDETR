import torch
import torch.nn as nn

class PointEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.query_emb = nn.Embedding(100, 256)


    #Points , Nx2
    #labels , N,
    #object_ids N,
    # def forward(self, batched_points, batched_labels, batched_object_ids, pos_encoder, label_encoder):
    def forward(self, points_supervision, pos_encoder, label_encoder, no_label_enc, no_pos_enc, device):
        # batch_size = len(batched_points)
        batch_size = len(points_supervision)
        #position embedding .... by points
        #label embedding ... by labels
        #... object embedding ??  ...
        #feature embedding .. by points && features (interpolation)

        embeddings = []
        for idx in range(batch_size):
            position_embedding = pos_encoder.calc_emb(points_supervision[idx]['points'])

            if no_label_enc:
                label_embedding = torch.zeros((position_embedding.size())).to(device)
                # N = len(points_supervision[idx]['points'])
                # label_embedding = pos_encoder.calc_emb(rand_pos)
            else:
                label_embedding = label_encoder.calc_emb(points_supervision[idx]['labels'])

            if no_pos_enc:
                position_embedding = torch.zeros((position_embedding.size())).to(device) #..

            query_embedding = position_embedding + label_embedding

            if no_label_enc and no_pos_enc:
                # print('position_embedding .. size ', query_embedding.size())
                N = len(position_embedding)
                query_embedding = self.query_emb.weight[:N]
                # print('query embedding .. size ', query_embedding.size())

            embeddings.append(query_embedding)

            # print('label embedding', label_embedding)
            # print('label_embedding', label_embedding.mean(), label_embedding.var(), label_embedding.size())
            # print('position embedding .. ', position_embedding.mean(), position_embedding.var(), position_embedding.size())
            # embeddings.append(label_embedding)
        return embeddings



def build_point_encoder():
    return PointEncoder()
