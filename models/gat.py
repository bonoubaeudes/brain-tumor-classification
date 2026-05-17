import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool
import torch.nn.functional as F


class GAT(nn.Module):
    def __init__(self, in_channels:int, hidden_dim:int, num_classes:int,
                 num_layers:int, num_heads:int, dropout:int,
                 attn_dropout:float):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        head_dim = hidden_dim // num_heads

        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            in_dim = in_channels if i == 0 else hidden_dim

            if is_last:
                self.convs.append(GATv2Conv(
                    in_channels =  in_dim,
                    out_channels= hidden_dim,
                    heads = num_heads,
                    concat = False,
                    dropout = attn_dropout
                    ))
            else:
                self.convs.append(GATv2Conv(
                    in_channels =  in_dim,
                    out_channels= head_dim,
                    heads = num_heads,
                    concat = True,
                    dropout = attn_dropout
                ))

            self.norms.append(nn.LayerNorm(hidden_dim))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, data, return_attn=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        attn_list = []

        for conv, norm in zip(self.convs, self.norms):
            x, attn = conv(x, edge_index, return_attention_weihts=True)
            attn_list.append(attn)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        logits = self.classifier(x)

        if return_attn:
            return logits, attn_list
        return logits

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)