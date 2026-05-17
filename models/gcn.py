import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F


class GCN (nn.Module):
    def __init__(self, in_channels, hidden_dim=128, num_classes=4, num_layers=3, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(x, edge_index):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        return self.classifier(x)

    @property
    def num_parameters(self):
        return sum(p.numel for p in self.parameters() if p.requires_grad)