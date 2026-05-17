import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, global_mean_pool


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    def forward(self, x): return self.net(x)


class VGRNN(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim,
                 gru_hidden, num_classes, num_gcn_layers, dropout):
        super().__init__()
        self.latent_dim = latent_dim
        self.gru_hidden = gru_hidden
        self.dropout = dropout

        # ── φˣ: node feature projection ─────────────────────────
        self.phi_x = MLP(in_channels, hidden_dim, dropout)

        # ── GCN encoder (shared layers) ──────────────────────────
        # Input to first conv: [φˣ(X) ‖ h_{t-1}] → hidden_dim + gru_hidden
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        first_in = hidden_dim + gru_hidden  # concat of features + hidden state
        self.convs.append(GCNConv(first_in, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_gcn_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # ── Variational heads: μ and log-σ ───────────────────────
        self.gcn_mu = nn.Linear(hidden_dim, latent_dim)
        self.gcn_sigma = nn.Linear(hidden_dim, latent_dim)

        # ── Prior network: p(Z | h_{t-1}) ────────────────────────
        self.phi_prior = nn.Sequential(
            nn.Linear(gru_hidden, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim)               # → [μ_prior | log-σ_prior]
        )

        # ── φᶻ: latent feature projection ───────────────────────
        self.phi_z = MLP(latent_dim, latent_dim, dropout)

        # ── GRUCell ──────────────────────────────────────────────
        # Input: [φˣ_pool ‖ φᶻ_pool]
        self.gru = nn.GRUCell(
            input_size = hidden_dim + latent_dim,
            hidden_size = gru_hidden
        )

        # ── MLP classifier ───────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden // 2, num_classes)
        )

    @staticmethod
    def reparameterize(mu, log_sigma):
        """
        Z = μ + ε ⊙ σ,  ε ~ N(0, I)
        log_sigma is used for numerical stability.
        """
        if not torch.is_grad_enabled():
            return mu
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ── KL divergence (closed form, two Gaussians) ────────────────
    @staticmethod
    def kl_divergence(mu_enc, log_sigma_enc, mu_prior, log_sigma_prior):
        sigma_enc_sq = torch.exp(log_sigma_enc)
        sigma_prior_sq = torch.exp(log_sigma_prior)
        kl = 0.5 * (
            sigma_enc_sq / sigma_prior_sq + (mu_enc - mu_prior).pow(2) / sigma_prior_sq
            - 1.0 + log_sigma_prior - log_sigma_enc
        )
        return kl.sum(dim=-1).mean()

    def _gcn_encode(self, edge_index, node_feats):
        x = node_feats
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward_snapshot(self, data, h):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        N_total = x.size(0)
        B = h.size(0)

        # 1. φˣ: project node features
        x_proj = self.phi_x(x)

        # 2. Expand h to node level: each node gets its graph's hidden state
        h_node = h[batch]

        # 3. Concatenate [φˣ(X) ‖ h_expanded] and encode with GCN
        node_input = torch.cat([x_proj, h_node], dim=1)
        gcn_out = self._gcn_encode(edge_index, node_input)

        # 4. Variational posterior parameters
        mu_enc = self.gcn_mu(gcn_out)
        log_sig_enc = self.gcn_sigma(gcn_out)

        # 5. Prior parameters conditioned on h
        prior_params = self.phi_prior(h)
        mu_prior = prior_params[:, :self.latent_dim]
        log_sig_prior = prior_params[:, self.latent_dim:]

        mu_prior_node = mu_prior[batch]
        log_sig_prior_node = log_sig_prior[batch]

        # 6. KL divergence (node-level, averaged)
        kl = self.kl_divergence(
            mu_enc, log_sig_enc, mu_prior_node, log_sig_prior_node
        )

        # 7. Sample Z via reparameterization
        z = self.reparameterize(mu_enc, log_sig_enc)

        # 8. φᶻ: project latent samples
        z_proj = self.phi_z(z)

        # 9. Graph-level pooling of both projections
        g_x = global_mean_pool(x_proj, batch)
        g_z = global_mean_pool(z_proj, batch)

        # 10. GRU update
        gru_input = torch.cat([g_x, g_z], dim=1)
        h_new = self.gru(gru_input, h)

        return h_new, kl, mu_enc, log_sig_enc

    def forward_sequence(self, snapshots:list, h=None, t=4):
        B = snapshots[0].num_graphs
        total_kl = torch.tensor(0.0, device="cuda" if torch.cuda.is_available() else "cpu")

        if h is None:
            h = torch.zeros(B, self.gru_hidden, device="cuda" if torch.cuda.is_available() else "cpu")

        for snapshot in snapshots:
            h, kl, _, _ = self.forward_snapshot(snapshot, h)
            total_kl = total_kl + kl

        logits = self.classifier(h)
        return logits, h, total_kl / t

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)




