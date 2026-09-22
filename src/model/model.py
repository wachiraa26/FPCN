import torch
from torch import nn

try:
    from torch_geometric.nn import GATv2Conv, global_mean_pool
except ImportError as exc:
    GATv2Conv = None
    global_mean_pool = None
    _PYG_IMPORT_ERROR = exc


class GNNConv(nn.Module):
    """GATv2 message with higher-rank paths used as edge attributes."""

    def __init__(self, hidden_dim, heads=4, dropout=0.2):
        super().__init__()
        if GATv2Conv is None:
            raise ImportError(
                "torch_geometric is required for the GATv2 path-complex branch. "
                "Install torch-geometric in the DNA_SEQ environment."
            ) from _PYG_IMPORT_ERROR
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by pcnn_heads.")
        self.conv = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim // heads, heads=heads, concat=True, add_self_loops=False, edge_dim=hidden_dim, dropout=dropout)

    def forward(self, x, edge_index, edge_attr):
        if x.numel() == 0 or edge_index.numel() == 0 or edge_attr.numel() == 0:
            return torch.zeros_like(x)
        edge_attr = edge_attr[:edge_index.shape[1]]
        return self.conv(x, edge_index.long(), edge_attr)


class IncidenceLift(nn.Module):
    """Lift the ordered endpoints of each path to the next rank."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU())

    def forward(self, lower, link, higher_count):
        if higher_count == 0:
            return lower.new_empty((0, lower.shape[-1]))
        if link.shape[1] != higher_count:
            raise ValueError("Each higher-rank path must have one incidence column.")
        endpoints = torch.cat([lower[link[0].long()], lower[link[1].long()]], dim=-1)
        return self.proj(endpoints)


class PNNLayer(nn.Module):
    def __init__(self, hidden_dim, heads=4, dropout=0.2, direction="bidirectional"):
        super().__init__()
        if direction not in {"bidirectional", "upward"}:
            raise ValueError("PCNN direction must be 'bidirectional' or 'upward'.")
        self.direction = direction
        self.path1_to_path0 = GNNConv(hidden_dim, heads=heads, dropout=dropout) if direction == "bidirectional" else None
        self.path2_to_path1 = GNNConv(hidden_dim, heads=heads, dropout=dropout) if direction == "bidirectional" else None
        self.path0_to_path1 = IncidenceLift(hidden_dim)
        self.path1_to_path2 = IncidenceLift(hidden_dim)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(3)])
        self.drop = nn.Dropout(dropout)

    def forward(self, x0, x1, x2, path0_link, path1_link):
        if self.direction == "upward":
            up1 = self.path0_to_path1(x0, path0_link, x1.shape[0])
            y0 = self.norms[0](x0)
            y1 = self.norms[1](x1 + self.drop(up1))
            up2 = self.path1_to_path2(y1, path1_link, x2.shape[0])
            y2 = self.norms[2](x2 + self.drop(up2))
            return y0, y1, y2

        down0 = self.path1_to_path0(x0, path0_link, x1)
        down1 = self.path2_to_path1(x1, path1_link, x2)
        up1 = self.path0_to_path1(x0, path0_link, x1.shape[0])
        up2 = self.path1_to_path2(x1, path1_link, x2.shape[0])
        y0 = self.norms[0](x0 + self.drop(down0))
        y1 = self.norms[1](x1 + self.drop(down1 + up1))
        y2 = self.norms[2](x2 + self.drop(up2))
        return y0, y1, y2


class PathComplexBranch(nn.Module):
    """PCNN branch with rank-wise mean pooling."""

    def __init__(self, input_dim=195, hidden_dim=512, layers=3, heads=4, dropout=0.2, direction="bidirectional"):
        super().__init__()

        def projection():
            return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())

        self.path0_proj = projection()
        self.path1_proj = projection()
        self.path2_proj = projection()
        self.layers = nn.ModuleList([PNNLayer(hidden_dim, heads=heads, dropout=dropout, direction=direction) for _ in range(layers)])
        self.out = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

    def init_one(self, sample):
        return self.path0_proj(sample["path0"].float()), self.path1_proj(sample["path1"].float()), self.path2_proj(sample["path2"].float())

    def step_one(self, sample, x0, x1, x2, layer_idx):
        return self.layers[layer_idx](x0, x1, x2, sample["path0_link"].long(), sample["path1_link"].long())

    @staticmethod
    def pool_tokens(x0, x1, x2):
        if x0.shape[0] == 0:
            raise ValueError("path0 is empty; cannot pool path-complex representation.")
        x0_pool = x0.mean(dim=0)
        x1_pool = x1.mean(dim=0) if x1.shape[0] else torch.zeros_like(x0_pool)
        x2_pool = x2.mean(dim=0) if x2.shape[0] else torch.zeros_like(x0_pool)
        return torch.stack([x0_pool, x1_pool, x2_pool], dim=0)

    def embed_from_states(self, x0, x1, x2):
        return self.out(self.pool_tokens(x0, x1, x2).flatten())

    def forward_one(self, sample):
        x0, x1, x2 = self.init_one(sample)
        for layer_idx in range(len(self.layers)):
            x0, x1, x2 = self.step_one(sample, x0, x1, x2, layer_idx)
        return self.embed_from_states(x0, x1, x2)

    @staticmethod
    def _batch_link(link, offset):
        if link.numel() == 0:
            return link.reshape(2, 0).long()
        return link.long() + int(offset)

    def _batch_samples(self, samples):
        if not samples:
            raise ValueError("At least one path-complex sample is required.")

        states = [[], [], []]
        links = [[], []]
        batches = [[], [], []]
        offsets = [0, 0]

        for graph_idx, sample in enumerate(samples):
            x0, x1, x2 = self.init_one(sample)
            path0_link, path1_link = sample["path0_link"], sample["path1_link"]

            if path0_link.shape[1] != x1.shape[0]:
                raise ValueError(f"Sample {sample.get('sid', graph_idx)} has {path0_link.shape[1]} path0 edges but {x1.shape[0]} path1 features.")
            if path1_link.shape[1] != x2.shape[0]:
                raise ValueError(f"Sample {sample.get('sid', graph_idx)} has {path1_link.shape[1]} path1 edges but {x2.shape[0]} path2 features.")

            for rank, state in enumerate((x0, x1, x2)):
                states[rank].append(state)
                batches[rank].append(torch.full((state.shape[0],), graph_idx, device=state.device, dtype=torch.long))

            links[0].append(self._batch_link(path0_link, offsets[0]))
            links[1].append(self._batch_link(path1_link, offsets[1]))
            offsets[0] += x0.shape[0]
            offsets[1] += x1.shape[0]

        return (
            *(torch.cat(parts, dim=0) for parts in states),
            *(torch.cat(parts, dim=1) for parts in links),
            *(torch.cat(parts, dim=0) for parts in batches),
        )

    @staticmethod
    def pool_batched(x0, x1, x2, batch0, batch1, batch2, num_graphs):
        reference = global_mean_pool(x0, batch0, size=num_graphs)
        pooled = []
        for state, batch in ((x0, batch0), (x1, batch1), (x2, batch2)):
            pooled.append(global_mean_pool(state, batch, size=num_graphs) if state.shape[0] else reference.new_zeros(reference.shape))
        return pooled

    def embed_batched_states(self, x0, x1, x2, batch0, batch1, batch2, num_graphs):
        pooled = self.pool_batched(x0, x1, x2, batch0, batch1, batch2, num_graphs)
        return self.out(torch.cat(pooled, dim=-1))

    def _forward_single_scale(self, samples):
        x0, x1, x2, path0_link, path1_link, batch0, batch1, batch2 = self._batch_samples(samples)
        for layer in self.layers:
            x0, x1, x2 = layer(x0, x1, x2, path0_link, path1_link)
        return self.embed_batched_states(x0, x1, x2, batch0, batch1, batch2, num_graphs=len(samples))

    def forward(self, samples):
        return self._forward_single_scale(samples)


class DNATopoFusionModel(nn.Module):
    """Multiscale path-complex neural network for viral classification."""

    def __init__(self, num_classes, path_input_dim=768, hidden_dim=512, pcnn_layers=3, pcnn_heads=4, dropout=0.2, pcnn_direction="bidirectional"):
        super().__init__()
        self.path_branch = PathComplexBranch(input_dim=path_input_dim, hidden_dim=hidden_dim, layers=pcnn_layers, heads=pcnn_heads, dropout=dropout, direction=pcnn_direction)
        self.scale_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def embed(self, samples):
        if not samples:
            raise ValueError("At least one path-complex sample is required.")
        if any("scales" not in sample for sample in samples):
            raise ValueError("Every sample must contain EXP3_C multiscale features.")

        expected_counts = samples[0]["scale_counts"]
        if any(sample["scale_counts"] != expected_counts for sample in samples):
            raise ValueError("All samples must contain the same fragment-count scales.")

        scale_count = len(samples[0]["scales"])
        if any(len(sample["scales"]) != scale_count for sample in samples):
            raise ValueError("All samples must contain the same fragment-count scales.")

        scale_embeddings = torch.stack(
            [self.path_branch([sample["scales"][index] for sample in samples]) for index in range(scale_count)], dim=1
        )
        scale_weights = torch.softmax(self.scale_gate(scale_embeddings).squeeze(-1), dim=1)
        return (scale_embeddings * scale_weights.unsqueeze(-1)).sum(dim=1)

    def forward(self, samples):
        return self.classifier(self.embed(samples))
