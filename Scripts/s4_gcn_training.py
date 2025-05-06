import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import RandomLinkSplit
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, average_precision_score
from s3_build_graph import build_temporal_graph


# GCN encoder
class GCNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 512):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.dropout = 0.2

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


# Dot product decoder
class DotProductDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z: Tensor, edge_index: Tensor) -> Tensor:
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=1)


# MLP decoder
class MLPDecoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, z: Tensor, edge_index: Tensor) -> Tensor:
        src, dst = edge_index
        h = torch.cat([z[src], z[dst]], dim=1)
        return self.mlp(h).squeeze(-1)


# Train / Eval
def train_epoch(model, decoder, data, optimizer):
    model.train()
    if isinstance(decoder, nn.Module):
        decoder.train()

    optimizer.zero_grad()
    z = model(data.x, data.edge_index)

    if callable(decoder):
        pos_score = decoder(z, data.pos_edge_label_index)
        neg_score = decoder(z, data.neg_edge_label_index)
    else:
        pos_score = decoder(z, data.pos_edge_label_index)
        neg_score = decoder(z, data.neg_edge_label_index)

    score = torch.cat([pos_score, neg_score])
    label = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
    loss = F.binary_cross_entropy_with_logits(score, label)

    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def eval_auc_ap(model, decoder, data):
    model.eval()
    z = model(data.x, data.edge_index)

    if callable(decoder):
        pos_score = decoder(z, data.pos_edge_label_index).sigmoid()
        neg_score = decoder(z, data.neg_edge_label_index).sigmoid()
    else:
        decoder.eval()
        pos_score = decoder(z, data.pos_edge_label_index).sigmoid()
        neg_score = decoder(z, data.neg_edge_label_index).sigmoid()

    score = torch.cat([pos_score, neg_score]).cpu().numpy()
    label = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)]).cpu().numpy()
    return roc_auc_score(label, score), average_precision_score(label, score)


@torch.no_grad()
def extract_embeddings(model, data):
    model.eval()
    return model(data.x, data.edge_index).cpu().numpy()


# Main

def train(label: str, device, decoder_type="auto", epochs=200, data=data):
    splitter = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=True,
        split_labels=True
    )
    train_data, val_data, test_data = splitter(data)

    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)
    data = data.to(device)

    model = GCNEncoder(train_data.num_features).to(device)

    # Auto-decide if not specified
    if decoder_type == "auto":
        decoder_type = "dp" if label.lower() == "bow" else "mlp"

    if decoder_type.lower() == "dp":
        decoder = DotProductDecoder().to(device)
    elif decoder_type.lower() == "mlp":
        decoder = MLPDecoder(in_channels=512).to(device)
    else:
        raise ValueError(f"Unsupported decoder_type: {decoder_type}")

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(decoder.parameters()),
        lr=1e-3, weight_decay=1e-4
    )

    best_val_auc, trigger, patience = 0.0, 0, 10
    for epoch in range(epochs):
        loss = train_epoch(model, decoder, train_data, optimizer)
        val_auc, _ = eval_auc_ap(model, decoder, val_data)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            trigger = 0
            torch.save({"model": model.state_dict(), "decoder": decoder.state_dict()}, f"tmp_{label}_{decoder_type}.pt")
        else:
            trigger += 1
            if trigger >= patience:
                break

    ckpt = torch.load(f"tmp_{label}_{decoder_type}.pt")
    model.load_state_dict(ckpt["model"])
    decoder.load_state_dict(ckpt["decoder"])
    test_auc, test_ap = eval_auc_ap(model, decoder, test_data)
    print(f"[{label}-{decoder_type.upper()}] Test AUC={test_auc:.4f}, AP={test_ap:.4f}")

    return model, decoder, test_data


@torch.no_grad()
def plot_prediction_outcomes(
        z,
        decoder,
        pos_edge_index,
        neg_edge_index,
        train_edge_index,
        data,
        threshold=0.5,
        title="Prediction Outcomes"
):
    """
    Visualize prediction outcomes: TP, FN, FP on 2D projection (PCA or t-SNE),
    using data.is_cold (based on year > cutoff) to define cold-start nodes.
    """
    # Ensure tensor type
    if isinstance(z, np.ndarray):
        z_tensor = torch.from_numpy(z)
    else:
        z_tensor = z

    params = list(decoder.parameters())
    device = params[0].device if params else next(model.parameters()).device
    z_tensor = z_tensor.to(device)

    # Compute prediction scores
    pos_scores = decoder(z_tensor, pos_edge_index).sigmoid().cpu()
    neg_scores = decoder(z_tensor, neg_edge_index).sigmoid().cpu()

    # Classify predictions
    tp_edges = pos_edge_index[:, pos_scores >= threshold]
    fn_edges = pos_edge_index[:, pos_scores < threshold]
    fp_edges = neg_edge_index[:, neg_scores >= threshold]

    # Project embeddings
    z_np = z_tensor.cpu().numpy()
    reducer = TSNE(n_components=2, random_state=42)
    z_2d = reducer.fit_transform(z_np)

    # Cold-start mask from year-based cutoff
    mask_new = data.is_cold.cpu().numpy()

    # Visualization
    plt.figure(figsize=(8, 8))

    plt.scatter(
        z_2d[~mask_new, 0], z_2d[~mask_new, 1],
        s=6, alpha=.35, color="lightgray", label="Old (seen)"
    )
    plt.scatter(
        z_2d[mask_new, 0], z_2d[mask_new, 1],
        facecolors="none", edgecolors="blue",
        linewidths=.8, s=22, label="New (cold-start)"
    )

    # Draw edges
    for edges, color, label in [
        (fn_edges, "crimson", "False Negative (FN)"),
        (fp_edges, "dodgerblue", "False Positive (FP)"),
        (tp_edges, "forestgreen", "True Positive (TP)")
    ]:
        for i, j in zip(edges[0], edges[1]):
            x1, y1 = z_2d[i]
            x2, y2 = z_2d[j]
            plt.plot([x1, x2], [y1, y2], color=color, linewidth=0.8, alpha=0.6)

    plt.title(title + "\nTP=Green, FN=Red, FP=Blue")
    plt.axis("off")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_f1_vs_threshold(model, decoder, data, label="Unknown", split="val"):
    model.eval()
    decoder.eval()

    z = model(data.x, data.edge_index)
    pos_score = decoder(z, data.pos_edge_label_index).sigmoid()
    neg_score = decoder(z, data.neg_edge_label_index).sigmoid()
    y_true = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)]).cpu().numpy()
    y_score = torch.cat([pos_score, neg_score]).detach().cpu().numpy()

    thresholds = np.linspace(0, 1, 200)
    f1s = []
    best_f1, best_thresh = 0, 0

    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        f1s.append(f1)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, f1s, label=f"{label} F1-Score")
    plt.axvline(best_thresh, color='r', linestyle='--', label=f"Best Thresh = {best_thresh:.2f}")
    plt.title(f"F1 vs Threshold ({label})")
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"Best F1 for [{label}] = {best_f1:.4f} at threshold = {best_thresh:.3f}")

    return best_thresh


def pipeline(label: str, device, decoder_type="auto", epochs=200, data_path="openalex_9000.json", plot=True):
    if label == "BoW":
        embed_path = "bow_embeddings_9000.npy"
    elif label == "SciBERT":
        embed_path = "sci_bert_embeddings_9000.npy"
    data = build_temporal_graph(data_path, embed_path, cutoff=2018)
    model, decoder, test_data = train(label=label, device=device, decoder_type=decoder_type, epochs=epochs, data=data)

    if plot:
        with torch.no_grad():
            z = extract_embeddings(model, data)
            best_thresh = plot_f1_vs_threshold(model, decoder, test_data, label=f"{label}-{decoder_type}")
            plot_prediction_outcomes(
                z=z,
                decoder=decoder,
                pos_edge_index=data.pos_edge_label_index,
                neg_edge_index=data.neg_edge_label_index,
                train_edge_index=data.edge_index,
                data=data,
                threshold=best_thresh,
                title=f"{label}-{decoder_type.upper()} Prediction Outcome on Test Set"
            )

    return model, decoder


device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")

pipeline(label="BoW", decoder_type="dp", device=device, plot=False)
pipeline(label="SciBERT", decoder_type="mlp", device=device, plot=False)
