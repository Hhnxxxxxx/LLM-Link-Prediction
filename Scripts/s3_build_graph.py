from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

import warnings

# Warning filter
warnings.filterwarnings(
    "ignore",
    message=r".*Deterministic behavior was enabled with either.*CuBLAS.*"
)
warnings.filterwarnings(
    "ignore",
    message=r"'train_test_split_edges' is deprecated.*"
)


# Utility helpers

def build_graph(json_path: str | Path, npy_path: str | Path) -> Data:
    """Load papers + embeddings and return a PyG `Data` object.

    Edge direction:  paper *i*  →  paper *j*  if *i* **cites** *j*.  We treat
    the graph as *undirected* for GCN training.
    """
    with Path(json_path).open("r", encoding="utf-8") as f:
        papers = json.load(f)

    x = torch.from_numpy(np.load(npy_path)).float()
    n_nodes = len(papers)
    assert x.size(0) == n_nodes, "Embeddings and papers count mismatch"

    # Build id to index map (OpenAlex IDs are the tail str after last slash)
    id2idx = {paper["id"].split("/")[-1]: idx for idx, paper in enumerate(papers)}

    edge_list: list[list[int]] = []
    for src_idx, paper in enumerate(papers):
        for ref in paper.get("references", []):
            ref_key = ref.split("/")[-1]
            tgt_idx = id2idx.get(ref_key)
            if tgt_idx is not None:
                edge_list.append([src_idx, tgt_idx])
                edge_list.append([tgt_idx, src_idx])  # undirected

    if not edge_list:
        raise RuntimeError("No edges were constructed – check reference parsing.")

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()  # [2,E]

    return Data(x=x, edge_index=edge_index)


def build_temporal_graph(
        json_path: str | Path,
        npy_path: str | Path,
        cutoff: int = 2018
) -> Data:
    with Path(json_path).open("r", encoding="utf-8") as f:
        papers = json.load(f)

    x = torch.from_numpy(np.load(npy_path)).float()
    n_nodes = len(papers)
    assert x.size(0) == n_nodes, "Embeddings and papers count mismatch"

    id2idx = {paper["id"].split("/")[-1]: idx for idx, paper in enumerate(papers)}
    year_list = []
    train_edges = []
    test_edges = []

    for src_idx, paper in enumerate(papers):
        src_year = paper.get("publication_year")
        year_list.append(src_year if src_year is not None else -1)

        for ref in paper.get("references", []):
            ref_key = ref.split("/")[-1]
            tgt_idx = id2idx.get(ref_key)
            if tgt_idx is not None:
                tgt_year = papers[tgt_idx].get("publication_year")
                if src_year is None or tgt_year is None:
                    continue
                if src_year <= cutoff and tgt_year <= cutoff:
                    train_edges.append([src_idx, tgt_idx])
                    train_edges.append([tgt_idx, src_idx])
                elif src_year > cutoff or tgt_year > cutoff:
                    test_edges.append([src_idx, tgt_idx])
                    test_edges.append([tgt_idx, src_idx])

    if not train_edges:
        raise RuntimeError("No training edges constructed.")
    if not test_edges:
        raise RuntimeError("No testing edges constructed.")

    train_edge_index = torch.tensor(train_edges, dtype=torch.long).t().contiguous()
    test_pos_edge_index = torch.tensor(test_edges, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=train_edge_index)
    data.pos_edge_label_index = test_pos_edge_index
    data.neg_edge_label_index = negative_sampling(
        edge_index=train_edge_index,
        num_nodes=n_nodes,
        num_neg_samples=test_pos_edge_index.size(1)
    )

    # Year info
    year_tensor = torch.tensor(year_list)
    data.year = year_tensor
    data.cutoff = cutoff
    data.is_cold = (year_tensor > cutoff)

    return data
