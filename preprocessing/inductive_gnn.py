import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import HeteroData

from model_classes import (   # noqa: F401  (needed for pickle)
    HGTFraudDetector,
    OHEMLoss,
    PartialMinMaxScaler,
    BifurcatedFraudANNWrapper,
    ResidualFraudANN,
    EMBEDDING_DIM,
    HGT_HEADS,
    HGT_NUM_LAYERS,
)

NODE_TYPES = ["transaction", "user", "card", "merchant"]
EDGE_TYPES = [
    ("transaction", "by_user",    "user"),
    ("user",        "rev_by",     "transaction"),
    ("transaction", "uses_card",  "card"),
    ("card",        "rev_uses",   "transaction"),
    ("transaction", "at_merchant","merchant"),
    ("merchant",    "rev_at",     "transaction"),
    ("transaction", "similar_to", "transaction"),
    ("transaction", "virtual_to", "transaction"),
]

_ENTITY_COLS = {
    "user":     "user_id",
    "card":     "_card_proxy",
    "merchant": "merchant_category",
}
_REL_MAP = {
    "user":     ("by_user",    "rev_by"),
    "card":     ("uses_card",  "rev_uses"),
    "merchant": ("at_merchant","rev_at"),
}


def _load_hgt(artifacts_dir: str, hgt_in_channels: dict) -> HGTFraudDetector:
    model = HGTFraudDetector(
        in_channels_dict = hgt_in_channels,
        hidden_channels  = EMBEDDING_DIM,
        metadata         = (NODE_TYPES, EDGE_TYPES),
        num_heads        = HGT_HEADS,
        num_layers       = HGT_NUM_LAYERS,
        dropout          = 0.0,
    )
    state = torch.load(
        os.path.join(artifacts_dir, "hgt_encoder.pth"),
        map_location="cpu",
    )
    model.load_state_dict(state)
    model.eval()
    return model


def resolve_user_nodes(
    batch_user_ids: np.ndarray,
    entity_maps: dict,
    entity_feat_matrices: dict,
    X_new: np.ndarray,
) -> tuple[np.ndarray, dict, list[str]]:
    """
    Extend the frozen user entity matrix with temporary nodes for unknown users.

    Each unique unknown user_id in the batch gets one new node whose feature
    vector is the mean of that user's rows in X_new.  Multiple rows from the
    same unknown user are thus still connected through a shared node.

    Returns
    -------
    extended_user_mat   np.ndarray — frozen training rows + any new temp rows
    extended_emap_user  dict       — {user_id -> node_idx}, extended copy
    """
    base_emap     = entity_maps.get("user", {})
    base_user_mat = entity_feat_matrices.get("user",
                        np.zeros((0, X_new.shape[1]), dtype=np.float32))
    n_known = len(base_user_mat)

    extended_emap  = dict(base_emap)   # do NOT mutate the original
    extra_feats: list[np.ndarray] = []

    # Collect row indices per unknown user_id
    unknown_uid_rows: dict[int, list[int]] = {}
    for row_idx, uid in enumerate(batch_user_ids):
        uid_int = int(uid)
        if uid_int not in base_emap:
            unknown_uid_rows.setdefault(uid_int, []).append(row_idx)

    for uid, row_indices in unknown_uid_rows.items():
        new_node_idx = n_known + len(extra_feats)
        extended_emap[uid] = new_node_idx
        extra_feats.append(X_new[row_indices].mean(axis=0).astype(np.float32))

    if extra_feats:
        extended_user_mat = np.vstack(
            [base_user_mat, np.array(extra_feats, dtype=np.float32)]
        )
    else:
        extended_user_mat = base_user_mat

    return extended_user_mat, extended_emap


class HGTEmbedder:
    """
    Loads Stage-2 artifacts; exposes get_embeddings().

    Returns (X_emb_64d, user_mode_per_row).
    """

    def __init__(self, artifacts_dir: str = "fraud_detection_artifacts"):
        ad = artifacts_dir

        with open(os.path.join(ad, "entity_maps_hgt.pkl"), "rb") as f:
            self.entity_maps = pickle.load(f)
        with open(os.path.join(ad, "entity_feat_matrices.pkl"), "rb") as f:
            self.entity_feat_matrices = pickle.load(f)
        with open(os.path.join(ad, "feature_cols_gnn.pkl"), "rb") as f:
            self.feature_cols_gnn = pickle.load(f)
        with open(os.path.join(ad, "known_users.pkl"), "rb") as f:
            self.known_users = pickle.load(f)
        with open(os.path.join(ad, "hgt_in_channels.pkl"), "rb") as f:
            self.hgt_in_channels = pickle.load(f)

        self.hgt_model = _load_hgt(ad, self.hgt_in_channels)

        anchor_df = pd.read_parquet(os.path.join(ad, "train_anchor.parquet"))
        self.anchor_feats = anchor_df[self.feature_cols_gnn].values.astype(np.float32)
        self.anchor_entity_cols = anchor_df[
            [c for c in ["user_id", "_card_proxy", "merchant_category"]
             if c in anchor_df.columns]
        ]

        print(f"✅ HGTEmbedder ready — anchor={len(self.anchor_feats)}, "
              f"entity types={list(self.entity_feat_matrices.keys())}")

    def _build_graph(
        self,
        df_new_feat: pd.DataFrame,
        df_entity_new: pd.DataFrame,
        extended_user_mat: np.ndarray,
        extended_emap_user: dict,
    ) -> tuple:
        n_anchor = len(self.anchor_feats)
        X_new    = df_new_feat[self.feature_cols_gnn].values.astype(np.float32)
        X_all    = np.concatenate([self.anchor_feats, X_new], axis=0)

        data = HeteroData()
        data["transaction"].x         = torch.tensor(X_all, dtype=torch.float)
        data["transaction"].num_nodes = len(X_all)

        # Entity feature matrices — use extended user matrix for this batch
        for etype, mat in self.entity_feat_matrices.items():
            if etype == "user":
                data[etype].x         = torch.tensor(extended_user_mat, dtype=torch.float)
                data[etype].num_nodes = len(extended_user_mat)
            else:
                data[etype].x         = torch.tensor(mat, dtype=torch.float)
                data[etype].num_nodes = len(mat)

        # Entity id arrays (anchor first, then new)
        all_entity_ids = {}
        for col in ["user_id", "_card_proxy", "merchant_category"]:
            anchor_vals = (
                self.anchor_entity_cols[col].values
                if col in self.anchor_entity_cols.columns
                else np.array(["__unknown__"] * n_anchor)
            )
            new_vals = (
                df_entity_new[col].values
                if col in df_entity_new.columns
                else np.array(["__unknown__"] * len(df_new_feat))
            )
            all_entity_ids[col] = np.concatenate([anchor_vals, new_vals])

        # Build edges — use extended user map for "user" entity
        for etype, ecol in _ENTITY_COLS.items():
            emap = extended_emap_user if etype == "user" else self.entity_maps.get(etype, {})
            col_vals = all_entity_ids[ecol]
            src, dst = [], []
            for tx_idx, val in enumerate(col_vals):
                key = int(val) if etype == "user" else val
                if key in emap:
                    src.append(tx_idx)
                    dst.append(emap[key])
            ei       = torch.tensor([src, dst], dtype=torch.long)
            fwd, rev = _REL_MAP[etype]
            data["transaction", fwd, etype].edge_index = ei
            data[etype, rev, "transaction"].edge_index = ei.flip(0)

        data["transaction", "similar_to", "transaction"].edge_index = \
            torch.zeros((2, 0), dtype=torch.long)

        # Cold-start virtual edges
        new_user_ids = df_entity_new["user_id"].astype(str).values
        cold_mask    = ~np.isin(new_user_ids, np.array(list(self.known_users), dtype=str))
        cold_new_idx = np.where(cold_mask)[0] + n_anchor
        warm_idx     = np.arange(n_anchor)

        if len(cold_new_idx) > 0 and len(warm_idx) > 0:
            k = min(8, len(warm_idx))
            nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree", n_jobs=-1)
            nbrs.fit(self.anchor_feats)
            _, local_idx = nbrs.kneighbors(X_all[cold_new_idx])
            v_src = np.repeat(cold_new_idx, k)
            v_dst = warm_idx[local_idx.flatten()]
            data["transaction", "virtual_to", "transaction"].edge_index = \
                torch.tensor([v_src, v_dst], dtype=torch.long)
        else:
            data["transaction", "virtual_to", "transaction"].edge_index = \
                torch.zeros((2, 0), dtype=torch.long)

        return data, n_anchor

    def get_embeddings(
        self,
        df_feat_raw: pd.DataFrame,
        df_entity:   pd.DataFrame,
    ) -> np.ndarray:
        """Returns X_emb_64d (n_new, 64)."""
        X_new = df_feat_raw[self.feature_cols_gnn].values.astype(np.float32)

        batch_user_ids = df_entity["user_id"].values
        extended_user_mat, extended_emap_user = resolve_user_nodes(
            batch_user_ids,
            self.entity_maps,
            self.entity_feat_matrices,
            X_new,
        )

        data, n_anchor = self._build_graph(
            df_feat_raw, df_entity, extended_user_mat, extended_emap_user
        )

        self.hgt_model.eval()
        with torch.no_grad():
            emb_all = self.hgt_model.get_embedding(data.x_dict, data.edge_index_dict)

        emb_np    = emb_all.cpu().numpy()
        return emb_np[n_anchor:]
