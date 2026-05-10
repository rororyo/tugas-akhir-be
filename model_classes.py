import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from torch_geometric.nn import HGTConv

# ── Constants (mirror model_final.py) ─────────────────────────────────────────
EMBEDDING_DIM        = 64
HGT_HEADS            = 4
HGT_NUM_LAYERS       = 2
GNN_DROPOUT          = 0.0
GNN_OHEM_RATIO       = 0.7
GNN_POS_WEIGHT_CAP   = 10.0
RANDOM_STATE         = 42

SC3_ANN_HIDDEN_TAB   = 128
SC3_ANN_EMB_HIDDEN   = 64
SC3_ANN_HEAD_WIDTH   = 64
SC3_ANN_DROPOUT      = 0.3
SC3_ANN_LR           = 0.001
SC3_ANN_EPOCHS       = 100
SC3_ANN_PATIENCE     = 10
SC3_ANN_BATCH        = 256


# ── HGTFraudDetector ──────────────────────────────────────────────────────────

class HGTFraudDetector(nn.Module):
    def __init__(self, in_channels_dict, hidden_channels, out_channels=2,
                 metadata=None, num_heads=HGT_HEADS, num_layers=HGT_NUM_LAYERS,
                 dropout=GNN_DROPOUT):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers      = num_layers
        self.dropout         = dropout

        self.lin_dict = nn.ModuleDict({
            nt: nn.Linear(in_ch, hidden_channels)
            for nt, in_ch in in_channels_dict.items()
        })
        self.convs = nn.ModuleList([
            HGTConv(hidden_channels, hidden_channels, metadata, heads=num_heads)
            for _ in range(num_layers)
        ])
        self.bns = nn.ModuleList([
            nn.ModuleDict({nt: nn.BatchNorm1d(hidden_channels)
                           for nt in in_channels_dict})
            for _ in range(num_layers)
        ])
        self.jk_lin     = nn.Linear(hidden_channels * num_layers, hidden_channels)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, out_channels),
        )

    def forward(self, x_dict, edge_index_dict, return_embedding=False, return_dict=False):
        h_dict = {
            nt: F.relu(self.lin_dict[nt](x))
            for nt, x in x_dict.items()
            if nt in self.lin_dict
        }
        tx_layer_outs = []
        for conv, bn_dict in zip(self.convs, self.bns):
            h_prev     = h_dict
            h_dict_new = conv(h_dict, edge_index_dict)
            h_dict  = {}
            for nt, h in h_dict_new.items():
                h = bn_dict[nt](h)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                if nt in h_prev:
                    h = h + h_prev[nt]
                h_dict[nt] = h
            tx_layer_outs.append(h_dict["transaction"])

        h_tx = torch.cat(tx_layer_outs, dim=-1)
        h_tx = F.relu(self.jk_lin(h_tx))
        h_tx = F.dropout(h_tx, p=self.dropout, training=self.training)

        if return_embedding:
            return h_tx

        logits = self.classifier(h_tx)

        if return_dict:
            return logits, h_dict

        return logits

    def get_embedding(self, x_dict, edge_index_dict):
        return self.forward(x_dict, edge_index_dict, return_embedding=True)


# ── OHEMLoss ──────────────────────────────────────────────────────────────────

class OHEMLoss(nn.Module):
    def __init__(self, ratio: float = GNN_OHEM_RATIO,
                 pos_weight: "torch.Tensor | None" = None):
        super().__init__()
        self.ratio      = ratio
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        per_sample = F.cross_entropy(
            logits, targets, weight=self.pos_weight, reduction="none")
        k = max(1, int(self.ratio * len(per_sample)))
        hard_losses, _ = torch.topk(per_sample, k)
        return hard_losses.mean()


# ── PartialMinMaxScaler ───────────────────────────────────────────────────────

class PartialMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, n_tab_cols=None):
        self.n_tab_cols = n_tab_cols
    def fit(self, X, y=None):
        X = np.asarray(X)
        n = self.n_tab_cols if self.n_tab_cols is not None else X.shape[1]
        self._scaler = MinMaxScaler()
        self._scaler.fit(X[:, :n])
        self._n_fitted = n
        return self
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        X_out = X.copy()
        X_out[:, :self._n_fitted] = self._scaler.transform(X[:, :self._n_fitted])
        return X_out
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


# ── ResidualFraudANN ──────────────────────────────────────────────────────────

class ResidualFraudANN(nn.Module):
    def __init__(self,
                 n_tab:      int,
                 emb_dim:    int   = EMBEDDING_DIM,
                 hidden_tab: int   = SC3_ANN_HIDDEN_TAB,
                 emb_hidden: int   = SC3_ANN_EMB_HIDDEN,
                 head_width: int   = SC3_ANN_HEAD_WIDTH,
                 dropout:    float = SC3_ANN_DROPOUT):
        super().__init__()
        self.n_tab   = n_tab
        self.emb_dim = emb_dim
        self.has_emb = emb_dim > 0
        half = head_width // 2

        self.tab_fc1   = nn.Linear(n_tab,     hidden_tab)
        self.tab_bn1   = nn.BatchNorm1d(hidden_tab)
        self.tab_fc2   = nn.Linear(hidden_tab, head_width)
        self.tab_bn2   = nn.BatchNorm1d(head_width)
        self.tab_drop  = nn.Dropout(dropout)
        self.tab_skip  = nn.Linear(n_tab, head_width, bias=False)

        if self.has_emb:
            self.emb_fc1 = nn.Linear(emb_dim, emb_hidden)
            self.emb_bn1 = nn.BatchNorm1d(emb_hidden)
            self.emb_drop1 = nn.Dropout(dropout)
            self.emb_fc2 = nn.Linear(emb_hidden, half)
            self.emb_bn2 = nn.BatchNorm1d(half)
            self.emb_drop2 = nn.Dropout(dropout)
            head_in = head_width + half
        else:
            head_in = head_width

        self.head = nn.Sequential(
            nn.Linear(head_in, head_width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_width, 2),
        )

    def forward(self, x_tab: torch.Tensor, x_emb: torch.Tensor) -> torch.Tensor:
        h         = self.tab_drop(F.relu(self.tab_bn1(self.tab_fc1(x_tab))))
        h         = self.tab_bn2(self.tab_fc2(h))
        h_tab_res = F.relu(h + self.tab_skip(x_tab))

        if self.has_emb:
            h_emb = F.relu(self.emb_bn1(self.emb_fc1(x_emb)))
            h_emb = self.emb_drop1(h_emb)
            h_emb = F.relu(self.emb_bn2(self.emb_fc2(h_emb)))
            h_emb = self.emb_drop2(h_emb)
            combined = torch.cat([h_tab_res, h_emb], dim=1)
        else:
            combined = h_tab_res

        return self.head(combined)


# ── BifurcatedFraudANNWrapper ─────────────────────────────────────────────────

class BifurcatedFraudANNWrapper:
    def __init__(self,
                 n_tab:      int,
                 emb_dim:    int   = EMBEDDING_DIM,
                 hidden_tab: int   = SC3_ANN_HIDDEN_TAB,
                 emb_hidden: int   = SC3_ANN_EMB_HIDDEN,
                 head_width: int   = SC3_ANN_HEAD_WIDTH,
                 dropout:    float = SC3_ANN_DROPOUT,
                 lr:         float = SC3_ANN_LR,
                 epochs:     int   = SC3_ANN_EPOCHS,
                 patience:   int   = SC3_ANN_PATIENCE,
                 batch_size: int   = SC3_ANN_BATCH):
        self.n_tab      = n_tab
        self.emb_dim    = emb_dim
        self.hidden_tab = hidden_tab
        self.emb_hidden = emb_hidden
        self.head_width = head_width
        self.dropout    = dropout
        self.lr         = lr
        self.epochs     = epochs
        self.patience   = patience
        self.batch_size = batch_size

        self.model_      = None
        self.emb_scaler_ = None
        self.device_     = torch.device("cpu")

    def fit(self, X_tab_sc: np.ndarray, X_emb_raw: np.ndarray, y: np.ndarray) -> "BifurcatedFraudANNWrapper":
        if self.emb_dim > 0:
            self.emb_scaler_ = RobustScaler()
            X_emb_sc = self.emb_scaler_.fit_transform(X_emb_raw).astype(np.float32)
        else:
            self.emb_scaler_ = None
            X_emb_sc = np.zeros((len(X_tab_sc), 0), dtype=np.float32)

        X_tab_sc = X_tab_sc.astype(np.float32)
        y        = y.astype(np.int64)

        from sklearn.model_selection import train_test_split
        idx = np.arange(len(y))
        tr_idx, va_idx = train_test_split(
            idx, test_size=0.10, stratify=y, random_state=RANDOM_STATE)

        t_tab_tr = torch.tensor(X_tab_sc[tr_idx], dtype=torch.float32)
        t_emb_tr = torch.tensor(X_emb_sc[tr_idx], dtype=torch.float32)
        t_y_tr   = torch.tensor(y[tr_idx],         dtype=torch.long)

        t_tab_va = torch.tensor(X_tab_sc[va_idx], dtype=torch.float32)
        t_emb_va = torch.tensor(X_emb_sc[va_idx], dtype=torch.float32)
        t_y_va   = torch.tensor(y[va_idx],         dtype=torch.long)

        n_legit  = int((y[tr_idx] == 0).sum())
        n_fraud  = int((y[tr_idx] == 1).sum())
        pos_w    = min(n_legit / max(n_fraud, 1), GNN_POS_WEIGHT_CAP)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, pos_w], dtype=torch.float32))

        self.model_ = ResidualFraudANN(
            n_tab      = self.n_tab,
            emb_dim    = self.emb_dim,
            hidden_tab = self.hidden_tab,
            emb_hidden = self.emb_hidden,
            head_width = self.head_width,
            dropout    = self.dropout,
        ).to(self.device_)

        optimizer  = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, verbose=False)

        best_val   = float("inf")
        best_state = None
        no_improve = 0

        tr_dataset = torch.utils.data.TensorDataset(t_tab_tr, t_emb_tr, t_y_tr)
        tr_loader  = torch.utils.data.DataLoader(tr_dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            for b_tab, b_emb, b_y in tr_loader:
                optimizer.zero_grad()
                logits = self.model_(b_tab, b_emb)
                loss   = criterion(logits, b_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                val_logits = self.model_(t_tab_va, t_emb_va)
                val_loss   = criterion(val_logits, t_y_va).item()

            scheduler.step(val_loss)

            if val_loss < best_val - 1e-5:
                best_val   = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.patience:
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_np  = np.asarray(X, dtype=np.float32)
        X_tab = X_np[:, :self.n_tab]
        if self.emb_dim > 0:
            X_emb = self.emb_scaler_.transform(X_np[:, self.n_tab:])
        else:
            X_emb = np.zeros((len(X_tab), 0), dtype=np.float32)
        t_tab = torch.tensor(X_tab, dtype=torch.float32)
        t_emb = torch.tensor(X_emb, dtype=torch.float32)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(t_tab, t_emb)
            probs  = torch.softmax(logits, dim=1).numpy()
        return probs
