"""Train and export the deployed LSTM storm classifier (3h horizon).

Mirrors notebooks/model-train/lstm.ipynb: an LSTM over a 16-bin (48h) window of
solar-wind + Ap history, trained with pos-weighted BCE on the same calendar split
as the deployed XGBoost model (train 2010-2021, test 2022-2024). The decision
threshold is tuned for F1 on a validation tail of train.

Writes two artifacts next to this script:
  - lstm_storm_3h_deployed.pt        (weights + preprocessing + config)
  - lstm_storm_3h_metadata.json      (display metadata, mirrors the XGBoost one)

Run from the repo root:  python3 time-series-modeling/train_lstm_deploy.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             precision_recall_curve, recall_score, roc_auc_score)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data" / "time_binned_dataset.csv"
PT_PATH = HERE / "lstm_storm_3h_deployed.pt"
META_PATH = HERE / "lstm_storm_3h_metadata.json"
PRED_PATH = HERE / "lstm_storm_3h_predictions.parquet"   # cached test-period P(storm) for the app

START_DATE = "2010-01-01"      # align to the deployed XGBoost training start
SPLIT_DATE = "2022-01-01"      # train < SPLIT_DATE, test >= SPLIT_DATE
SEQ_COLS = ['bz_gsm_nt_last', 'b_magnitude_avg_nt_last', 'flow_speed_kms_last',
            'proton_density_cm3_last', 'flow_pressure_npa_last',
            'electric_field_mvpm_last', 'bz_south_last', 'ap_now']
SEQ_LEN = 16
CLS_TARGET = "storm_3h"
HIDDEN, LAYERS, DROPOUT = 128, 2, 0.2
SEED = 0

device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')


class LSTMNet(nn.Module):
    def __init__(self, n_feat, hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=layers,
                            dropout=dropout if layers > 1 else 0.0, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


def build_sequences(df, split, mu=None, sd=None, train_means=None):
    """Standardised [N, SEQ_LEN, F] windows aligned so window i ends at df row i+SEQ_LEN-1."""
    vals_df = df[SEQ_COLS].ffill()
    if train_means is None:
        train_means = vals_df.iloc[:split].mean()
    vals = vals_df.fillna(train_means).to_numpy(dtype=np.float32)
    if mu is None or sd is None:
        mu, sd = vals[:split].mean(axis=0), vals[:split].std(axis=0) + 1e-8
    vals = (vals - mu) / sd
    seqs = np.lib.stride_tricks.sliding_window_view(vals, SEQ_LEN, axis=0).transpose(0, 2, 1)
    return seqs, mu, sd, train_means


def predict_logits(model, X, batch=8192):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.tensor(X[i:i+batch]).to(device)).cpu().numpy()
                               for i in range(0, len(X), batch)])


def best_f1_threshold(y_true, scores):
    p, r, t = precision_recall_curve(y_true, scores)
    f1 = 2 * p * r / np.clip(p + r, 1e-9, None)
    return float(t[np.argmax(f1[:-1])])


def main():
    df = pd.read_csv(DATA, parse_dates=["datetime"]).set_index("datetime").sort_index()
    df = df[df.index >= START_DATE]
    split = int((df.index >= SPLIT_DATE).argmax())
    vsplit = int(split * 0.85)

    seqs, mu, sd, train_means = build_sequences(df, split)
    off = SEQ_LEN - 1
    yc = df[CLS_TARGET].to_numpy(dtype=np.float32)[off:]
    tr_end, v_end = vsplit - off, split - off
    X_tr, X_val, X_te = seqs[:tr_end], seqs[tr_end:v_end], seqs[v_end:]
    yc_tr, yc_val, yc_te = yc[:tr_end], yc[tr_end:v_end], yc[v_end:]
    print(f"device {device} | train {len(X_tr)} | val {len(X_val)} | test {len(X_te)} "
          f"| storm rate test {yc_te.mean():.2%}")

    torch.manual_seed(SEED)
    model = LSTMNet(len(SEQ_COLS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt, yt = torch.tensor(X_tr), torch.tensor(yc_tr)
    pos_weight = torch.tensor((1 - yc_tr.mean()) / yc_tr.mean()).to(device)

    best, bad, best_state = -np.inf, 0, None
    for epoch in range(50):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 512):
            idx = perm[i:i+512]
            xb, yb = Xt[idx].to(device), yt[idx].to(device)
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(xb), yb, pos_weight=pos_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        val_ap = average_precision_score(yc_val, 1 / (1 + np.exp(-predict_logits(model, X_val))))
        if val_ap > best:
            best, bad = val_ap, 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 8:
                break
    model.load_state_dict(best_state)

    val_prob = 1 / (1 + np.exp(-predict_logits(model, X_val)))
    threshold = best_f1_threshold(yc_val, val_prob)
    test_prob = 1 / (1 + np.exp(-predict_logits(model, X_te)))
    y_hat = (test_prob >= threshold).astype(int)
    metrics = {
        "test_roc_auc": round(float(roc_auc_score(yc_te, test_prob)), 3),
        "test_pr_auc": round(float(average_precision_score(yc_te, test_prob)), 3),
        "storm_precision": round(float(precision_score(yc_te, y_hat, zero_division=0)), 3),
        "storm_recall": round(float(recall_score(yc_te, y_hat, zero_division=0)), 3),
        "storm_f1": round(float(f1_score(yc_te, y_hat, zero_division=0)), 3),
    }
    print("test metrics:", metrics, "| threshold", round(threshold, 3))

    torch.save({
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "seq_cols": SEQ_COLS, "seq_len": SEQ_LEN,
        "hidden": HIDDEN, "layers": LAYERS, "dropout": DROPOUT,
        "mu": mu.astype(np.float32), "sd": sd.astype(np.float32),
        "train_means": train_means.astype(np.float32).to_dict(),
        "operating_threshold": threshold,
    }, PT_PATH)

    metadata = {
        "model_name": "LSTM 3h Storm Classifier",
        "model_type": "LSTM (2x128) + pos-weighted BCE over 48h solar-wind windows",
        "target": CLS_TARGET, "horizon_hours": 3,
        "train_period": "2010-01-01 to 2021-12-31",
        "test_period": "2022-01-01 to 2024-12-30",
        "feature_set": f"{SEQ_LEN}-bin (48h) window of {len(SEQ_COLS)} solar-wind + Ap channels",
        "sequence_features": SEQ_COLS,
        "operating_threshold": round(threshold, 3),
        **metrics,
        "notes": "Sequence model; predicts from a rolling 48h history rather than a single bin. "
                 "Trained on the same calendar split as the deployed XGBoost for direct comparison.",
    }
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    # Cache the test-period predictions so the Streamlit app can serve the LSTM
    # without importing torch (keeps the dashboard process light and stable).
    test_index = df.index[split:]
    pd.DataFrame({"datetime": test_index, "storm_probability": test_prob.astype(np.float32)}) \
        .to_parquet(PRED_PATH, index=False)
    print("saved:", PT_PATH.name, ",", META_PATH.name, ",", PRED_PATH.name)


if __name__ == "__main__":
    main()
