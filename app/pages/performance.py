"""Model performance — head-to-head on the identical held-out test period."""
from __future__ import annotations

import pandas as pd
import streamlit as st
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app import charts, ui
from app.data import (
    MODEL_NAMES,
    SPLIT_DATE,
    TARGET,
    load_data,
    operating_threshold,
    score_all,
)
from app.theme import tokens


def render() -> None:
    t = tokens()
    df = load_data()
    scored = score_all()
    y_true = scored[TARGET].to_numpy()
    proba = {name: scored[f"proba_{name}"].to_numpy() for name in MODEL_NAMES}
    tuned = {name: operating_threshold(name) for name in MODEL_NAMES}

    ui.page_header(
        "Model performance",
        "All three models scored on the <b>same</b> held-out test period "
        f"({SPLIT_DATE} to {df['datetime'].max():%Y-%m-%d}), so every number below is "
        "directly comparable. The sliders start at each model's tuned operating point.",
    )

    thresholds = _threshold_bar(y_true, proba, tuned)
    preds = {name: (proba[name] >= thresholds[name]).astype(int) for name in MODEL_NAMES}

    _metrics_table(y_true, proba, preds, thresholds)
    _confusion(y_true, preds, thresholds, t)
    _pr_curves(y_true, proba, preds, tuned, t)
    _narrative(y_true, preds)
    ui.footer()


# --------------------------------------------------------- threshold bar --
def _threshold_bar(y_true, proba: dict, tuned: dict) -> dict:
    ui.section(
        "Decision thresholds",
        "A model outputs a probability; the threshold turns it into a call. Lower it to "
        "catch more storms and accept more false alarms, raise it for the reverse.",
    )

    thresholds = {}
    with st.container(border=True):
        cols = st.columns(len(MODEL_NAMES), gap="medium")
        for col, name in zip(cols, MODEL_NAMES):
            with col:
                thresholds[name] = st.slider(
                    f"{name} threshold",
                    min_value=0.05, max_value=0.99, value=tuned[name], step=0.01,
                    key=f"thr_{name}",
                )
                tuned_f1 = f1_score(
                    y_true, (proba[name] >= tuned[name]).astype(int), zero_division=0
                )
                st.caption(
                    f"Tuned optimum **{tuned[name]:.2f}** → F1 **{tuned_f1:.2f}** "
                    "(maximised on validation)"
                )
    return thresholds


# -------------------------------------------------------- metrics table --
def _metrics_table(y_true, proba: dict, preds: dict, thresholds: dict) -> None:
    ui.section("Head-to-head")

    rows = [
        {
            "Model": name,
            "Threshold": thresholds[name],
            "Precision": precision_score(y_true, preds[name], zero_division=0),
            "Recall": recall_score(y_true, preds[name], zero_division=0),
            "F1": f1_score(y_true, preds[name], zero_division=0),
            "ROC AUC": roc_auc_score(y_true, proba[name]),
            "PR AUC": average_precision_score(y_true, proba[name]),
        }
        for name in MODEL_NAMES
    ]

    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            "Model": st.column_config.TextColumn(width="small"),
            "Threshold": st.column_config.NumberColumn(format="%.2f", width="small"),
            "Precision": st.column_config.ProgressColumn(
                format="%.2f", min_value=0.0, max_value=1.0,
                help="Of the storms this model called, how many really were storms.",
            ),
            "Recall": st.column_config.ProgressColumn(
                format="%.2f", min_value=0.0, max_value=1.0,
                help="Of the real storms, how many this model caught.",
            ),
            "F1": st.column_config.ProgressColumn(
                format="%.2f", min_value=0.0, max_value=1.0,
                help="The balance between precision and recall.",
            ),
            "ROC AUC": st.column_config.NumberColumn(format="%.3f", width="small"),
            "PR AUC": st.column_config.NumberColumn(format="%.3f", width="small"),
        },
    )

    with st.expander("How to read these"):
        st.markdown(
            "- **Precision** — of the storms a model called, how many really were storms. "
            "Low precision means crying wolf.\n"
            "- **Recall** — of the real storms, how many the model caught. Low recall "
            "means storms arrive unannounced.\n"
            "- **F1** — the harmonic mean of the two; a single number for the balance.\n"
            "- **ROC AUC / PR AUC** — threshold-independent rankings, so they don't move "
            "with the sliders. With storms at under 2% of bins, **PR AUC is the honest "
            "one** — ROC AUC flatters every model on data this imbalanced."
        )


# ------------------------------------------------------------- confusion --
def _confusion(y_true, preds: dict, thresholds: dict, t: dict) -> None:
    ui.section(
        "Where each model gets it wrong",
        "Outcomes at the selected thresholds. The diagonal is correct calls; top-right is "
        "a false alarm, bottom-left is a missed storm.",
    )
    cols = st.columns(len(MODEL_NAMES), gap="medium")
    for col, name in zip(cols, MODEL_NAMES):
        cm = confusion_matrix(y_true, preds[name])
        with col, st.container(border=True):
            st.markdown(
                f'<div style="font-size:.85rem;font-weight:600;color:var(--sp-ink);'
                f'margin-bottom:.2rem">{name}'
                f'<span style="color:var(--sp-ink-muted);font-weight:400">'
                f" · threshold {thresholds[name]:.2f}</span></div>",
                unsafe_allow_html=True,
            )
            st.altair_chart(charts.confusion_matrix(cm, t), theme=None)


# ------------------------------------------------------------ PR curves --
def _pr_curves(y_true, proba: dict, preds: dict, tuned: dict, t: dict) -> None:
    ui.section(
        "Precision–recall trade-off",
        "Curves further toward the top-right catch more storms with fewer false alarms. "
        "The ring marks each model's tuned threshold; the filled dot moves with your "
        "slider — they start on top of each other.",
    )

    curve_frames, points = [], []
    for name in MODEL_NAMES:
        prec, rec, _ = precision_recall_curve(y_true, proba[name])
        ap = average_precision_score(y_true, proba[name])
        curve_frames.append(
            pd.DataFrame({"model": f"{name} (AP {ap:.2f})", "recall": rec, "precision": prec})
        )
        tuned_pred = (proba[name] >= tuned[name]).astype(int)
        points.append(
            {
                "model": f"{name} (AP {ap:.2f})", "kind": "tuned",
                "label": f"Tuned threshold {tuned[name]:.2f}",
                "recall": recall_score(y_true, tuned_pred),
                "precision": precision_score(y_true, tuned_pred, zero_division=0),
            }
        )
        points.append(
            {
                "model": f"{name} (AP {ap:.2f})", "kind": "selected",
                "label": "Your selected threshold",
                "recall": recall_score(y_true, preds[name]),
                "precision": precision_score(y_true, preds[name], zero_division=0),
            }
        )

    with st.container(border=True):
        st.altair_chart(
            charts.pr_curves(pd.concat(curve_frames), pd.DataFrame(points), t), theme=None
        )


# ------------------------------------------------------------ narrative --
def _narrative(y_true, preds: dict) -> None:
    ui.section(
        "At the selected thresholds",
        "The same trade-off in plain counts, over the whole test period.",
    )
    tiles = []
    for name in MODEL_NAMES:
        _tn, fp, fn, tp = confusion_matrix(y_true, preds[name]).ravel()
        tiles.append((name, f"{tp} / {tp + fn}", f"storms caught · {fp:,} false alarms"))
    ui.stat_row(tiles)
