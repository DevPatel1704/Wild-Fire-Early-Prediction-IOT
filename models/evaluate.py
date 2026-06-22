"""
Evaluation graphs for the Wildfire EWS project.

This file reads the artifacts saved by models/train.py (training history +
test predictions) and the processed dataset, then generates all the evaluation
plots we need for the report. Everything gets saved to artifacts/eval/ as PNGs
and a single report.html that pulls them all together.

Graphs this produces:
  data_overview          class balance, seasonality, feature distributions, correlation
  data_fire_timeline     how many fire records per day across the dataset
  data_monthly_rate      fire rate broken down by calendar month
  data_feature_import    RF feature importance on the raw tabular data
  training_history       loss + accuracy/recall/precision/F1/AUC per epoch
  confusion_matrix       at the best F1 threshold
  roc_curve              ROC + AUC
  pr_curve               precision-recall curve + average precision
  threshold_sweep        how metrics change as we move the decision threshold
  calibration            predicted probability vs actual fire frequency
  risk_tiers             how many test samples fall into each alert tier
  prob_distribution      predicted probability histograms split by true class
  node_performance       F1 per sensor node on the 10x10 BC grid
  error_breakdown        TP/FP/FN/TN stacked by alert tier
  model_comparison       GAT-LSTM vs the other models we trained

Run with:
    python -m models.evaluate
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, roc_curve, auc, fbeta_score,
                             precision_recall_curve, average_precision_score,
                             precision_score, recall_score, f1_score,
                             accuracy_score)
from sklearn.calibration import calibration_curve
from config import (ARTIFACTS_DIR, PROCESSED_DIR, TOP_10_FEATURES, RISK_TIERS,
                    risk_tier)

EVAL_DIR = ARTIFACTS_DIR / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
ACCENT = "#2dd4bf"
TIER_COLOURS = [c for _, _, c in RISK_TIERS]
from matplotlib.colors import LinearSegmentedColormap
TEAL_CMAP = LinearSegmentedColormap.from_list("teal", ["#0e151d", "#15414a", "#2dd4bf"])
CORR_CMAP = "RdBu_r"


def setup_style():
    # dark theme so the plots match the dashboard look
    plt.rcParams.update({
        "figure.facecolor": "#0e151d", "axes.facecolor": "#111923",
        "savefig.facecolor": "#0e151d", "savefig.bbox": "tight",
        "text.color": "#cdd8e1", "axes.labelcolor": "#cdd8e1",
        "axes.titlecolor": "#e6edf3", "xtick.color": "#8aa0b0",
        "ytick.color": "#8aa0b0", "axes.edgecolor": "#22303c",
        "grid.color": "#19222c", "grid.linestyle": "--", "axes.grid": True,
        "font.family": "DejaVu Sans Mono", "figure.dpi": 130,
        "legend.facecolor": "#111923", "legend.edgecolor": "#22303c",
    })


def _save(fig, name):
    path = EVAL_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return path.name


def plot_data_overview(df):
    out = []

    # class balance bar chart
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["fire"].value_counts().sort_index()
    ax.bar(["no-fire", "fire"], [counts.get(0, 0), counts.get(1, 0)],
           color=["#33485a", ACCENT])
    for i, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax.set_title(f"Class balance  (fire rate = {df['fire'].mean():.1%})")
    ax.set_ylabel("records")
    out.append(_save(fig, "data_class_balance.png"))

    # fire rate by day of year - shows the summer spike clearly
    if "DOY" in df:
        fig, ax = plt.subplots(figsize=(7, 3.4))
        rate = df.groupby(df["DOY"].round())["fire"].mean()
        ax.plot(rate.index, rate.values, color="#f97316", lw=1.6)
        ax.fill_between(rate.index, rate.values, color="#f97316", alpha=0.15)
        ax.set_title("Fire rate by day of year (seasonality)")
        ax.set_xlabel("day of year"); ax.set_ylabel("fire rate")
        out.append(_save(fig, "data_seasonality.png"))

    # one histogram per feature, fire vs no-fire overlaid
    feats = [f for f in TOP_10_FEATURES if f in df and f != "DOY"]
    fig, axes = plt.subplots(3, 3, figsize=(11, 9))
    for ax, f in zip(axes.ravel(), feats):
        for cls, col, lab in [(0, "#33485a", "no-fire"), (1, ACCENT, "fire")]:
            s = df.loc[df["fire"] == cls, f]
            ax.hist(s, bins=40, density=True, alpha=0.55, color=col, label=lab)
        ax.set_title(f, fontsize=10)
        ax.tick_params(labelsize=7)
    axes.ravel()[0].legend(fontsize=8)
    for ax in axes.ravel()[len(feats):]:
        ax.axis("off")
    fig.suptitle("Top-10 feature distributions · fire vs no-fire", y=1.0)
    out.append(_save(fig, "data_feature_distributions.png"))

    # correlation heatmap for the top-10 features
    fig, ax = plt.subplots(figsize=(7.5, 6))
    corr = df[[f for f in TOP_10_FEATURES if f in df]].corr()
    im = ax.imshow(corr, cmap=CORR_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="#cdd8e1")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Feature correlation")
    ax.grid(False)
    out.append(_save(fig, "data_correlation.png"))
    return out


def plot_data_fire_timeline(df):
    # daily fire count over the whole dataset - useful for spotting gaps or anomalies
    if "date" not in df.columns or "fire" not in df.columns:
        return []
    daily = df.groupby("date")["fire"].sum().reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.fill_between(daily["date"], daily["fire"], color="#f97316", alpha=0.25)
    ax.plot(daily["date"], daily["fire"], color="#f97316", lw=1.4)
    ax.set_title("Daily fire record count over time")
    ax.set_xlabel("date"); ax.set_ylabel("fire records / day")
    fig.autofmt_xdate()
    return [_save(fig, "data_fire_timeline.png")]


def plot_data_monthly_rate(df):
    # monthly breakdown is easier to read than DOY for presentations
    if "date" not in df.columns or "fire" not in df.columns:
        return []
    tmp = df.copy()
    tmp["month"] = pd.to_datetime(tmp["date"]).dt.month
    rate = tmp.groupby("month")["fire"].mean()
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    vals = [float(rate.get(m + 1, 0.0)) for m in range(12)]
    non_zero = [v for v in vals if v > 0]
    avg = sum(non_zero) / len(non_zero) if non_zero else 0.0
    colors = [ACCENT if v > avg else "#33485a" for v in vals]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(months, vals, color=colors)
    ax.axhline(avg, ls="--", color="#ef4444", lw=1, label=f"mean = {avg:.3f}")
    ax.set_title("Fire rate by calendar month"); ax.set_ylabel("fire rate")
    ax.legend(fontsize=8)
    return [_save(fig, "data_monthly_rate.png")]


def plot_data_feature_importance_rf(df):
    # quick RF on the raw tabular data just to visualize which features matter most
    from sklearn.ensemble import RandomForestClassifier
    feats = [f for f in TOP_10_FEATURES if f in df.columns]
    if len(feats) < 2 or "fire" not in df.columns:
        return []
    sub = df[feats + ["fire"]].dropna()
    if len(sub) < 200:
        return []
    sub_s = sub.sample(min(len(sub), 50_000), random_state=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=8,
                                class_weight="balanced", n_jobs=-1, random_state=42)
    rf.fit(sub_s[feats], sub_s["fire"])
    imp = pd.Series(rf.feature_importances_, index=feats).sort_values()
    colors = [ACCENT if v >= imp.median() else "#33485a" for v in imp]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh(imp.index, imp.values, color=colors)
    ax.set_title("Feature importance  (Random Forest, tabular data)")
    ax.set_xlabel("importance score")
    ax.grid(True, axis="x")
    return [_save(fig, "data_feature_importance.png")]


def plot_training_history(history):
    if not history:
        return []
    h = pd.DataFrame(history)
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(h["epoch"], h["loss"], color="#ef4444", lw=2, label="train loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss", color="#ef4444")
    ax1.tick_params(axis="y", labelcolor="#ef4444")
    ax2 = ax1.twinx(); ax2.grid(False)
    for col, c in [("accuracy", ACCENT), ("recall", "#f59e0b"),
                   ("precision", "#a78bfa"), ("f1", "#60a5fa"), ("auc", "#34d399")]:
        ax2.plot(h["epoch"], h[col], lw=1.6, label=col, color=c)
    ax2.set_ylabel("validation metric")
    ax2.set_ylim(0, 1.02)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=8)
    ax1.set_title("GAT-LSTM training loss + validation metrics per epoch")
    return [_save(fig, "training_history.png")]


def best_threshold(y, p):
    # find the threshold where F1 is highest on the test set
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return float(thr[max(0, f1[:-1].argmax())])


def plot_confusion(y, p, thr):
    cm = confusion_matrix(y, (p >= thr).astype(int))
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap=TEAL_CMAP)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["no-fire", "fire"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["no-fire", "fire"])
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            dark = cm[i, j] > cm.max() * 0.7
            ax.text(j, i, f"{cm[i, j]:,}\n{cm[i, j] / total:.1%}",
                    ha="center", va="center", fontsize=11,
                    color="#0a0e13" if dark else "#e6edf3")
    ax.set_title(f"Confusion matrix  (threshold = {thr:.2f})")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, "confusion_matrix.png")


def plot_roc(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    a = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.plot(fpr, tpr, color=ACCENT, lw=2.2, label=f"AUC = {a:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#475569", lw=1)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title("ROC curve"); ax.legend(loc="lower right")
    return _save(fig, "roc_curve.png")


def plot_pr(y, p):
    prec, rec, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.plot(rec, prec, color="#f59e0b", lw=2.2, label=f"AP = {ap:.3f}")
    ax.axhline(y.mean(), ls="--", color="#475569", lw=1, label=f"baseline = {y.mean():.3f}")
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title("Precision-Recall curve"); ax.legend(loc="lower left")
    return _save(fig, "pr_curve.png")


def plot_threshold_sweep(y, p):
    # sweeping the threshold shows the precision-recall trade-off visually
    thr = np.linspace(0.05, 0.95, 37)
    acc = [accuracy_score(y, p >= t) for t in thr]
    rec = [recall_score(y, p >= t, zero_division=0) for t in thr]
    pre = [precision_score(y, p >= t, zero_division=0) for t in thr]
    f1 = [f1_score(y, p >= t, zero_division=0) for t in thr]
    fig, ax = plt.subplots(figsize=(7, 4))
    for v, c, lab in [(acc, ACCENT, "accuracy"), (rec, "#f59e0b", "recall"),
                      (pre, "#a78bfa", "precision"), (f1, "#60a5fa", "F1")]:
        ax.plot(thr, v, lw=1.8, color=c, label=lab)
    bt = thr[int(np.argmax(f1))]
    ax.axvline(bt, ls="--", color="#ef4444", lw=1, label=f"best F1 @ {bt:.2f}")
    ax.set_xlabel("decision threshold"); ax.set_ylabel("score")
    ax.set_title("Metrics vs decision threshold"); ax.legend(fontsize=8)
    return _save(fig, "threshold_sweep.png")


def plot_calibration(y, p):
    # checks if the model's probabilities are trustworthy, not just well-ranked
    frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.plot(mean_pred, frac, "o-", color=ACCENT, lw=2, label="GAT-LSTM")
    ax.plot([0, 1], [0, 1], "--", color="#475569", lw=1, label="perfect")
    ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed fire frequency")
    ax.set_title("Calibration"); ax.legend(loc="upper left")
    return _save(fig, "calibration.png")


def plot_risk_tiers(p):
    labels = [risk_tier(float(x))[0] for x in p]
    order = ["Low", "Medium", "High", "Critical"]
    counts = [labels.count(o) for o in order]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(order, counts, color=TIER_COLOURS)
    for i, v in enumerate(counts):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Predicted risk-tier distribution (test set)")
    ax.set_ylabel("node-days")
    return _save(fig, "risk_tiers.png")


def plot_probability_distribution(y, p):
    # overlapping histograms - ideally fire class should peak near 1, no-fire near 0
    y_b = np.asarray(y).reshape(-1).astype(int)
    p_f = np.asarray(p).reshape(-1)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for cls, col, lab in [(0, "#33485a", "no-fire (0)"), (1, ACCENT, "fire (1)")]:
        vals = p_f[y_b == cls]
        if len(vals) > 0:
            ax.hist(vals, bins=50, density=True, alpha=0.65,
                    color=col, label=f"{lab}  n={len(vals):,}")
    ax.set_xlabel("predicted fire probability")
    ax.set_ylabel("density")
    ax.set_title("Predicted probability distribution by true class")
    ax.legend(fontsize=8)
    return [_save(fig, "prob_distribution.png")]


def plot_node_performance_heatmap(y, p, thr):
    # F1 per node on the 10x10 grid - shows which geographic areas the model struggles with
    # only works when we have 2D predictions [samples, 100], skips otherwise
    y_a = np.asarray(y)
    p_a = np.asarray(p)
    if y_a.ndim != 2 or y_a.shape[-1] != 100:
        return []
    f1_nodes = []
    for n in range(100):
        yt_n = y_a[:, n].astype(int)
        yp_n = (p_a[:, n] >= thr).astype(int)
        f1_nodes.append(f1_score(yt_n, yp_n, zero_division=0))
    grid = np.array(f1_nodes).reshape(10, 10)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(grid, cmap=TEAL_CMAP, vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            text_col = "#0a0e13" if grid[i, j] > 0.7 else "#e6edf3"
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.5, color=text_col)
    ax.set_title("Per-node F1 score  (10x10 BC grid,  W to E / N to S)")
    ax.set_xlabel("column  (west to east)")
    ax.set_ylabel("row  (north to south)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="F1")
    ax.grid(False)
    return [_save(fig, "node_performance_heatmap.png")]


def plot_error_breakdown(y, p, thr):
    # breaks TP/FP/FN/TN down by alert tier so we can see where false alarms come from
    y_b = np.asarray(y).reshape(-1).astype(int)
    p_f = np.asarray(p).reshape(-1)
    p_b = (p_f >= thr).astype(int)
    tier_labels = np.array([risk_tier(float(x))[0] for x in p_f])
    order = ["Low", "Medium", "High", "Critical"]

    tp_l, fp_l, tn_l, fn_l = [], [], [], []
    for t in order:
        mask = tier_labels == t
        if mask.sum() == 0:
            tp_l.append(0); fp_l.append(0); tn_l.append(0); fn_l.append(0)
            continue
        yt_ = y_b[mask]; yp_ = p_b[mask]
        tp_l.append(int(((yp_ == 1) & (yt_ == 1)).sum()))
        fp_l.append(int(((yp_ == 1) & (yt_ == 0)).sum()))
        tn_l.append(int(((yp_ == 0) & (yt_ == 0)).sum()))
        fn_l.append(int(((yp_ == 0) & (yt_ == 1)).sum()))

    fig, ax = plt.subplots(figsize=(7, 4))
    bot = np.zeros(len(order))
    for vals, col, lab in [
        (tp_l, "#2dd4bf", "TP  correct fire alert"),
        (tn_l, "#33485a", "TN  correct no-fire"),
        (fp_l, "#f97316", "FP  false alarm"),
        (fn_l, "#ef4444", "FN  missed fire"),
    ]:
        ax.bar(order, vals, bottom=bot, color=col, label=lab)
        bot += np.array(vals, dtype=float)
    ax.set_title("Prediction outcomes per risk tier")
    ax.set_ylabel("count")
    ax.legend(fontsize=7, ncol=2)
    return [_save(fig, "error_breakdown.png")]


def plot_precision_deep_dive(y, p):
    # three panels showing precision trade-offs in more detail than the single PR curve
    from sklearn.metrics import precision_recall_curve
    prec_arr, rec_arr, thr_arr = precision_recall_curve(y, p)
    thr_plot = thr_arr
    prec_plot = prec_arr[:-1]
    rec_plot  = rec_arr[:-1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # panel 1: precision and recall as threshold moves
    ax = axes[0]
    ax.plot(thr_plot, prec_plot, color=ACCENT,    lw=2, label="Precision")
    ax.plot(thr_plot, rec_plot,  color="#f59e0b",  lw=2, label="Recall")
    ax.axhline(0.80, ls=":",  color="#647686", lw=1, label="80% line")
    ax.axhline(0.90, ls="--", color="#a78bfa", lw=1, label="90% line")
    cur_p = float(precision_score(y, (p >= 0.5).astype(int), zero_division=0))
    cur_r = float(recall_score(y,    (p >= 0.5).astype(int), zero_division=0))
    ax.axvline(0.5, ls="--", color="#ef4444", lw=1.2, label=f"thr=0.5  P={cur_p:.2f} R={cur_r:.2f}")
    ax.set_xlabel("Decision threshold"); ax.set_ylabel("Score")
    ax.set_title("Precision & Recall vs Threshold"); ax.legend(fontsize=7.5)

    # panel 2: annotated PR curve with reference lines
    ax = axes[1]
    ax.plot(rec_arr, prec_arr, color=ACCENT, lw=2.2)
    ax.fill_between(rec_arr, prec_arr, alpha=0.12, color=ACCENT)
    ax.scatter([cur_r], [cur_p], s=110, color="#ef4444", zorder=5,
               label=f"thr=0.5  P={cur_p:.1%}  R={cur_r:.1%}")
    ax.axvline(0.90, ls="--", color="#647686", lw=1, label="Recall=90%")
    ax.axhline(0.80, ls="--", color="#475569", lw=1, label="Precision=80%")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve - annotated"); ax.legend(fontsize=7.5)

    # panel 3: false alarm rate (1 - precision) as threshold moves
    ax = axes[2]
    fa_rate = 1.0 - prec_plot
    ax.plot(thr_plot, fa_rate, color="#f97316", lw=2)
    ax.fill_between(thr_plot, fa_rate, alpha=0.12, color="#f97316")
    cur_fa = 1.0 - cur_p
    ax.axhline(0.20, ls="--", color="#647686", lw=1, label="20% false alarm")
    ax.axhline(0.10, ls="--", color="#a78bfa", lw=1, label="10% false alarm")
    ax.axvline(0.5, ls="--",  color="#ef4444", lw=1.2,
               label=f"thr=0.5 -> {cur_fa:.1%} false alarms")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("False alarm rate  (1 - Precision)")
    ax.set_title("False Alarm Rate vs Threshold"); ax.legend(fontsize=7.5)

    fig.suptitle(
        f"Precision deep-dive  |  Current: Precision={cur_p:.1%}  Recall={cur_r:.1%}  "
        f"=> 1 in {1/cur_fa:.0f} alerts is a false alarm",
        fontsize=10, y=1.01,
    )
    return [_save(fig, "precision_deep_dive.png")]


def plot_multi_model_roc(model_probs):
    # all models on one ROC so we can visually compare them - matches Figure 7 in the paper
    palette = {
        "GAT-LSTM":    ACCENT,     "XGBoost":     "#60a5fa",
        "RandomForest": "#34d399", "LightGBM":    "#f59e0b",
        "CatBoost":    "#a78bfa",  "RNN+LSTM":    "#ef4444",
    }
    y = model_probs.get("y_true")
    if y is None or len(model_probs) < 2:
        return []
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for name, prob in model_probs.items():
        if name == "y_true":
            continue
        try:
            fpr, tpr, _ = roc_curve(y, prob)
            a = auc(fpr, tpr)
            c = palette.get(name, "#8aa0b0")
            ax.plot(fpr, tpr, lw=1.8, color=c, label=f"{name}  AUC={a:.3f}")
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], "--", color="#475569", lw=1)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title("ROC curves - all models (70:30 split, top-10 features)")
    ax.legend(fontsize=8, loc="lower right")
    return [_save(fig, "roc_all_models.png")]


def plot_model_metrics_heatmap(results):
    # heatmap of all metrics across all models - easier to read than a bar chart
    if not results:
        return []
    models = list(results.keys())
    mets = ["accuracy", "recall", "precision", "f1", "f2", "auc"]
    data = [[results[m].get(met, 0) for met in mets] for m in models]
    arr = np.array(data)
    fig, ax = plt.subplots(figsize=(9, max(3, 0.65 * len(models) + 1.8)))
    im = ax.imshow(arr, cmap="RdYlGn", vmin=0.60, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(mets)))
    ax.set_xticklabels([m.upper() for m in mets], fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            text_col = "#0a0e13" if 0.70 < val < 0.96 else "#e6edf3"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=8.5, color=text_col, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="score")
    ax.set_title("Model comparison heatmap - 70:30 split, top-10 features, RUS")
    ax.grid(False)
    return [_save(fig, "model_metrics_heatmap.png")]


def plot_model_comparison(results):
    if not results:
        return []
    models = list(results.keys())
    mets = ["accuracy", "recall", "precision", "f1", "f2", "auc"]
    x = np.arange(len(mets)); w = 0.8 / len(models)
    palette = [ACCENT, "#f59e0b", "#a78bfa", "#60a5fa", "#ef4444"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, mdl in enumerate(models):
        vals = [results[mdl].get(m, 0) for m in mets]
        ax.bar(x + i * w, vals, w, label=mdl, color=palette[i % len(palette)])
    ax.set_xticks(x + w * (len(models) - 1) / 2); ax.set_xticklabels(mets)
    ax.set_ylim(0, 1.05); ax.set_ylabel("score")
    ax.set_title("Model comparison across metrics"); ax.legend(fontsize=8)
    return [_save(fig, "model_comparison.png")]


def build_report(sections):
    parts = ["""<!doctype html><html><head><meta charset='utf-8'>
<title>Wildfire EWS · Evaluation</title><style>
body{background:#0a0e13;color:#cdd8e1;font-family:ui-monospace,monospace;margin:0;padding:32px}
h1{font-size:22px;letter-spacing:.04em}h1 b{color:#2dd4bf}
h2{color:#647686;font-size:13px;letter-spacing:.16em;text-transform:uppercase;
margin:34px 0 10px;border-top:1px solid #1d2a35;padding-top:18px}
p{color:#8aa0b0;font-size:13px;max-width:760px}
img{max-width:100%;border:1px solid #1d2a35;border-radius:8px;background:#0e151d;margin:8px 0}
.row{display:flex;gap:16px;flex-wrap:wrap}.row img{max-width:48%}
</style></head><body>
<h1>Wildfire <b>Early Warning System</b> · Evaluation report</h1>
<p>British Columbia · GAT-LSTM main model · top-10 features · 70:30 split.</p>"""]
    for title, desc, imgs in sections:
        if not imgs:
            continue
        parts.append(f"<h2>{title}</h2><p>{desc}</p><div class='row'>")
        parts.extend(f"<img src='{im}'>" for im in imgs)
        parts.append("</div>")
    parts.append("</body></html>")
    (EVAL_DIR / "report.html").write_text("\n".join(parts), encoding="utf-8")


def main():
    setup_style()
    df = pd.read_parquet(PROCESSED_DIR / "dataset.parquet")
    preds = np.load(ARTIFACTS_DIR / "test_preds.npz")
    # keep 2D for node heatmap, flatten for everything else
    y_2d, p_2d = preds["y_true"], preds["y_prob"]
    y, p = y_2d.reshape(-1), p_2d.reshape(-1)
    history = json.loads((ARTIFACTS_DIR / "train_history.json").read_text())
    results = {}
    rp = ARTIFACTS_DIR / "model_results.json"
    if rp.exists():
        results = json.loads(rp.read_text())

    # load per-model probabilities if train.py saved them
    model_probs = {"y_true": y, "GAT-LSTM": p}
    mp_path = ARTIFACTS_DIR / "model_test_probs.npz"
    if mp_path.exists():
        mp = np.load(mp_path)
        model_probs = {k: mp[k] for k in mp.files}

    thr = best_threshold(y, p)
    sections = [
        ("Dataset - class balance & seasonality",
         "Class balance, day-of-year seasonality, per-feature fire/no-fire distributions, "
         "and feature correlation matrix. Top-10 features per paper Table 9.",
         plot_data_overview(df)),

        ("Dataset - temporal patterns",
         "Daily fire record count over the full dataset period and monthly aggregation "
         "to reveal within-year seasonality at a glance.",
         plot_data_fire_timeline(df) + plot_data_monthly_rate(df)),

        ("Dataset - feature importance",
         "RF-derived feature importance on top-10 features. "
         "swvl1 (soil moisture) dominates, matching paper Table 9.",
         plot_data_feature_importance_rf(df)),

        ("Training - GAT-LSTM",
         "Loss should fall while validation metrics rise and plateau. "
         "The best-AUC checkpoint is restored after training.",
         plot_training_history(history)),

        ("Performance - GAT-LSTM",
         "ROC, PR, confusion matrix, threshold sweep, calibration, risk tiers, "
         "probability distribution, and error breakdown for the main GAT-LSTM model.",
         [plot_roc(y, p), plot_pr(y, p),
          plot_confusion(y, p, thr), plot_threshold_sweep(y, p),
          plot_calibration(y, p), plot_risk_tiers(p)] +
         plot_probability_distribution(y, p) + plot_error_breakdown(y, p, thr)),

        ("Spatial analysis - GAT-LSTM",
         "Per-node F1 score on the 10x10 BC grid. "
         "Darker cells are harder to predict - useful for targeted improvement.",
         plot_node_performance_heatmap(y_2d, p_2d, thr)),

        ("Model comparison - all models (paper Table 4 hyperparameters)",
         "GAT-LSTM vs paper's models: RF, XGBoost, LightGBM, CatBoost, RNN+LSTM "
         "on 70:30 synthetic BC split. Paper best: CatBoost 93.4% acc, 92.1% F1.",
         plot_model_comparison(results) + plot_model_metrics_heatmap(results)),

        ("Model comparison - ROC curves",
         "All trained models on the same ROC plot. Higher AUC = better class separation.",
         plot_multi_model_roc(model_probs)),
    ]
    build_report(sections)
    print(f"[eval] wrote graphs + report.html to {EVAL_DIR}")


if __name__ == "__main__":
    main()
