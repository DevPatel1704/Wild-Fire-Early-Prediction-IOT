"""Generate architecture diagrams for the WildfireEWS paper."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

BG   = "#0e151d"
PANEL= "#111923"
LINE = "#1d2a35"
TEXT = "#e6edf3"
MUT  = "#8aa0b0"
TEAL = "#2dd4bf"
AMB  = "#f59e0b"
PURP = "#a78bfa"
RED  = "#ef4444"
BLUE = "#60a5fa"
GRN  = "#34d399"

def save(fig, name):
    fig.savefig(name, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"  saved {name}")


# ══════════════════════════════════════════════════════════════
#  FIGURE 1 — 7-layer system pipeline
# ══════════════════════════════════════════════════════════════
def draw_system_arch():
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.set_xlim(0, 13); ax.set_ylim(0, 7)

    # ── layer definitions ──────────────────────────────────────
    layers = [
        # (x_centre, y_centre, width, height, colour, title, subtitle)
        (1.3, 5.5, 2.2, 1.0, "#1e3a5f", "Layer 1\nData Sources",
         "ERA5 NetCDF · NFDB CSV\nNASA FIRMS · Synthetic Gen"),
        (4.0, 5.5, 2.0, 1.0, "#1e3a4a", "Layer 2\nKafka Ingestion",
         "Topic: sensor.readings\nat-least-once delivery"),
        (6.6, 5.5, 2.0, 1.0, "#1a3a2a", "Layer 3\nPreprocessing",
         "Feature select · Node snap\nSliding window · RUS · Scale"),
        (9.2, 5.5, 2.0, 1.0, "#2a2a1e", "Layer 4\nIoT Sensor Graph",
         "100 nodes · 10×10 BC grid\nk-NN adjacency (k=4)"),
        (11.6, 5.5, 1.5, 1.0, "#2a1e2a", "Layer 5\nModel Training",
         "GAT-LSTM\n+ 5 baselines"),

        (2.2, 3.2, 2.2, 1.0, "#1e2a3a", "Layer 6\nStream Prediction",
         "GAT-LSTM forecast (7 min)\nXGBoost alert (60 s)"),
        (5.5, 3.2, 2.2, 1.0, "#1e3a3a", "Layer 7\nFastAPI Backend",
         "REST /nodes /alerts\nWebSocket /ws broadcast"),
        (9.0, 3.2, 2.5, 1.0, "#1e2a1e", "React Dashboard",
         "Live BC risk map · Alert feed\nResults tab · 21 eval charts"),
    ]

    boxes = {}
    for (xc, yc, w, h, col, title, sub) in layers:
        rect = FancyBboxPatch((xc - w/2, yc - h/2), w, h,
                              boxstyle="round,pad=0.06",
                              facecolor=col, edgecolor=TEAL,
                              linewidth=1.2)
        ax.add_patch(rect)
        ax.text(xc, yc + 0.16, title, ha="center", va="center",
                fontsize=8, fontweight="bold", color=TEXT,
                fontfamily="DejaVu Sans Mono")
        ax.text(xc, yc - 0.22, sub, ha="center", va="center",
                fontsize=6.2, color=MUT,
                fontfamily="DejaVu Sans Mono")
        boxes[title.split("\n")[-1]] = (xc, yc, w, h)

    # ── horizontal arrows (top row) ────────────────────────────
    top_xs = [1.3, 4.0, 6.6, 9.2, 11.6]
    top_y  = 5.5
    for i in range(len(top_xs) - 1):
        x0 = top_xs[i] + [2.2,2.0,2.0,2.0,1.5][i]/2
        x1 = top_xs[i+1] - [2.2,2.0,2.0,2.0,1.5][i+1]/2
        ax.annotate("", xy=(x1, top_y), xytext=(x0, top_y),
                    arrowprops=dict(arrowstyle="-|>",
                                   color=TEAL, lw=1.4))

    # ── down arrow layer 4 -> layer 6 ─────────────────────────
    ax.annotate("", xy=(2.2, 3.7), xytext=(9.2, 5.0),
                arrowprops=dict(arrowstyle="-|>",
                                color=AMB, lw=1.4,
                                connectionstyle="arc3,rad=0.3"))
    ax.text(5.2, 4.55, "trained model", fontsize=7,
            color=AMB, ha="center", fontfamily="DejaVu Sans Mono")

    # ── horizontal arrows (bottom row) ────────────────────────
    bot = [(2.2, 5.5), (5.5, 3.2), (9.0, 3.2)]
    bw  = [2.2,        2.2,         2.5]
    for i in range(len(bot)-1):
        x0 = bot[i][0] + bw[i]/2
        x1 = bot[i+1][0] - bw[i+1]/2
        ax.annotate("", xy=(x1, 3.2), xytext=(x0, 3.2),
                    arrowprops=dict(arrowstyle="-|>",
                                   color=TEAL, lw=1.4))

    # ── Kafka loop arrow layer2 -> layer6 ─────────────────────
    ax.annotate("", xy=(2.2, 3.7), xytext=(4.0, 5.0),
                arrowprops=dict(arrowstyle="-|>",
                                color=BLUE, lw=1.2,
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(2.8, 4.42, "Kafka\nconsumer", fontsize=6.5,
            color=BLUE, ha="center", fontfamily="DejaVu Sans Mono")

    # ── title ──────────────────────────────────────────────────
    ax.text(6.5, 6.75, "WildfireEWS — End-to-End IoT Pipeline",
            ha="center", va="center", fontsize=13,
            fontweight="bold", color=TEXT,
            fontfamily="DejaVu Sans Mono")
    ax.text(6.5, 6.40,
            "Kafka ingestion → Preprocessing → IoT Sensor Graph → "
            "GAT-LSTM Training → Stream Prediction → Live Dashboard",
            ha="center", va="center", fontsize=7.5,
            color=MUT, fontfamily="DejaVu Sans Mono")

    # ── legend ─────────────────────────────────────────────────
    ax.text(0.2, 1.5, "Arrow colours:", fontsize=7, color=MUT,
            fontfamily="DejaVu Sans Mono")
    for col, label, y in [(TEAL, "Data / prediction flow", 1.1),
                          (AMB,  "Trained model artefact",  0.7),
                          (BLUE, "Kafka consumer path",     0.3)]:
        ax.plot([0.2, 0.9], [y, y], color=col, lw=2)
        ax.annotate("", xy=(0.9, y), xytext=(0.7, y),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5))
        ax.text(1.0, y, label, fontsize=7, color=col, va="center",
                fontfamily="DejaVu Sans Mono")

    save(fig, "system_arch.png")


# ══════════════════════════════════════════════════════════════
#  FIGURE 2 — GAT-LSTM model architecture
# ══════════════════════════════════════════════════════════════
def draw_gat_lstm():
    fig, ax = plt.subplots(figsize=(13, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.set_xlim(0, 13); ax.set_ylim(0, 8)

    def box(xc, yc, w, h, col, ec, title, sub="", sub2=""):
        rect = FancyBboxPatch((xc-w/2, yc-h/2), w, h,
                              boxstyle="round,pad=0.07",
                              facecolor=col, edgecolor=ec, linewidth=1.3)
        ax.add_patch(rect)
        ax.text(xc, yc + (0.14 if sub else 0), title,
                ha="center", va="center", fontsize=8,
                fontweight="bold", color=TEXT,
                fontfamily="DejaVu Sans Mono")
        if sub:
            ax.text(xc, yc - 0.18, sub, ha="center", va="center",
                    fontsize=6.5, color=MUT,
                    fontfamily="DejaVu Sans Mono")
        if sub2:
            ax.text(xc, yc - 0.40, sub2, ha="center", va="center",
                    fontsize=6.0, color=MUT,
                    fontfamily="DejaVu Sans Mono")

    def arr(x0, y0, x1, y1, col=TEAL, rad=0.0, label=""):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(
                        arrowstyle="-|>", color=col, lw=1.5,
                        connectionstyle=f"arc3,rad={rad}"))
        if label:
            mx, my = (x0+x1)/2, (y0+y1)/2
            ax.text(mx+0.05, my, label, fontsize=6.5,
                    color=col, ha="left", va="center",
                    fontfamily="DejaVu Sans Mono")

    # ── title ──────────────────────────────────────────────────
    ax.text(6.5, 7.65, "GAT-LSTM Architecture",
            ha="center", fontsize=14, fontweight="bold",
            color=TEXT, fontfamily="DejaVu Sans Mono")
    ax.text(6.5, 7.30,
            "Input X  [Batch, T=7, N=100, F=10]  +  Adjacency A  [100×100]",
            ha="center", fontsize=8, color=MUT,
            fontfamily="DejaVu Sans Mono")

    # ── input block ────────────────────────────────────────────
    box(1.2, 6.3, 1.8, 0.8, "#1a2a3a", BLUE,
        "Input", "[B, 7, 100, 10]")

    # ── timestep loop label ────────────────────────────────────
    ax.text(3.5, 6.95, "For each timestep t ∈ {1…7}:",
            fontsize=7.5, color=AMB,
            fontfamily="DejaVu Sans Mono", style="italic")
    rect = FancyBboxPatch((3.0, 5.55), 5.8, 1.25,
                          boxstyle="round,pad=0.1",
                          facecolor="none",
                          edgecolor=AMB, linewidth=1.0,
                          linestyle="dashed")
    ax.add_patch(rect)

    # ── GAT layer 1 ────────────────────────────────────────────
    box(4.0, 6.2, 2.0, 0.85, "#1e3050", TEAL,
        "GAT Layer 1",
        "in=10 → out=64",
        "heads=4, concat → [B,N,256]")

    # ── GAT layer 2 ────────────────────────────────────────────
    box(6.5, 6.2, 2.0, 0.85, "#1e3050", TEAL,
        "GAT Layer 2",
        "in=256 → out=64",
        "heads=1, mean → [B,N,64]")

    # ── layer norm after GAT ───────────────────────────────────
    box(8.8, 6.2, 1.4, 0.65, "#1a2a1a", GRN,
        "LayerNorm", "[B,N,64]")

    # ── arrows inside loop ─────────────────────────────────────
    arr(2.1, 6.3, 3.0, 6.3, BLUE, 0.0, "[B,N,10]")
    arr(5.0, 6.2, 5.5, 6.2, TEAL, 0.0, "[B,N,256]")
    arr(7.5, 6.2, 8.1, 6.2, TEAL, 0.0, "[B,N,64]")

    # ── stack over time ────────────────────────────────────────
    box(10.9, 6.2, 1.8, 0.75, "#2a2a1a", AMB,
        "Stack T steps", "[B,T,N,64]")
    arr(9.5, 6.2, 10.0, 6.2, GRN, 0.0)

    # ── reshape ────────────────────────────────────────────────
    box(10.9, 4.9, 1.8, 0.65, "#2a2a1a", AMB,
        "Reshape", "[B·N, T, 64]")
    arr(10.9, 5.82, 10.9, 5.23, AMB, 0.0)

    # ── LSTM ───────────────────────────────────────────────────
    box(8.5, 3.6, 2.2, 1.0, "#1e1e3a", PURP,
        "LSTM", "hidden=128, layers=2",
        "dropout=0.3  [B·N, T, 128]")
    arr(10.9, 4.58, 9.6, 3.9, AMB, -0.2, "feed")

    # ── layer norm after LSTM ──────────────────────────────────
    box(6.0, 3.6, 1.6, 0.65, "#1a2a1a", GRN,
        "LayerNorm", "[B·N, 128]")
    ax.text(7.4, 3.70, "last step h_T", fontsize=6.5,
            color=MUT, ha="left", fontfamily="DejaVu Sans Mono")
    arr(7.4, 3.6, 6.8, 3.6, PURP, 0.0)

    # ── head ───────────────────────────────────────────────────
    box(3.8, 3.6, 2.0, 1.0, "#2a1a2a", RED,
        "MLP Head",
        "Linear(128→64) → GELU",
        "Dropout(0.3) → Linear(64→1)")
    arr(5.2, 3.6, 4.8, 3.6, GRN, 0.0)

    # ── reshape to [B,N] ──────────────────────────────────────
    box(1.8, 3.6, 1.6, 0.65, "#2a1e1e", RED,
        "Reshape", "[B, N=100]")
    arr(2.8, 3.6, 2.6, 3.6, RED, 0.0)

    # ── sigmoid / output ──────────────────────────────────────
    box(1.8, 2.3, 1.6, 0.75, "#1e3020", GRN,
        "Sigmoid", "prob ∈ [0,1]^N")
    arr(1.8, 3.28, 1.8, 2.68, RED, 0.0, "inference")

    # ── loss (training) ───────────────────────────────────────
    box(4.5, 2.3, 2.2, 0.75, "#2a1a1a", RED,
        "BCEWithLogitsLoss",
        "+ pos_weight auto")
    arr(1.8, 3.28, 4.5, 2.68, RED, -0.3, "train logits")

    # ── output label ──────────────────────────────────────────
    ax.text(1.8, 1.6, "Output: per-node fire probability\n[B, 100]  → risk tier → alert",
            ha="center", fontsize=8, color=TEAL,
            fontfamily="DejaVu Sans Mono", fontweight="bold")

    # ── adjacency input ───────────────────────────────────────
    box(1.2, 4.8, 1.8, 0.65, "#1a2030", BLUE,
        "Adjacency A", "[100×100]  k-NN k=4")
    ax.annotate("", xy=(3.2, 6.0), xytext=(1.5, 5.13),
                arrowprops=dict(arrowstyle="-|>", color=BLUE,
                                lw=1.2,
                                connectionstyle="arc3,rad=-0.3"))
    ax.annotate("", xy=(5.7, 6.0), xytext=(1.7, 5.13),
                arrowprops=dict(arrowstyle="-|>", color=BLUE,
                                lw=1.2,
                                connectionstyle="arc3,rad=-0.4"))

    # ── legend ─────────────────────────────────────────────────
    legend_items = [
        (TEAL,  "GAT attention"),
        (PURP,  "LSTM temporal"),
        (GRN,   "Normalisation"),
        (RED,   "Head / loss"),
        (AMB,   "Reshape / stack"),
        (BLUE,  "Input / graph"),
    ]
    for i, (c, l) in enumerate(legend_items):
        xi = 0.3 + (i % 3) * 3.0
        yi = 1.05 if i < 3 else 0.55
        ax.plot([xi, xi+0.35], [yi+0.05, yi+0.05], color=c, lw=2.5)
        ax.text(xi+0.45, yi+0.05, l, fontsize=7, color=c,
                va="center", fontfamily="DejaVu Sans Mono")

    save(fig, "gat_lstm_arch.png")


if __name__ == "__main__":
    print("[arch] Generating architecture diagrams...")
    draw_system_arch()
    draw_gat_lstm()
    print("[arch] Done — system_arch.png  gat_lstm_arch.png")
