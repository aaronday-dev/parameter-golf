#!/usr/bin/env python3
"""
Generate the Parameter Golf Hypothesis #1 worksheet as a human-centered PDF.
Uses reportlab for layout + matplotlib for information graphics.
"""
import io
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.figure import Figure

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether, HRFlowable,
)
from reportlab.lib.utils import ImageReader

# ── constants ────────────────────────────────────────────────────────────────

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(OUT_DIR, "parameter-golf-hypothesis-001.pdf")

PAGE_W, PAGE_H = letter
MARGIN = 0.65 * inch
CONTENT_W = PAGE_W - 2 * MARGIN

# colours
C_BG       = colors.HexColor("#FAFAF8")
C_BLACK    = colors.HexColor("#1A1A1A")
C_DARK     = colors.HexColor("#333333")
C_MID      = colors.HexColor("#666666")
C_LIGHT    = colors.HexColor("#999999")
C_RULE     = colors.HexColor("#CCCCCC")
C_ACCENT   = colors.HexColor("#2D5F8A")
C_WIN      = colors.HexColor("#2E7D32")
C_KILL     = colors.HexColor("#C62828")
C_WARN     = colors.HexColor("#E65100")
C_HIGHLIGHT= colors.HexColor("#FFF3E0")

# matplotlib palette
MPL_BG     = "#FAFAF8"
MPL_ACCENT = "#2D5F8A"
MPL_WIN    = "#2E7D32"
MPL_KILL   = "#C62828"
MPL_WARN   = "#E65100"
MPL_GRID   = "#E0E0E0"
MPL_TEXT   = "#333333"
MPL_LIGHT  = "#999999"


# ── styles ───────────────────────────────────────────────────────────────────

def build_styles():
    ss = getSampleStyleSheet()
    styles = {}

    styles["title"] = ParagraphStyle(
        "title", parent=ss["Title"],
        fontName="Helvetica-Bold", fontSize=20, leading=24,
        textColor=C_BLACK, spaceAfter=4,
    )
    styles["subtitle"] = ParagraphStyle(
        "subtitle", parent=ss["Normal"],
        fontName="Helvetica", fontSize=10, leading=13,
        textColor=C_MID, spaceAfter=14,
    )
    styles["h1"] = ParagraphStyle(
        "h1", parent=ss["Heading1"],
        fontName="Helvetica-Bold", fontSize=14, leading=18,
        textColor=C_ACCENT, spaceBefore=16, spaceAfter=6,
    )
    styles["h2"] = ParagraphStyle(
        "h2", parent=ss["Heading2"],
        fontName="Helvetica-Bold", fontSize=11, leading=14,
        textColor=C_BLACK, spaceBefore=10, spaceAfter=4,
    )
    styles["body"] = ParagraphStyle(
        "body", parent=ss["Normal"],
        fontName="Helvetica", fontSize=9.5, leading=13,
        textColor=C_DARK, spaceAfter=6,
    )
    styles["body_tight"] = ParagraphStyle(
        "body_tight", parent=styles["body"],
        spaceAfter=2,
    )
    styles["mono"] = ParagraphStyle(
        "mono", parent=ss["Code"],
        fontName="Courier", fontSize=8.5, leading=11,
        textColor=C_DARK, spaceAfter=4,
        backColor=colors.HexColor("#F5F5F0"),
        borderPadding=(4, 6, 4, 6),
    )
    styles["callout"] = ParagraphStyle(
        "callout", parent=styles["body"],
        fontName="Helvetica-Bold", fontSize=9.5,
        textColor=C_ACCENT, leftIndent=12, spaceAfter=4,
    )
    styles["evidence_yes"] = ParagraphStyle(
        "ev_yes", parent=styles["body"],
        fontName="Helvetica", fontSize=9, leading=12,
        textColor=C_WIN, leftIndent=14, spaceAfter=2,
    )
    styles["evidence_no"] = ParagraphStyle(
        "ev_no", parent=styles["body"],
        fontName="Helvetica", fontSize=9, leading=12,
        textColor=C_KILL, leftIndent=14, spaceAfter=2,
    )
    styles["evidence_neutral"] = ParagraphStyle(
        "ev_neutral", parent=styles["body"],
        fontName="Helvetica", fontSize=9, leading=12,
        textColor=C_MID, leftIndent=14, spaceAfter=2,
    )
    styles["verdict"] = ParagraphStyle(
        "verdict", parent=styles["body"],
        fontName="Helvetica-Bold", fontSize=10, leading=13,
        textColor=C_BLACK, spaceBefore=4, spaceAfter=8,
        borderWidth=1, borderColor=C_ACCENT, borderPadding=(6, 8, 6, 8),
        backColor=colors.HexColor("#EBF2F8"),
    )
    styles["small"] = ParagraphStyle(
        "small", parent=ss["Normal"],
        fontName="Helvetica", fontSize=7.5, leading=10,
        textColor=C_LIGHT, spaceAfter=2,
    )
    styles["footer"] = ParagraphStyle(
        "footer", parent=ss["Normal"],
        fontName="Helvetica", fontSize=7, leading=9,
        textColor=C_LIGHT, alignment=TA_CENTER,
    )
    return styles


# ── figure helpers ───────────────────────────────────────────────────────────

def fig_to_image(fig: Figure, dpi=180) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=MPL_BG, edgecolor="none", pad_inches=0.08)
    plt.close(fig)
    buf.seek(0)
    # measure
    from PIL import Image as PILImage
    pil = PILImage.open(buf)
    w_px, h_px = pil.size
    buf.seek(0)
    # scale to content width
    aspect = h_px / w_px
    img_w = min(CONTENT_W, 6.8 * inch)
    img_h = img_w * aspect
    return Image(buf, width=img_w, height=img_h)


def make_experiment_waterfall():
    """Horizontal bar chart: experiment results ranked, with current best highlighted."""
    runs = [
        ("seq MLP×4  (best)",       2.3580, True),
        ("seq MLP×3",               2.3733, False),
        ("mirror MLP×3 + dir-C",    2.3813, False),
        ("mirror + dir-C",          2.3899, False),
    ]
    smoke_runs = [
        ("smoke: seq MLP×4",        2.6117, True),
        ("smoke: dim528 MLP×4",     2.6144, False),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.2), gridspec_kw={"width_ratios": [3, 2]})
    fig.patch.set_facecolor(MPL_BG)

    for ax, data, title_text in [(ax1, runs, "Promoted runs (200 iter)"),
                                  (ax2, smoke_runs, "Smoke runs (80 iter)")]:
        ax.set_facecolor(MPL_BG)
        names = [r[0] for r in data]
        vals  = [r[1] for r in data]
        bests = [r[2] for r in data]
        bar_colors = [MPL_ACCENT if b else MPL_GRID for b in bests]
        edge_colors = [MPL_ACCENT if b else MPL_LIGHT for b in bests]

        y_pos = np.arange(len(names))
        # plot relative to a baseline for visual clarity
        baseline = min(vals) - 0.005
        widths = [v - baseline for v in vals]
        bars = ax.barh(y_pos, widths, left=baseline, color=bar_colors,
                       edgecolor=edge_colors, linewidth=0.8, height=0.55)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=7.5, color=MPL_TEXT)
        ax.invert_yaxis()
        ax.set_xlabel("val_bpb (exact)", fontsize=7.5, color=MPL_TEXT)
        ax.set_title(title_text, fontsize=8.5, fontweight="bold", color=MPL_TEXT, pad=6)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.tick_params(axis="x", labelsize=7, colors=MPL_TEXT)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(MPL_GRID)
        ax.spines["bottom"].set_color(MPL_GRID)
        ax.grid(axis="x", color=MPL_GRID, linewidth=0.5, alpha=0.7)

        # annotate exact values
        for i, (v, b) in enumerate(zip(vals, bests)):
            txt_color = "white" if b else MPL_TEXT
            ax.text(v - 0.001, i, f" {v:.4f}", va="center", ha="right",
                    fontsize=7, fontweight="bold" if b else "normal", color=txt_color)

    fig.tight_layout(w_pad=3)
    return fig_to_image(fig)


def make_parameter_budget_bar():
    """Horizontal stacked bar showing where the bytes go — no label collisions."""
    labels = [
        "MLP proj (2048→512)",
        "MLP fc (512→2048)",
        "Attn Q+proj (512→512)",
        "Attn K+V (512→256)",
        "Embedding (1024×512)",
        "Control tensors (fp16)",
    ]
    sizes_M = [
        9 * 1048576 / 1e6,   # MLP proj
        9 * 1048576 / 1e6,   # MLP fc
        9 * 2 * 262144 / 1e6,  # Q + proj
        9 * 2 * 131072 / 1e6,  # K + V
        524288 / 1e6,          # embedding
        9 * 2056 * 2 / 1e6,   # control tensors
    ]
    total = sum(sizes_M)
    pcts = [s / total * 100 for s in sizes_M]

    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    fig.patch.set_facecolor(MPL_BG)
    ax.set_facecolor(MPL_BG)

    bar_colors = [MPL_KILL, MPL_ACCENT, "#3A7CA5", "#6BA3C7", "#A8D0E6", "#D0D0D0"]
    edge_colors = [MPL_KILL, MPL_ACCENT, "#3A7CA5", "#6BA3C7", "#A8D0E6", MPL_LIGHT]

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, sizes_M, color=bar_colors, edgecolor=edge_colors,
                   linewidth=0.8, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8, color=MPL_TEXT)
    ax.invert_yaxis()
    ax.set_xlabel("Parameters (millions)", fontsize=8, color=MPL_TEXT)
    ax.set_title("Where the bytes live (pre-compression int8 payload)", fontsize=9.5,
                 fontweight="bold", color=MPL_TEXT, pad=8)

    # annotate with size + pct
    for i, (s, p) in enumerate(zip(sizes_M, pcts)):
        txt_color = "white" if i < 2 else MPL_TEXT
        ha = "right" if s > 2 else "left"
        xpos = s - 0.15 if s > 2 else s + 0.15
        ax.text(xpos, i, f" {s:.1f}M  ({p:.0f}%)", va="center", ha=ha,
                fontsize=7.5, fontweight="bold", color=txt_color)

    # highlight hypothesis target
    ax.annotate("HYPOTHESIS TARGET",
                xy=(sizes_M[0] + 0.2, 0), xytext=(sizes_M[0] + 1.5, 0.7),
                fontsize=8, fontweight="bold", color=MPL_KILL,
                arrowprops=dict(arrowstyle="-|>", color=MPL_KILL, lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", fc="#FFEBEE", ec=MPL_KILL, lw=0.8))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MPL_GRID)
    ax.spines["bottom"].set_color(MPL_GRID)
    ax.grid(axis="x", color=MPL_GRID, linewidth=0.5, alpha=0.5)
    ax.tick_params(axis="x", labelsize=7)

    fig.tight_layout()
    return fig_to_image(fig)


def make_activation_comparison():
    """Show relu² vs gated relu² output distributions — the mechanistic argument."""
    x = np.linspace(-3, 3, 500)
    relu2 = np.maximum(x, 0) ** 2

    # Gated: gate * up, where gate = relu²(x_gate), up = linear(x_up)
    # Show the effective output distribution shape
    np.random.seed(42)
    n = 10000
    x_gate = np.random.randn(n)
    x_up = np.random.randn(n)
    gated_out = np.maximum(x_gate, 0) ** 2 * x_up  # key: x_up is symmetric

    relu2_out = np.maximum(np.random.randn(n), 0) ** 2  # always ≥ 0

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.0))
    fig.patch.set_facecolor(MPL_BG)

    # Panel 1: activation functions
    ax = axes[0]
    ax.set_facecolor(MPL_BG)
    ax.plot(x, relu2, color=MPL_KILL, linewidth=1.5, label="relu²(x)")
    ax.plot(x, np.maximum(x, 0), color=MPL_LIGHT, linewidth=1, linestyle="--", label="relu(x)", alpha=0.5)
    ax.set_title("Activation shape", fontsize=8, fontweight="bold", color=MPL_TEXT)
    ax.set_xlabel("input", fontsize=7, color=MPL_TEXT)
    ax.set_ylabel("output", fontsize=7, color=MPL_TEXT)
    ax.legend(fontsize=6, loc="upper left")
    ax.tick_params(labelsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color=MPL_GRID, linewidth=0.3)

    # Panel 2: relu² output histogram (input to mlp.proj)
    ax = axes[1]
    ax.set_facecolor(MPL_BG)
    ax.hist(relu2_out, bins=60, color=MPL_KILL, alpha=0.8, edgecolor="white", linewidth=0.3, density=True)
    ax.axvline(0, color=MPL_TEXT, linewidth=0.5, linestyle="--")
    ax.set_title("relu² → proj input\n(one-sided, heavy tail)", fontsize=8,
                 fontweight="bold", color=MPL_KILL)
    ax.set_xlabel("activation value", fontsize=7, color=MPL_TEXT)
    ax.tick_params(labelsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.annotate("quantization\nwastes half\nthe int8 grid\n(negatives unused)",
                xy=(0.5, 0.65), xycoords="axes fraction",
                fontsize=6, color=MPL_KILL, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="#FFEBEE", ec=MPL_KILL, lw=0.5))

    # Panel 3: gated output histogram
    ax = axes[2]
    ax.set_facecolor(MPL_BG)
    ax.hist(gated_out, bins=60, color=MPL_WIN, alpha=0.8, edgecolor="white", linewidth=0.3, density=True)
    ax.axvline(0, color=MPL_TEXT, linewidth=0.5, linestyle="--")
    ax.set_title("gated relu² → proj input\n(symmetric, compressor-friendly)", fontsize=8,
                 fontweight="bold", color=MPL_WIN)
    ax.set_xlabel("activation value", fontsize=7, color=MPL_TEXT)
    ax.tick_params(labelsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.annotate("full int8 range\nutilized\n(symmetric dist)",
                xy=(0.5, 0.65), xycoords="axes fraction",
                fontsize=6, color=MPL_WIN, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="#E8F5E9", ec=MPL_WIN, lw=0.5))

    fig.tight_layout(w_pad=2)
    return fig_to_image(fig)


def make_locus_diagram():
    """Flow diagram showing where in the forward pass the hypothesis targets."""
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    fig.patch.set_facecolor(MPL_BG)
    ax.set_facecolor(MPL_BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.5)
    ax.axis("off")

    # Pipeline stages as boxes
    stages = [
        (0.3, 1.2, 1.4, 1.0, "Embedding\n(1024→512)", "#E0E0E0", MPL_TEXT),
        (2.0, 1.2, 1.4, 1.0, "Attention\nQ/K/V → proj", "#E0E0E0", MPL_TEXT),
        (3.7, 1.2, 1.4, 1.0, "MLP fc\n512→2048", "#E0E0E0", MPL_TEXT),
        (5.4, 0.6, 1.4, 2.2, "relu²\nactivation\n\n→ one-sided\n→ heavy tail\n→ sparse", "#FFCDD2", MPL_KILL),
        (7.1, 0.6, 1.4, 2.2, "MLP proj\n2048→512\n\n→ skewed input\n→ bad int8 fit\n→ lzma pain", "#FFCDD2", MPL_KILL),
        (8.8, 1.2, 1.0, 1.0, "residual\n+ next", "#E0E0E0", MPL_TEXT),
    ]

    for x, y, w, h, txt, fc, tc in stages:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08",
            facecolor=fc, edgecolor=tc, linewidth=0.8,
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, txt, ha="center", va="center",
                fontsize=6.5, color=tc, fontweight="bold" if fc != "#E0E0E0" else "normal")

    # Arrows
    arrow_kw = dict(arrowstyle="-|>", color=MPL_LIGHT, lw=1.2)
    for x1, x2 in [(1.7, 2.0), (3.4, 3.7), (5.1, 5.4), (6.8, 7.1), (8.5, 8.8)]:
        ax.annotate("", xy=(x2, 1.7), xytext=(x1, 1.7), arrowprops=arrow_kw)

    # Big annotation arrow pointing to the problem zone
    ax.annotate("LOCUS: the relu² → proj boundary",
                xy=(6.1, 0.45), xytext=(3.5, 0.15),
                fontsize=8, fontweight="bold", color=MPL_KILL,
                arrowprops=dict(arrowstyle="-|>", color=MPL_KILL, lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3E0", ec=MPL_WARN, lw=1))

    # "×9 blocks" label
    ax.text(5.0, 3.2, "× 9 blocks  →  dominates artifact size",
            fontsize=8, color=MPL_ACCENT, fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="#EBF2F8", ec=MPL_ACCENT, lw=0.8))
    # bracket
    ax.plot([2.0, 2.0, 8.5, 8.5], [2.9, 3.0, 3.0, 2.9], color=MPL_ACCENT, lw=1)

    return fig_to_image(fig)


def make_patch_diagram():
    """Side-by-side: current MLP vs proposed gated MLP, with param counts."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    fig.patch.set_facecolor(MPL_BG)

    def draw_mlp(ax, title, layers, color, total_params, notes):
        ax.set_facecolor(MPL_BG)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis("off")

        ax.text(5, 5.6, title, ha="center", va="center",
                fontsize=10, fontweight="bold", color=color)

        for i, (x, y, w, h, txt, fc) in enumerate(layers):
            rect = mpatches.FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.06",
                facecolor=fc, edgecolor=color, linewidth=1,
            )
            ax.add_patch(rect)
            ax.text(x + w/2, y + h/2, txt, ha="center", va="center",
                    fontsize=7, color=MPL_TEXT, fontweight="bold")

        # param count box
        ax.text(5, 0.4, f"Total: {total_params:,} params/block",
                ha="center", va="center", fontsize=8, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1))

        for y_pos, note in zip([0.9 + 0.3*i for i in range(len(notes))], notes):
            pass  # skip, use annotation below instead

    # Current MLP
    ax = ax1
    ax.set_facecolor(MPL_BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.text(5, 6.6, "CURRENT: relu² MLP", ha="center", fontsize=10, fontweight="bold", color=MPL_KILL)

    boxes_current = [
        (1, 4.8, 8, 0.8, "input x  (512-dim)"),
        (1, 3.5, 8, 0.8, "fc: 512 → 2048   (1,048,576 params)"),
        (1, 2.2, 8, 0.8, "relu²   (one-sided, heavy-tailed)"),
        (1, 0.9, 8, 0.8, "proj: 2048 → 512   (1,048,576 params)"),
    ]
    fc_colors = ["#E0E0E0", "#FFCDD2", "#FFCDD2", "#FFCDD2"]
    for (x, y, w, h, txt), fc in zip(boxes_current, fc_colors):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                        facecolor=fc, edgecolor=MPL_KILL, linewidth=0.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=7.5, color=MPL_TEXT)
    for y1, y2 in [(4.8, 4.3), (3.5, 3.0), (2.2, 1.7)]:
        ax.annotate("", xy=(5, y2), xytext=(5, y1), arrowprops=dict(arrowstyle="-|>", color=MPL_LIGHT, lw=1))
    ax.text(5, 0.3, "2,097,152 params/block", ha="center", fontsize=9, fontweight="bold", color=MPL_KILL,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=MPL_KILL, lw=1))

    # Proposed gated MLP
    ax = ax2
    ax.set_facecolor(MPL_BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.text(5, 6.6, "PROPOSED: gated relu² MLP", ha="center", fontsize=10, fontweight="bold", color=MPL_WIN)

    # Two parallel paths — pulled inward to avoid edge clipping
    # gate path (left)
    gate_boxes = [
        (0.8, 4.8, 3.8, 0.7, "gate: 512 → 1365"),
        (0.8, 3.6, 3.8, 0.7, "relu²"),
    ]
    # up path (right)
    up_boxes = [
        (5.4, 4.8, 3.8, 0.7, "up: 512 → 1365"),
    ]
    # merge
    merge_boxes = [
        (0.8, 2.3, 8.4, 0.7, "element-wise multiply  (symmetric output)"),
        (0.8, 1.1, 8.4, 0.7, "proj: 1365 → 512   (698,880 params)"),
    ]

    input_box = (0.8, 5.8, 8.4, 0.6, "input x  (512-dim)")
    rect = mpatches.FancyBboxPatch((0.8, 5.8), 8.4, 0.6, boxstyle="round,pad=0.06",
                                    facecolor="#E0E0E0", edgecolor=MPL_WIN, linewidth=0.8)
    ax.add_patch(rect)
    ax.text(5, 6.1, "input x  (512-dim)", ha="center", va="center", fontsize=7.5, color=MPL_TEXT)

    for (x, y, w, h, txt) in gate_boxes:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                        facecolor="#E8F5E9", edgecolor=MPL_WIN, linewidth=0.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=7, color=MPL_TEXT)

    for (x, y, w, h, txt) in up_boxes:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                        facecolor="#E8F5E9", edgecolor=MPL_WIN, linewidth=0.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=7, color=MPL_TEXT)

    for (x, y, w, h, txt) in merge_boxes:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                        facecolor="#E8F5E9", edgecolor=MPL_WIN, linewidth=0.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=7, color=MPL_TEXT)

    # arrows — adjusted for new box positions
    ax.annotate("", xy=(2.7, 5.5), xytext=(3.5, 5.8), arrowprops=dict(arrowstyle="-|>", color=MPL_LIGHT, lw=1))
    ax.annotate("", xy=(7.3, 5.5), xytext=(6.5, 5.8), arrowprops=dict(arrowstyle="-|>", color=MPL_LIGHT, lw=1))
    ax.annotate("", xy=(2.7, 4.3), xytext=(2.7, 4.8), arrowprops=dict(arrowstyle="-|>", color=MPL_LIGHT, lw=1))
    # gate and up merge
    ax.annotate("", xy=(3.5, 3.0), xytext=(2.7, 3.6), arrowprops=dict(arrowstyle="-|>", color=MPL_LIGHT, lw=1))
    ax.annotate("", xy=(6.5, 3.0), xytext=(7.3, 5.1), arrowprops=dict(arrowstyle="-|>", color=MPL_LIGHT, lw=1))
    ax.annotate("", xy=(5, 1.8), xytext=(5, 2.3), arrowprops=dict(arrowstyle="-|>", color=MPL_LIGHT, lw=1))

    ax.text(5, 0.4, "2,096,640 params/block  (−512 net)", ha="center", fontsize=9,
            fontweight="bold", color=MPL_WIN,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=MPL_WIN, lw=1))

    # no external path labels needed — box text is self-explanatory

    fig.tight_layout(w_pad=2)
    return fig_to_image(fig)


def make_decision_threshold():
    """Visual kill/win threshold on a number line."""
    fig, ax = plt.subplots(figsize=(7.0, 1.8))
    fig.patch.set_facecolor(MPL_BG)
    ax.set_facecolor(MPL_BG)

    # Number line
    xmin, xmax = 2.575, 2.655
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.3, 2.8)

    # Zones
    baseline = 2.6117
    win_thresh = 2.600
    kill_thresh = 2.620

    # Kill zone (right of kill_thresh)
    ax.axvspan(kill_thresh, xmax, alpha=0.15, color=MPL_KILL)
    # Ambiguous zone
    ax.axvspan(win_thresh, kill_thresh, alpha=0.08, color=MPL_WARN)
    # Win zone (left of win_thresh)
    ax.axvspan(xmin, win_thresh, alpha=0.15, color=MPL_WIN)

    # Win threshold
    ax.axvline(win_thresh, color=MPL_WIN, linewidth=1.5, linestyle="--", zorder=4)
    ax.text(win_thresh - 0.005, 0.8, f"WIN\n< {win_thresh:.3f}", ha="center",
            fontsize=8, fontweight="bold", color=MPL_WIN)

    # Kill threshold
    ax.axvline(kill_thresh, color=MPL_KILL, linewidth=1.5, linestyle="--", zorder=4)
    ax.text(kill_thresh + 0.007, 0.8, f"KILL\n> {kill_thresh:.3f}", ha="center",
            fontsize=8, fontweight="bold", color=MPL_KILL)

    # Ambiguous label (between thresholds, low)
    ax.text((win_thresh + kill_thresh) / 2, 0.2, "inconclusive",
            ha="center", fontsize=6.5, color=MPL_WARN, fontstyle="italic")

    # Baseline marker — positioned above and offset to avoid overlapping zone labels
    ax.axvline(baseline, color=MPL_ACCENT, linewidth=2, linestyle="-", zorder=5)
    ax.annotate(f"baseline  {baseline:.4f}", xy=(baseline, 1.6), xytext=(baseline + 0.012, 2.3),
                fontsize=8, fontweight="bold", color=MPL_ACCENT, ha="center",
                arrowprops=dict(arrowstyle="-|>", color=MPL_ACCENT, lw=1.2),
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=MPL_ACCENT, lw=0.8))

    ax.set_xlabel("smoke val_bpb (exact)", fontsize=8, color=MPL_TEXT)
    ax.set_title("Decision thresholds — one smoke run decides", fontsize=9,
                 fontweight="bold", color=MPL_TEXT, pad=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    fig.tight_layout()
    return fig_to_image(fig)


# ── document assembly ────────────────────────────────────────────────────────

def build_pdf():
    styles = build_styles()
    doc = SimpleDocTemplate(
        OUT_PATH,
        pagesize=letter,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
    )

    story = []
    P = lambda text, style_name: Paragraph(text, styles[style_name])
    SP = lambda h=6: Spacer(1, h)
    HR = lambda: HRFlowable(width="100%", thickness=0.5, color=C_RULE, spaceAfter=8, spaceBefore=4)

    # ── HEADER ──
    story.append(P("Parameter Golf — Hypothesis #001", "title"))
    story.append(P("Gated relu² MLP for quantization-friendly weight distributions", "subtitle"))
    story.append(P("2026-03-22  ·  Target: move val_bpb by hundredths, not thousandths", "subtitle"))
    story.append(HR())

    # ── 1. EVIDENCE READ ──
    story.append(P("1. What the evidence says", "h1"))

    story.append(P(
        "Four facts from recent experiments, in order of importance:", "body"
    ))
    story.append(P(
        "&#x2714;  MLP capacity is the binding constraint — MLP×4 beat MLP×3 by 0.015 bpb on promoted runs",
        "evidence_yes"
    ))
    story.append(P(
        "&#x2718;  Width doesn't help — dim 528 lost to dim 512 despite more raw parameters",
        "evidence_no"
    ))
    story.append(P(
        "&#x2718;  Shared-core recurrence / stabilization consistently hurt post-compression score",
        "evidence_no"
    ))
    story.append(P(
        "&#x25CB;  Control tensors (fp16 passthrough) are a direct byte cost with diminishing returns",
        "evidence_neutral"
    ))
    story.append(SP(4))
    story.append(P(
        "The model is not undertrained or under-dimensioned. It is <b>using its dimensions inefficiently "
        "through the quantization bottleneck</b>. The MLP is where most parameters live, "
        "and the activation function determines how those parameters compress.",
        "verdict"
    ))
    story.append(SP(6))
    story.append(make_experiment_waterfall())
    story.append(SP(8))

    # ── 2. LOCUS ──
    story.append(P("2. Locus — where is the failure?", "h1"))
    story.append(P(
        "The MLP projection matrix (<font face='Courier'>mlp.proj</font>) in each of the 9 blocks. "
        "These 9 matrices hold <b>9.4M parameters</b> — roughly 36% of all model parameters — "
        "and they receive the output of relu², which is strictly non-negative and heavy-tailed.",
        "body"
    ))
    story.append(P(
        "This is a <b>boundary problem</b>: the failure occurs at the boundary between "
        "the activation function and the projection that must survive int8 quantization + lzma compression.",
        "body"
    ))
    story.append(SP(4))
    story.append(make_locus_diagram())
    story.append(SP(4))
    story.append(make_parameter_budget_bar())
    story.append(SP(4))

    # ── 3. MECHANISM ──
    story.append(P("3. Mechanism — what could fix it?", "h1"))
    story.append(P(
        "The core argument in three steps:", "body"
    ))
    story.append(SP(2))
    story.append(P(
        "<b>Step A: The input distribution to mlp.proj is pathological for int8.</b> "
        "relu² outputs are ≥ 0 and heavy-tailed. The int8 grid spans [−127, +127], "
        "but negative values are never used. Half the representable states are wasted. "
        "The heavy tail forces high per-row scales, crushing precision for typical activations.",
        "body"
    ))
    story.append(P(
        "<b>Step B: A gated activation produces symmetric, lighter-tailed inputs to proj.</b> "
        "gate(x) · up(x) multiplies the one-sided relu² gate output by a signed linear projection. "
        "The result is symmetric around zero and uses the full int8 range.",
        "body"
    ))
    story.append(P(
        "<b>Step C: Symmetric distributions compress better under lzma.</b> "
        "lzma exploits byte-level patterns. Symmetric int8 values around zero have more "
        "regular byte patterns than one-sided values, improving the compression ratio "
        "for the same information content.",
        "body"
    ))
    story.append(SP(6))
    story.append(make_activation_comparison())
    story.append(SP(8))

    # ── 3b. WHY NOT JUST SwiGLU ──
    story.append(P("Why not standard SwiGLU?", "h2"))
    story.append(P(
        "SwiGLU uses SiLU (swish) as the gate activation. That would work, but relu² has a known advantage "
        "in this codebase: it is cheaper to compute and the gradient signal is compatible with the Muon optimizer. "
        "The hypothesis is specifically about <b>gating with relu²</b>, not about changing the activation family. "
        "The key insight is that the gate × up product creates symmetry <i>regardless of the gate activation shape</i>.",
        "body"
    ))

    # ── 4. PATCH ──
    story.append(P("4. The patch — minimal code change", "h1"))
    story.append(P(
        "Replace the MLP class. One class, ~10 lines changed:", "body"
    ))
    story.append(P(
        '<font face="Courier" size="8">'
        'class MLP(nn.Module):<br/>'
        '&nbsp;&nbsp;def __init__(self, dim: int, mlp_mult: int):<br/>'
        '&nbsp;&nbsp;&nbsp;&nbsp;super().__init__()<br/>'
        '&nbsp;&nbsp;&nbsp;&nbsp;hidden = int(dim * mlp_mult * 2 / 3)&nbsp;&nbsp;# 1365 for dim=512, mult=4<br/>'
        '&nbsp;&nbsp;&nbsp;&nbsp;self.fc_gate = CastedLinear(dim, hidden)<br/>'
        '&nbsp;&nbsp;&nbsp;&nbsp;self.fc_up&nbsp;&nbsp; = CastedLinear(dim, hidden)<br/>'
        '&nbsp;&nbsp;&nbsp;&nbsp;self.proj&nbsp;&nbsp;&nbsp; = CastedLinear(hidden, dim)<br/>'
        '<br/>'
        '&nbsp;&nbsp;def __call__(self, x: mx.array) -> mx.array:<br/>'
        '&nbsp;&nbsp;&nbsp;&nbsp;return self.proj(nn.relu2(self.fc_gate(x)) * self.fc_up(x))<br/>'
        '</font>',
        "body"
    ))
    story.append(SP(6))
    story.append(make_patch_diagram())
    story.append(SP(6))

    # ── 5. SMOKE PROTOCOL ──
    story.append(P("5. Smoke run protocol", "h1"))
    story.append(P(
        "One run. 80 iterations on smoke data. Compare exact post-quantized val_bpb against the smoke baseline.", "body"
    ))

    # Table: run config
    config_data = [
        ["Parameter", "Value", "Rationale"],
        ["MLP activation", "gated relu²", "The hypothesis under test"],
        ["MLP hidden dim", "1365", "int(512 × 4 × 2/3) — param neutral"],
        ["MODEL_DIM", "512", "Unchanged"],
        ["NUM_LAYERS", "9", "Unchanged"],
        ["MLP_MULT", "4", "Unchanged (reinterpreted for gating)"],
        ["ITERATIONS", "80", "Smoke length"],
        ["Dataset", "fineweb10B_sp1024_smoke", "Standard smoke data"],
        ["Compressor", "lzma", "Match baseline compressor"],
        ["Everything else", "identical to baseline", "Isolate the variable"],
    ]
    config_table = Table(config_data, colWidths=[1.4*inch, 2.0*inch, 2.6*inch])
    config_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEADING", (0, 0), (-1, -1), 11),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, C_RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(config_table)
    story.append(SP(10))

    # ── 6. DECISION CRITERIA ──
    story.append(P("6. Decision criteria — exact thresholds", "h1"))
    story.append(make_decision_threshold())
    story.append(SP(6))

    # Decision table
    decision_data = [
        ["Outcome", "Smoke val_bpb", "Action"],
        ["WIN", "< 2.600", "Promote to 200-iter run immediately"],
        ["INTERESTING", "2.600 – 2.611", "Worth a promoted run to confirm"],
        ["NOISE", "2.611 – 2.620", "Within noise of baseline — inconclusive, park it"],
        ["KILL", "> 2.620", "Kill the hypothesis. Do not tune. Move on."],
    ]
    dt = Table(decision_data, colWidths=[1.2*inch, 1.4*inch, 3.4*inch])
    dt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("LEADING", (0, 0), (-1, -1), 12),
        ("BACKGROUND", (0, 1), (0, 1), colors.HexColor("#E8F5E9")),
        ("TEXTCOLOR", (0, 1), (0, 1), C_WIN),
        ("FONTNAME", (0, 1), (0, 1), "Helvetica-Bold"),
        ("BACKGROUND", (0, 2), (0, 2), colors.HexColor("#FFF3E0")),
        ("TEXTCOLOR", (0, 2), (0, 2), C_WARN),
        ("FONTNAME", (0, 2), (0, 2), "Helvetica-Bold"),
        ("BACKGROUND", (0, 3), (0, 3), colors.HexColor("#FFF3E0")),
        ("TEXTCOLOR", (0, 3), (0, 3), C_LIGHT),
        ("BACKGROUND", (0, 4), (0, 4), colors.HexColor("#FFEBEE")),
        ("TEXTCOLOR", (0, 4), (0, 4), C_KILL),
        ("FONTNAME", (0, 4), (0, 4), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, C_RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    # Fix: apply colored backgrounds to first column cells
    dt.setStyle(TableStyle([
        ("BACKGROUND", (0, 1), (0, 1), colors.HexColor("#E8F5E9")),
        ("BACKGROUND", (0, 2), (0, 2), colors.HexColor("#FFF3E0")),
        ("BACKGROUND", (0, 3), (0, 3), colors.HexColor("#F5F5F5")),
        ("BACKGROUND", (0, 4), (0, 4), colors.HexColor("#FFEBEE")),
    ]))
    story.append(dt)
    story.append(SP(10))

    # ── 7. PRE-FLIGHT CHECK ──
    story.append(P("7. Pre-flight diagnostic (optional but recommended)", "h1"))
    story.append(P(
        "Before running the smoke, load the best <font face='Courier'>.ptx</font> artifact and dump "
        "per-row scale statistics for every quantized tensor. If <font face='Courier'>mlp.proj</font> "
        "rows show high scale variance relative to <font face='Courier'>mlp.fc</font> and attention matrices, "
        "that confirms the mechanistic premise. If they don't, reconsider the locus before burning a smoke run.",
        "body"
    ))
    story.append(P(
        "20-line script: load artifact → extract <font face='Courier'>scales</font> dict → "
        "compute <font face='Courier'>std(scales) / mean(scales)</font> per tensor → rank by CV. "
        "The hypothesis predicts <font face='Courier'>mlp.proj</font> will have the highest coefficient of variation.",
        "body"
    ))

    story.append(SP(8))
    story.append(HR())

    # ── 8. WHAT THIS IS NOT ──
    not_section = [
        P("8. What this hypothesis is NOT", "h1"),
        P("&#x2718;&nbsp; Not a global capacity change — param count is held constant within ±0.02%", "body_tight"),
        P("&#x2718;&nbsp; Not a training trick — no LR change, no schedule change, no regularizer", "body_tight"),
        P("&#x2718;&nbsp; Not a thousandths shave — the mechanism targets the largest tensor family "
          "and changes the distribution shape, which can move by hundredths", "body_tight"),
        P("&#x2718;&nbsp; Not another control tensor — zero new fp16 passthrough cost", "body_tight"),
        P("&#x2718;&nbsp; Not speculative — the premise (one-sided inputs quantize worse) is directly verifiable "
          "from the existing artifact before running anything", "body_tight"),
    ]
    story.append(KeepTogether(not_section))

    story.append(SP(12))
    story.append(HR())
    story.append(P(
        "Generated 2026-03-22  ·  Parameter Golf Hypothesis Worksheet  ·  "
        "Confirm the locus → run the smoke → read the number → decide",
        "footer"
    ))

    doc.build(story)
    print(f"PDF written to: {OUT_PATH}")
    print(f"Size: {os.path.getsize(OUT_PATH):,} bytes")


if __name__ == "__main__":
    build_pdf()
