"""
SecureNet Training Curve Plotter
=================================
Reads checkpoints/*.json and produces one graph per difficulty tier,
plus a combined overview chart. Saves all as PNG files.
"""

import os, json, sys
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")          # headless / no GUI needed
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    print("matplotlib not found. Installing...")
    os.system(f"{sys.executable} -m pip install matplotlib -q")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

# ── Config ────────────────────────────────────────────────────────────
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR     = "checkpoints"
TIERS          = ["easy", "medium", "hard", "critical", "nightmare"]
SMOOTH_WIN     = 100   # rolling-average window

TIER_COLORS = {
    "easy":      "#00c896",   # teal-green
    "medium":    "#3b9eff",   # sky-blue
    "hard":      "#f5a623",   # amber
    "critical":  "#e74c3c",   # red
    "nightmare": "#9b59b6",   # purple
}

# ── Helpers ───────────────────────────────────────────────────────────
def rolling(arr, w):
    out = []
    for i in range(len(arr)):
        chunk = arr[max(0, i - w + 1): i + 1]
        out.append(np.mean(chunk))
    return np.array(out)

def load_tier(tier):
    path = os.path.join(CHECKPOINT_DIR, f"{tier}.json")
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        d = json.load(f)
    return np.array(d.get("rewards", [])), np.array(d.get("scores", []))

def style_ax(ax, title, ylabel, color):
    ax.set_facecolor("#0d1117")
    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel("Episode", color="#aaaaaa", fontsize=10)
    ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=10)
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.grid(True, color="#1e2530", linewidth=0.6, linestyle="--")
    ax.axhline(0.6, color=color, linewidth=0.8, linestyle=":", alpha=0.5, label="0.6 threshold")

# ── Individual plots ──────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

for tier in TIERS:
    rewards, scores = load_tier(tier)
    if rewards is None:
        print(f"  [{tier}] No data found, skipping.")
        continue

    n   = len(scores)
    eps = np.arange(1, n + 1)
    color = TIER_COLORS[tier]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle(
        f"SecureNet DQN — {tier.upper()} Tier  ({n} episodes)",
        color="white", fontsize=15, fontweight="bold", y=0.98
    )

    # ── Top panel: Containment Score ──────────────────────────────────
    style_ax(ax1, "Containment Score", "Score (0–1)", color)
    ax1.fill_between(eps, scores, alpha=0.15, color=color)
    ax1.plot(eps, scores,           color=color,   alpha=0.25, linewidth=0.4, label="Raw")
    ax1.plot(eps, rolling(scores, SMOOTH_WIN), color=color, linewidth=2.0,  label=f"MA-{SMOOTH_WIN}")

    # Win-rate bands
    win_rate_100 = [
        np.mean(scores[max(0, i - 100): i]) if i >= 1 else 0
        for i in range(1, n + 1)
    ]
    ax1.fill_between(eps, win_rate_100, 0, alpha=0.06, color=color)

    # Annotate last-500 average
    last500 = np.mean(scores[-500:]) if n >= 500 else np.mean(scores)
    ax1.annotate(
        f"Last-500 avg: {last500:.3f}",
        xy=(n, last500), xytext=(max(1, n * 0.75), min(last500 + 0.1, 0.95)),
        color="white", fontsize=9, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#aaaaaa", lw=1),
    )
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="upper left", facecolor="#1a1f2e", edgecolor="#333", labelcolor="white", fontsize=8)

    # ── Bottom panel: Episode Reward ───────────────────────────────────
    style_ax(ax2, "Episode Reward", "Total Reward", color)
    ax2.fill_between(eps, rewards, alpha=0.12, color=color)
    ax2.plot(eps, rewards,                  color=color, alpha=0.2, linewidth=0.4, label="Raw")
    ax2.plot(eps, rolling(rewards, SMOOTH_WIN), color=color, linewidth=2.0, label=f"MA-{SMOOTH_WIN}")
    ax2.legend(loc="upper left", facecolor="#1a1f2e", edgecolor="#333", labelcolor="white", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(OUTPUT_DIR, f"{tier}_training.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  [OK] Saved -> {out_path}")

# -- Combined overview chart --------------------------------------------
print("\n  Generating combined overview...")

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor("#0d1117")
fig.suptitle("SecureNet DQN — All Tiers Training Overview", color="white",
             fontsize=17, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.3)

axes_score  = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
               fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 0]),
               fig.add_subplot(gs[1, 1])]

# Combined comparison on last slot
ax_cmp = fig.add_subplot(gs[1, 2])
ax_cmp.set_facecolor("#0d1117")
ax_cmp.set_title("All Tiers — MA Score Comparison", color="white", fontsize=11, fontweight="bold", pad=8)
ax_cmp.set_xlabel("Episode", color="#aaaaaa", fontsize=9)
ax_cmp.set_ylabel("Score (MA-100)", color="#aaaaaa", fontsize=9)
ax_cmp.tick_params(colors="#aaaaaa")
for spine in ax_cmp.spines.values():
    spine.set_edgecolor("#333333")
ax_cmp.grid(True, color="#1e2530", linewidth=0.6, linestyle="--")
ax_cmp.axhline(0.6, color="white", linewidth=0.7, linestyle=":", alpha=0.4)

for ax, tier in zip(axes_score, TIERS):
    rewards, scores = load_tier(tier)
    color = TIER_COLORS[tier]
    if rewards is None:
        ax.set_facecolor("#0d1117")
        ax.set_title(f"{tier.upper()} — No Data", color="#555", fontsize=10)
        continue

    n   = len(scores)
    eps = np.arange(1, n + 1)
    ma  = rolling(scores, SMOOTH_WIN)

    style_ax(ax, tier.upper(), "Score", color)
    ax.fill_between(eps, scores, alpha=0.10, color=color)
    ax.plot(eps, scores, color=color, alpha=0.20, linewidth=0.5)
    ax.plot(eps, ma,     color=color, linewidth=1.8, label=f"MA-{SMOOTH_WIN}")
    ax.set_ylim(-0.05, 1.05)

    last500 = np.mean(scores[-500:]) if n >= 500 else np.mean(scores)
    ax.text(0.97, 0.07, f"L500: {last500:.2f}", transform=ax.transAxes,
            color="white", fontsize=8, ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1f2e", edgecolor=color, alpha=0.8))
    ax.legend(loc="upper left", facecolor="#1a1f2e", edgecolor="#333", labelcolor="white", fontsize=7)

    # Add to comparison chart
    ax_cmp.plot(eps, ma, color=color, linewidth=1.8, label=tier.capitalize())
    ax_cmp.tick_params(colors="#aaaaaa")

ax_cmp.legend(facecolor="#1a1f2e", edgecolor="#333", labelcolor="white", fontsize=8)

overview_path = os.path.join(OUTPUT_DIR, "all_tiers_overview.png")
plt.savefig(overview_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"  [OK] Saved -> {overview_path}")
print("\n  Done! All graphs saved to checkpoints/")
