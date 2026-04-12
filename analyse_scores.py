"""
SecureNet RL Score Analyser
============================
Reads all per-difficulty checkpoint JSON files and produces a full
diagnostic report: mean, peak, trend, plateau detection, and a
concise recommendation for each tier.
"""

import json, math, os, statistics

CHECKPOINTS = {
    "easy":      "securenet_env/checkpoints/easy.json",
    "medium":    "securenet_env/checkpoints/medium.json",
    "hard":      "securenet_env/checkpoints/hard.json",
    "critical":  "securenet_env/checkpoints/critical.json",
    "nightmare": "securenet_env/checkpoints/nightmare.json",
}

WINDOW = 200   # rolling window for trend detection
SUCCESS = 0.60 # threshold to count as "solved"

def rolling(scores, w):
    out = []
    for i in range(w, len(scores)+1, w):
        out.append(round(statistics.mean(scores[max(0,i-w):i]), 4))
    return out

def linear_trend(vals):
    """Least-squares slope across values."""
    if len(vals) < 2:
        return 0.0
    n = len(vals)
    xs = list(range(n))
    mx = sum(xs)/n; my = sum(vals)/n
    num = sum((x-mx)*(y-my) for x,y in zip(xs,vals))
    den = sum((x-mx)**2 for x in xs)
    return num/den if den else 0.0

def plateau_start(vals, tol=0.005):
    """Return the index where rolling avg stopped improving (within tol)."""
    for i in range(len(vals)-1, 0, -1):
        if vals[i] - vals[i-1] > tol:
            return i
    return 0

def load(path):
    with open(path) as f:
        d = json.load(f)
    return d["scores"]

print("\n" + "="*90)
print("  SECURENET RL  —  SCORE ANALYSIS REPORT  (5000 ep per difficulty)")
print("="*90)

results = {}
for diff, path in CHECKPOINTS.items():
    if not os.path.exists(path):
        print(f"\n  {diff.upper():10s}: FILE NOT FOUND — {path}")
        continue

    sc    = load(path)
    n     = len(sc)
    roll  = rolling(sc, WINDOW)
    slope = linear_trend(roll)
    pct60 = 100 * sum(1 for s in sc if s >= SUCCESS) / n
    best  = max(sc)
    mean  = statistics.mean(sc)
    q_early = statistics.mean(sc[:n//5])
    q_late  = statistics.mean(sc[4*n//5:])
    gain    = q_late - q_early
    plat_i  = plateau_start(roll)
    plat_ep = plat_i * WINDOW

    # Status
    if q_late >= 0.65:
        status = "MASTERED"
    elif q_late >= 0.50:
        status = "LEARNING"
    elif gain > 0.02:
        status = "IMPROVING"
    elif abs(gain) <= 0.01:
        status = "PLATEAU"
    else:
        status = "REGRESSING"

    results[diff] = dict(n=n, mean=mean, best=best, pct60=pct60,
                         q_early=q_early, q_late=q_late, gain=gain,
                         slope=slope, plat_ep=plat_ep, roll=roll, status=status)

    bar_e = "#" * int(q_early * 40)
    bar_l = "#" * int(q_late  * 40)

    print(f"\n  {diff.upper():10s} [{status}]  n={n}")
    print(f"    Overall mean : {mean:.3f}    Peak episode score: {best:.3f}")
    print(f"    Success>=0.6 : {pct60:.1f}%  ({int(n*pct60/100)} / {n} episodes)")
    print(f"    Early avg    : {q_early:.3f}  |{bar_e}")
    print(f"    Late  avg    : {q_late:.3f}  |{bar_l}")
    print(f"    Net gain     : {gain:+.3f}    Slope: {slope:+.5f}")
    print(f"    Rolling avgs : {roll}")
    if plat_ep > 0:
        print(f"    Last improvement at ep ~{plat_ep}")

# ── Summary table ────────────────────────────────────────────────────────────
print("\n" + "="*90)
print("  SUMMARY TABLE")
print(f"  {'Tier':12s} {'Mean':>7s} {'Late avg':>9s} {'Gain':>8s} {'Success%':>9s} {'Status':>12s}")
print("  " + "-"*60)
for diff, r in results.items():
    print(f"  {diff:12s} {r['mean']:7.3f} {r['q_late']:9.3f} {r['gain']:+8.3f} {r['pct60']:9.1f}%  {r['status']:>12s}")

# ── Recommendations ──────────────────────────────────────────────────────────
print("\n" + "="*90)
print("  RECOMMENDATIONS")
print("="*90)

rec_map = {
    "MASTERED":   ("green",  "Keep as-is. Load checkpoint as baseline for the next tier."),
    "LEARNING":   ("yellow", "Continue training — still improving. Run 3000 more episodes."),
    "IMPROVING":  ("yellow", "Slow but positive trend. Run 2000 more episodes with lower LR."),
    "PLATEAU":    ("red",    "Stuck. Apply reward shaping boost + lower eps_end + more buffer."),
    "REGRESSING": ("red",    "Score declining. Clear checkpoint and retrain from scratch with lower LR."),
}

actions = []
for diff, r in results.items():
    colour, advice = rec_map[r["status"]]
    print(f"\n  [{diff.upper()}]  Status={r['status']}")
    print(f"    Late avg={r['q_late']:.3f}  Gain={r['gain']:+.3f}  Slope={r['slope']:+.5f}")
    print(f"    => {advice}")
    actions.append((diff, r["status"], r["q_late"], r["gain"]))

# Decide what to actually run next
print("\n" + "="*90)
print("  ACTION PLAN (ordered by priority)")
print("="*90)
priority = sorted(actions, key=lambda x: x[2])  # lowest late-avg first
for i, (diff, status, late, gain) in enumerate(priority, 1):
    if status in ("PLATEAU", "REGRESSING"):
        act = "RETRAIN with improved hyperparams"
        extra_ep = 5000
    elif status == "IMPROVING":
        act = "EXTEND training"
        extra_ep = 2000
    elif status == "LEARNING":
        act = "EXTEND training"
        extra_ep = 3000
    else:
        act = "DONE — use checkpoint"
        extra_ep = 0
    print(f"  {i}. {diff:10s} late={late:.3f}  -> {act}  (+{extra_ep} ep)")

print()
