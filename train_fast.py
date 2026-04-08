"""
SecureNet RL Fast Training — v2 (No HTTP, Optimised)
======================================
Bypasses the HTTP server entirely — calls SecureNetEnvironment in-process.
Expected speedup: 50-100x vs train.py (HTTP version).

Estimated timing:
  HTTP version : ~43s/episode → 2000 eps ≈ 24 hours
  Fast version : ~0.3s/episode → 2000 eps ≈ 10 minutes

Usage (from repo root):
  python securenet_env/train_fast.py
  python securenet_env/train_fast.py --episodes 3000 --difficulty_schedule progressive
"""

import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ── Import environment DIRECTLY (no HTTP) ─────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "server"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from securenet_environment import SecureNetEnvironment

# ─────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────
DEFAULTS = dict(
    episodes             = 3000,
    lr                   = 3e-4,
    gamma                = 0.99,
    entropy_coeff        = 0.03,
    log_interval         = 100,
    checkpoint           = "securenet_env/checkpoints/policy_fast.pt",
    stats_out            = "securenet_env/checkpoints/stats.json",   # shared with HTTP version (feeds dashboard)
    difficulty_schedule  = "progressive",
)

# Lowered thresholds — REINFORCE with large action spaces converges slowly
CURRICULUM = [
    ("easy",      "medium",    0.45),
    ("medium",    "hard",      0.50),
    ("hard",      "critical",  0.60),
    ("critical",  "nightmare", 0.70),
]

# ─────────────────────────────────────────────────────────────────────
# VOCABULARY (must match train.py)
# ─────────────────────────────────────────────────────────────────────
VOCAB = [
    "workstation","server","database","laptop","firewall","backup","node","router","switch",
    "pipeline","kubernetes","k8s","cicd","hsm","siem","edr","vpn","dmz","webserver",
    "ssh","powershell","ransomware","malware","mimikatz","lockbit","psexec","webshell",
    "cryptominer","ntds","lsass","kerberoast","gpo","exploit","rce","injection","keylogger",
    "failed","suspicious","alert","critical","anomal","infected","compromised","lateral",
    "movement","credential","harvest","encryption","c2","phishing","payload","exfil",
    "outbound","spike","beacon","backdoor","brute","force","shadow","delete","encoded",
    "healthy","clean","isolated","contained","blocked","quarantine","normal","standard",
    "supply","chain","firmware","poisoned","artifact","build","deploy",
    "10.0.0","192.168","172.16","185.220","193.56","45.33","91.92","45.142","update-cdn",
    "t1486","t1110","t1059","t1003","t1190","t1505","t1610","t1195","t1568","t1041",
    "easy","medium","hard","critical","nightmare","initialized","curriculum","kill","chain",
    "reconnaissance","initial","access","persistence","exfiltration","impact",
    "sla","violation","catastrophic","failure","timeout","score","grader",
]
VOCAB_SIZE = len(VOCAB)

# ─────────────────────────────────────────────────────────────────────
# COMPLETE ACTION SPACE
# ─────────────────────────────────────────────────────────────────────
ALL_NODES = [
    "Workstation-A", "Server-1",
    "Workstation-B", "Database-Primary", "Mail-Server", "HR-Workstation",
    "Node-1", "Node-2", "Node-3", "CEO-Laptop", "Firewall-GW", "Backup-Server",
    "DMZ-WebServer", "AD-Domain-Controller", "Finance-Workstation",
    "EDR-Server", "VPN-Gateway", "SIEM-Platform",
    "Internet-Router", "Core-Switch", "Prod-DB-Primary", "K8s-Master",
    "CI-CD-Pipeline", "HSM-Server", "Backup-Vault", "CISO-Workstation",
]
ALL_IPS = [
    "10.0.0.99", "45.33.32.156", "172.16.0.200",
    "185.220.101.45", "193.56.29.11", "45.142.212.100", "91.92.109.200",
]
ALL_IOCS = ALL_IPS + ["a3f4b9c1d2e5", "deadbeef1234", "cafebabe5678", "update-cdn.ru", "telemetry.xyz"]
# Pruned: only the most plausible node→process pairs to shrink action space
# from 338 quarantine combos → 13 targeted ones (one per process at likely host)
QUARANTINE_PAIRS = [
    ("Server-1",             "/tmp/.x (malicious)"),
    ("Workstation-B",        "mimikatz.exe"),
    ("Node-1",               "LockBit3.exe"),
    ("Node-1",               "PsExec.exe"),
    ("DMZ-WebServer",        "webshell.php (malicious)"),
    ("AD-Domain-Controller", "ntds-dumper.exe (malicious)"),
    ("Backup-Server",        "svc_update (malicious)"),
    ("K8s-Master",           "cryptominer (malicious)"),
    ("Prod-DB-Primary",      "xtrabackup (malicious)"),
    ("Finance-Workstation",  "keylogger.dll (malicious)"),
    ("Mail-Server",          "/bin/sh -i (malicious)"),
    ("CI-CD-Pipeline",       "malicious-build-step.sh"),
    ("Prod-DB-Primary",      "python3 (c2-agent)"),
]

# 5-tuple: (action_type, target_node, ip, ioc, process_name)
ACTIONS = (
    [("query_logs",         n,    None, None, None) for n in ALL_NODES] +
    [("analyze_process",    n,    None, None, None) for n in ALL_NODES] +
    [("isolate_host",       n,    None, None, None) for n in ALL_NODES] +
    [("block_ip",           None, ip,   None, None) for ip in ALL_IPS]  +
    [("threat_intel",       None, None, ioc,  None) for ioc in ALL_IOCS] +
    [("quarantine_process", n,    None, None, proc) for n, proc in QUARANTINE_PAIRS]
)
ACTION_SIZE = len(ACTIONS)


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────
def obs_to_tensor(obs_text: str) -> torch.Tensor:
    lo = obs_text.lower()
    v  = torch.zeros(VOCAB_SIZE, dtype=torch.float32)
    for i, w in enumerate(VOCAB):
        if w in lo:
            v[i] = 1.0
    return v


def compute_returns(rewards, gamma):
    G, ret = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        ret.insert(0, G)
    t = torch.tensor(ret, dtype=torch.float32)
    if t.std() > 1e-6:
        t = (t - t.mean()) / (t.std() + 1e-8)
    return t


# ─────────────────────────────────────────────────────────────────────
# POLICY NETWORK
# ─────────────────────────────────────────────────────────────────────
class PolicyNetwork(nn.Module):
    def __init__(self, in_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256),    nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128),    nn.ReLU(),
            nn.Linear(128, act_dim),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


# ─────────────────────────────────────────────────────────────────────
# EPISODE RUNNER (direct env call — no HTTP)
# ─────────────────────────────────────────────────────────────────────
def run_episode(policy: PolicyNetwork, env: SecureNetEnvironment, difficulty: str):
    log_probs, rewards, entropies = [], [], []
    final_score = 0.0

    resp     = env.reset(difficulty)
    obs_text = resp.get("result", "")
    done     = resp.get("done", False)

    while not done:
        state = obs_to_tensor(obs_text)
        probs = policy(state)
        dist  = Categorical(probs)
        idx   = dist.sample()

        log_probs.append(dist.log_prob(idx))
        entropies.append(dist.entropy())

        act_type, node, ip, ioc, proc = ACTIONS[idx.item()]
        resp = env.step(
            action_type=act_type,
            target_node=node,
            ip_address=ip,
            ioc=ioc,
            process_name=proc,
        )
        rewards.append(resp.get("reward", 0.0))
        obs_text = resp.get("result", "")
        done     = resp.get("done", False)

        if "total_score" in resp.get("info", {}):
            final_score = resp["info"]["total_score"]

    return log_probs, rewards, entropies, final_score


# ─────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────
def train(args):
    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)

    env    = SecureNetEnvironment()
    policy = PolicyNetwork(VOCAB_SIZE, ACTION_SIZE)
    opt    = optim.Adam(policy.parameters(), lr=args.lr)
    sched  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.episodes, eta_min=1e-5)

    # Entropy annealing: start exploratory (0.08), decay to exploitation (0.005)
    ent_start = 0.08
    ent_end   = 0.005

    start_ep   = 1
    difficulty = "easy" if args.difficulty_schedule == "progressive" else args.difficulty_schedule
    all_rewards, all_scores, score_history = [], [], []
    eps_times: list = []

    # Resume
    if os.path.exists(args.checkpoint):
        try:
            ckpt       = torch.load(args.checkpoint, weights_only=True)
            policy.load_state_dict(ckpt["policy"])
            opt.load_state_dict(ckpt["optimizer"])
            start_ep   = ckpt.get("episode", 1) + 1
            difficulty = ckpt.get("difficulty", "easy")
            print(f"▶ Resumed from ep {start_ep - 1} (difficulty={difficulty})")
        except Exception as e:
            print(f"⚠ Could not load checkpoint: {e}  — starting fresh.")

    print("\n" + "═" * 72)
    print("  ⚡ SecureNet FAST Training (in-process, no HTTP overhead)  ")
    print("═" * 72)
    print(f"  Policy params  : {sum(p.numel() for p in policy.parameters()):,}")
    print(f"  Action space   : {ACTION_SIZE}")
    print(f"  Vocabulary     : {VOCAB_SIZE}")
    print(f"  Difficulty     : {args.difficulty_schedule}")
    print(f"  Episodes       : {args.episodes}")
    print("═" * 72 + "\n")

    t_total = time.time()

    for ep in range(start_ep, args.episodes + 1):
        t_ep = time.time()

        # Curriculum advancement
        if args.difficulty_schedule == "progressive" and len(score_history) >= 10:
            rolling = sum(score_history[-10:]) / 10
            for cur, nxt, thresh in CURRICULUM:
                if difficulty == cur and rolling >= thresh:
                    difficulty = nxt
                    print(f"\n  🎓 CURRICULUM → {nxt.upper()}  (rolling avg = {rolling:.3f})\n")
                    break

        log_probs, rewards, entropies, score = run_episode(policy, env, difficulty)
        returns = compute_returns(rewards, args.gamma)

        # Entropy annealing: linearly decay from ent_start → ent_end
        progress     = (ep - start_ep) / max(args.episodes - start_ep, 1)
        entropy_coeff = ent_start + (ent_end - ent_start) * progress

        pg_loss  = torch.stack([-lp * G for lp, G in zip(log_probs, returns)]).sum()
        ent_loss = -entropy_coeff * torch.stack(entropies).mean()
        loss     = pg_loss + ent_loss

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        sched.step()

        ep_reward = sum(rewards)
        all_rewards.append(ep_reward)
        all_scores.append(score)
        score_history.append(score)
        eps_times.append(time.time() - t_ep)

        if ep % args.log_interval == 0:
            avg_r   = np.mean(all_rewards[-args.log_interval:])
            avg_s   = np.mean(all_scores[-args.log_interval:])
            avg_t   = np.mean(eps_times[-args.log_interval:])
            elapsed = time.time() - t_total
            eta     = avg_t * (args.episodes - ep)
            print(
                f"Ep {ep:>5}/{args.episodes} │ "
                f"AvgReward {avg_r:>+7.3f} │ "
                f"AvgScore {avg_s:.3f} │ "
                f"Tier {difficulty:<9} │ "
                f"Ent {entropy_coeff:.4f} │ "
                f"{avg_t:.2f}s/ep │ "
                f"ETA {eta/60:.0f}min"
            )

        # Checkpoint + write stats for live dashboard
        if ep % 100 == 0:
            torch.save({
                "policy":     policy.state_dict(),
                "optimizer":  opt.state_dict(),
                "episode":    ep,
                "difficulty": difficulty,
            }, args.checkpoint)
            with open(args.stats_out, "w") as f:
                json.dump({
                    "rewards":    all_rewards,
                    "scores":     all_scores,
                    "difficulty": difficulty,
                }, f)

    # Final save
    torch.save({"policy": policy.state_dict(), "optimizer": opt.state_dict(),
                "episode": args.episodes, "difficulty": difficulty}, args.checkpoint)
    print(f"\n✅ Training complete! — {args.checkpoint}")
    print(f"   Total time : {(time.time() - t_total)/60:.1f} min")
    print(f"   Final Avg Score (last {args.log_interval} ep): "
          f"{np.mean(all_scores[-args.log_interval:]):.3f}")

    # Save chart
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), facecolor="#030d0a")
        for ax in (ax1, ax2):
            ax.set_facecolor("#061410")
            ax.tick_params(colors="#2a5040")
            ax.spines[:].set_color("#0d2b22")
        W = 100
        roll = lambda a: np.convolve(a, np.ones(W)/W, mode="valid")

        ax1.plot(all_rewards, alpha=0.2, color="#ff3050", lw=.7, label="Raw")
        if len(all_rewards) >= W:
            ax1.plot(range(W-1, len(all_rewards)), roll(all_rewards),
                     color="#ff3050", lw=2.2, label=f"Roll({W})")
        ax1.set_title("Episode Reward", color="#b0d8c8", fontsize=12)
        ax1.legend(framealpha=0.2, labelcolor="#b0d8c8")

        ax2.plot(all_scores, alpha=0.2, color="#00ff88", lw=.7, label="Raw")
        if len(all_scores) >= W:
            ax2.plot(range(W-1, len(all_scores)), roll(all_scores),
                     color="#00ff88", lw=2.2, label=f"Roll({W})")
        ax2.set_title("Containment Score", color="#b0d8c8", fontsize=12)
        ax2.set_ylim(0, 1.05)
        ax2.legend(framealpha=0.2, labelcolor="#b0d8c8")

        plt.tight_layout(pad=2.5)
        path = "securenet_env/checkpoints/training_curves.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"   Chart saved → {path}")
    except ImportError:
        pass


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SecureNet Fast RL Training (no HTTP)")
    p.add_argument("--episodes",            type=int,   default=DEFAULTS["episodes"])
    p.add_argument("--lr",                  type=float, default=DEFAULTS["lr"])
    p.add_argument("--gamma",               type=float, default=DEFAULTS["gamma"])
    p.add_argument("--entropy_coeff",       type=float, default=DEFAULTS["entropy_coeff"])  # unused, kept for CLI compat
    p.add_argument("--log_interval",        type=int,   default=DEFAULTS["log_interval"])
    p.add_argument("--checkpoint",          type=str,   default=DEFAULTS["checkpoint"])
    p.add_argument("--stats_out",           type=str,   default=DEFAULTS["stats_out"])
    p.add_argument("--difficulty_schedule", type=str,   default=DEFAULTS["difficulty_schedule"])
    train(p.parse_args())
