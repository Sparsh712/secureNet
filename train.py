"""
SecureNet RL Training Script — v2 (Full Functionality)
=======================================================
Algorithm   : REINFORCE with entropy regularization + baseline
Policy      : Extended BoW → 3-layer MLP → Softmax
Action space: All 6 action types × all nodes + all IOCs + process names
Curriculum  : easy → medium → hard → critical → nightmare
"""

import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client import SecureNetClient

# ─────────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────
DEFAULTS = dict(
    episodes             = 2000,
    lr                   = 3e-4,
    gamma                = 0.99,
    entropy_coeff        = 0.02,
    log_interval         = 50,
    checkpoint           = "securenet_env/checkpoints/policy.pt",
    stats_out            = "securenet_env/checkpoints/stats.json",
    difficulty_schedule  = "progressive",
)

CURRICULUM_THRESHOLDS = [
    ("easy",      "medium",    0.65),
    ("medium",    "hard",      0.70),
    ("hard",      "critical",  0.75),
    ("critical",  "nightmare", 0.80),
]

# ─────────────────────────────────────────────────────────────
# VOCABULARY — Extended for all 5 tiers
# ─────────────────────────────────────────────────────────────
VOCAB = [
    # Node types
    "workstation","server","database","laptop","firewall","backup","node","router","switch",
    "pipeline","kubernetes","k8s","cicd","hsm","siem","edr","vpn","dmz","webserver",
    # Attack keywords
    "ssh","powershell","ransomware","malware","mimikatz","lockbit","psexec","webshell",
    "cryptominer","ntds","lsass","kerberoast","gpo","exploit","rce","injection","keylogger",
    # Alert keywords
    "failed","suspicious","alert","critical","anomal","infected","compromised","lateral",
    "movement","credential","harvest","encryption","c2","phishing","payload","exfil",
    "outbound","spike","beacon","backdoor","brute","force","shadow","delete","encoded",
    # Context keywords
    "healthy","clean","isolated","contained","blocked","quarantine","normal","standard",
    "supply","chain","firmware","backdoor","poisoned","artifact","build","deploy",
    # IPs / domains (partial matches)
    "10.0.0","192.168","172.16","185.220","193.56","45.33","91.92","45.142","update-cdn",
    # MITRE keywords
    "t1486","t1110","t1059","t1003","t1190","t1505","t1610","t1195","t1568","t1041",
    # Difficulty / meta
    "easy","medium","hard","critical","nightmare","initialized","curriculum","kill","chain",
    "reconnaissance","initial","access","persistence","exfiltration","impact",
    # Reward signal words
    "sla","violation","catastrophic","failure","timeout","score","grader",
]
VOCAB_SIZE = len(VOCAB)

# ─────────────────────────────────────────────────────────────
# COMPLETE ACTION SPACE (covers all 5 difficulty tiers)
# ─────────────────────────────────────────────────────────────
ALL_NODES = [
    # EASY
    "Workstation-A", "Server-1",
    # MEDIUM
    "Workstation-B", "Database-Primary", "Mail-Server", "HR-Workstation",
    # HARD
    "Node-1", "Node-2", "Node-3", "CEO-Laptop", "Firewall-GW", "Backup-Server",
    # CRITICAL
    "DMZ-WebServer", "AD-Domain-Controller", "Finance-Workstation",
    "EDR-Server", "VPN-Gateway", "SIEM-Platform",
    # NIGHTMARE
    "Internet-Router", "Core-Switch", "Prod-DB-Primary", "K8s-Master",
    "CI-CD-Pipeline", "HSM-Server", "Backup-Vault", "CISO-Workstation",
]

ALL_IPS = [
    "10.0.0.99", "45.33.32.156",
    "172.16.0.200", "185.220.101.45",
    "193.56.29.11",
    "45.142.212.100", "91.92.109.200",
]

ALL_IOCS = ALL_IPS + ["a3f4b9c1d2e5", "deadbeef1234", "cafebabe5678", "update-cdn.ru", "telemetry.xyz"]

SUSPICIOUS_PROCS = [
    "/tmp/.x (malicious)", "mimikatz.exe", "LockBit3.exe", "PsExec.exe",
    "webshell.php (malicious)", "ntds-dumper.exe (malicious)", "svc_update (malicious)",
    "cryptominer (malicious)", "xtrabackup (malicious)", "keylogger.dll (malicious)",
    "/bin/sh -i (malicious)", "malicious-build-step.sh", "python3 (c2-agent)",
]

# Build flat action list: (action_type, target_node, ip, ioc, process_name)
ACTIONS = (
    [("query_logs",         n,    None, None,       None)  for n in ALL_NODES] +
    [("analyze_process",    n,    None, None,       None)  for n in ALL_NODES] +
    [("isolate_host",       n,    None, None,       None)  for n in ALL_NODES] +
    [("block_ip",           None, ip,   None,       None)  for ip in ALL_IPS]  +
    [("threat_intel",       None, None, ioc,        None)  for ioc in ALL_IOCS] +
    [("quarantine_process", n,    None, None,       proc)
        for n in ALL_NODES for proc in SUSPICIOUS_PROCS]
)
ACTION_SIZE = len(ACTIONS)


# ─────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────
def obs_to_tensor(obs_text: str) -> torch.Tensor:
    obs_lower = obs_text.lower()
    vec = torch.zeros(VOCAB_SIZE, dtype=torch.float32)
    for i, w in enumerate(VOCAB):
        if w in obs_lower:
            vec[i] = 1.0
    return vec


# ─────────────────────────────────────────────────────────────
# POLICY NETWORK
# ─────────────────────────────────────────────────────────────
class PolicyNetwork(nn.Module):
    def __init__(self, in_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)


def compute_returns(rewards: list, gamma: float) -> torch.Tensor:
    G, ret = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        ret.insert(0, G)
    t = torch.tensor(ret, dtype=torch.float32)
    if t.std() > 1e-6:
        t = (t - t.mean()) / (t.std() + 1e-8)
    return t


# ─────────────────────────────────────────────────────────────
# EPISODE RUNNER
# ─────────────────────────────────────────────────────────────
def run_episode(policy: PolicyNetwork, client: SecureNetClient, difficulty: str):
    log_probs, rewards, entropies = [], [], []
    final_score = 0.0

    resp     = client.reset(difficulty)
    obs_text = resp.get("result", "")
    done     = resp.get("done", False)

    while not done:
        state = obs_to_tensor(obs_text)
        probs = policy(state)
        dist  = Categorical(probs)
        idx   = dist.sample()

        log_probs.append(dist.log_prob(idx))
        entropies.append(dist.entropy())

        # Unpack 5-tuple action
        act_type, node, ip, ioc, proc = ACTIONS[idx.item()]
        resp = client.step(act_type, node, ip, process_name=proc, ioc=ioc)

        rewards.append(resp.get("reward", 0.0))
        obs_text = resp.get("result", "")
        done     = resp.get("done", False)

        if "total_score" in resp.get("info", {}):
            final_score = resp["info"]["total_score"]

    return log_probs, rewards, entropies, final_score


# ─────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────
def train(args):
    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)

    client = SecureNetClient()
    # Verify server is up
    try:
        client.health()
    except RuntimeError as e:
        print(f"❌ {e}")
        sys.exit(1)

    policy = PolicyNetwork(VOCAB_SIZE, ACTION_SIZE)
    opt    = optim.Adam(policy.parameters(), lr=args.lr)
    sched  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.episodes, eta_min=1e-5)

    start_ep   = 1
    difficulty = "easy" if args.difficulty_schedule == "progressive" else args.difficulty_schedule
    all_rewards, all_scores, score_history = [], [], []

    # Resume from checkpoint
    if os.path.exists(args.checkpoint):
        try:
            ckpt       = torch.load(args.checkpoint, weights_only=True)
            policy.load_state_dict(ckpt["policy"])
            opt.load_state_dict(ckpt["optimizer"])
            start_ep   = ckpt.get("episode", 1) + 1
            difficulty = ckpt.get("difficulty", "easy")
            print(f"▶ Resumed from ep {start_ep - 1}, difficulty={difficulty}")
        except Exception as e:
            print(f"⚠ Could not load checkpoint ({e}), starting fresh.")

    print("\n" + "═" * 70)
    print("  🛡  SecureNet SOC — REINFORCE + Entropy + 5-Tier Curriculum  ")
    print("═" * 70)
    print(f"  Policy params  : {sum(p.numel() for p in policy.parameters()):,}")
    print(f"  Action space   : {ACTION_SIZE}")
    print(f"  Vocabulary     : {VOCAB_SIZE}")
    print(f"  Difficulty     : {args.difficulty_schedule}")
    print(f"  Episodes       : {args.episodes}")
    print("═" * 70 + "\n")

    t0 = time.time()

    for ep in range(start_ep, args.episodes + 1):

        # Progressive curriculum
        if args.difficulty_schedule == "progressive" and len(score_history) >= 10:
            rolling_avg = sum(score_history[-10:]) / 10
            for cur, nxt, thresh in CURRICULUM_THRESHOLDS:
                if difficulty == cur and rolling_avg >= thresh:
                    difficulty = nxt
                    print(f"  🎓 Curriculum → {nxt.upper()} (rolling avg={rolling_avg:.3f})")
                    break

        log_probs, rewards, entropies, score = run_episode(policy, client, difficulty)
        returns = compute_returns(rewards, args.gamma)

        pg_loss  = torch.stack([-lp * G for lp, G in zip(log_probs, returns)]).sum()
        ent_loss = -args.entropy_coeff * torch.stack(entropies).mean()
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

        if ep % args.log_interval == 0:
            avg_r    = np.mean(all_rewards[-args.log_interval:])
            avg_s    = np.mean(all_scores[-args.log_interval:])
            elapsed  = time.time() - t0
            print(
                f"Ep {ep:>5}/{args.episodes} │ "
                f"AvgReward {avg_r:>+7.3f} │ "
                f"AvgScore {avg_s:.3f} │ "
                f"Tier {difficulty:<9} │ "
                f"Loss {loss.item():>8.4f} │ "
                f"Elapsed {elapsed:>6.0f}s"
            )

        # Save checkpoint every 100 episodes
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
            print(f"  💾 Checkpoint saved at ep {ep}.")

    # Final save
    torch.save({"policy": policy.state_dict(), "optimizer": opt.state_dict(),
                "episode": args.episodes, "difficulty": difficulty}, args.checkpoint)
    print(f"\n✅ Training complete! — {args.checkpoint}")
    print(f"   Final Avg Score (last {args.log_interval} ep): "
          f"{np.mean(all_scores[-args.log_interval:]):.3f}")

    # Save training chart
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), facecolor="#030d0a")
        for ax in (ax1, ax2):
            ax.set_facecolor("#061410")
            ax.tick_params(colors="#2a5040")
            ax.spines[:].set_color("#0d2b22")
        W = 50
        roll = lambda a: np.convolve(a, np.ones(W)/W, mode="valid")

        ax1.plot(all_rewards, alpha=0.2, color="#ff3050", lw=0.7, label="Raw")
        ax1.plot(range(W-1, len(all_rewards)), roll(all_rewards),
                 color="#ff3050", lw=2.2, label=f"Roll({W})")
        ax1.set_title("Episode Reward", color="#b0d8c8", fontsize=12)
        ax1.legend(framealpha=0.2, labelcolor="#b0d8c8")
        ax1.set_xlabel("Episode", color="#2a5040")

        ax2.plot(all_scores, alpha=0.2, color="#00ff88", lw=0.7, label="Raw")
        ax2.plot(range(W-1, len(all_scores)), roll(all_scores),
                 color="#00ff88", lw=2.2, label=f"Roll({W})")
        ax2.set_title("Containment Score (F1-hybrid)", color="#b0d8c8", fontsize=12)
        ax2.legend(framealpha=0.2, labelcolor="#b0d8c8")
        ax2.set_xlabel("Episode", color="#2a5040")
        ax2.set_ylim(0, 1.05)

        plt.tight_layout(pad=2.5)
        chart = "securenet_env/checkpoints/training_curves.png"
        plt.savefig(chart, dpi=150, bbox_inches="tight")
        print(f"   Chart saved → {chart}")
    except ImportError:
        print("   (matplotlib not installed — skipping chart)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SecureNet SOC RL Training v2")
    p.add_argument("--episodes",            type=int,   default=DEFAULTS["episodes"])
    p.add_argument("--lr",                  type=float, default=DEFAULTS["lr"])
    p.add_argument("--gamma",               type=float, default=DEFAULTS["gamma"])
    p.add_argument("--entropy_coeff",       type=float, default=DEFAULTS["entropy_coeff"])
    p.add_argument("--log_interval",        type=int,   default=DEFAULTS["log_interval"])
    p.add_argument("--checkpoint",          type=str,   default=DEFAULTS["checkpoint"])
    p.add_argument("--stats_out",           type=str,   default=DEFAULTS["stats_out"])
    p.add_argument("--difficulty_schedule", type=str,   default=DEFAULTS["difficulty_schedule"])
    train(p.parse_args())
