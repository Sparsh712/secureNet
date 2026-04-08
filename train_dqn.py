"""
SecureNet — Dueling DQN Trainer (No HTTP)
==========================================
Algorithm : Dueling Deep Q-Network
          + Experience Replay (circular buffer)
          + Target Network (hard update every N steps)
          + Epsilon-greedy exploration (1.0 → 0.05 over training)
          + 5-tier curriculum (easy → nightmare)

Why DQN over REINFORCE?
  • Replay buffer breaks temporal correlations that plateau REINFORCE
  • Target network gives stable Q-value targets (no oscillating loss)
  • Epsilon-greedy is far more effective than entropy noise for sparse rewards
  • Dueling arch separates state value from action advantage → faster convergence

Expected results:
  ~500  episodes → 0.5+ score (EASY mastered)
  ~1500 episodes → 0.7+ score (MEDIUM mastered)
  ~3000 episodes → 0.8+ score (HARD with adversarial drift)

Usage:
  python securenet_env/train_dqn.py
  python securenet_env/train_dqn.py --episodes 5000
"""

import os, sys, json, time, argparse, random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "server"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from securenet_environment import SecureNetEnvironment

# ─────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────
DEFAULTS = dict(
    episodes          = 4000,
    lr                = 5e-4,
    gamma             = 0.99,
    batch_size        = 128,
    buffer_size       = 20_000,
    target_update     = 200,       # hard-copy online → target every N steps
    eps_start         = 1.0,
    eps_end           = 0.05,
    eps_decay_steps   = 3000,      # episodes over which epsilon decays
    log_interval      = 100,
    checkpoint        = "securenet_env/checkpoints/policy_dqn.pt",
    stats_out         = "securenet_env/checkpoints/stats.json",
    difficulty        = "progressive",
)

CURRICULUM = [
    ("easy",      "medium",    0.45),
    ("medium",    "hard",      0.55),
    ("hard",      "critical",  0.65),
    ("critical",  "nightmare", 0.75),
]

# ─────────────────────────────────────────────────────────────────────
# VOCABULARY  (118 tokens, shared with train_fast.py)
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

# ─────────────────────────────────────────────────────────────────────
# ACTION SPACE  (110 actions — pruned quarantine combos)
# ─────────────────────────────────────────────────────────────────────
ALL_NODES = [
    "Workstation-A","Server-1",
    "Workstation-B","Database-Primary","Mail-Server","HR-Workstation",
    "Node-1","Node-2","Node-3","CEO-Laptop","Firewall-GW","Backup-Server",
    "DMZ-WebServer","AD-Domain-Controller","Finance-Workstation",
    "EDR-Server","VPN-Gateway","SIEM-Platform",
    "Internet-Router","Core-Switch","Prod-DB-Primary","K8s-Master",
    "CI-CD-Pipeline","HSM-Server","Backup-Vault","CISO-Workstation",
]
ALL_IPS  = ["10.0.0.99","45.33.32.156","172.16.0.200","185.220.101.45",
            "193.56.29.11","45.142.212.100","91.92.109.200"]
ALL_IOCS = ALL_IPS + ["a3f4b9c1d2e5","deadbeef1234","cafebabe5678","update-cdn.ru","telemetry.xyz"]
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
ACTIONS = (
    [("query_logs",         n,    None, None, None) for n in ALL_NODES] +
    [("analyze_process",    n,    None, None, None) for n in ALL_NODES] +
    [("isolate_host",       n,    None, None, None) for n in ALL_NODES] +
    [("block_ip",           None, ip,   None, None) for ip in ALL_IPS]  +
    [("threat_intel",       None, None, ioc,  None) for ioc in ALL_IOCS] +
    [("quarantine_process", n,    None, None, proc) for n, proc in QUARANTINE_PAIRS]
)
ACTION_SIZE = len(ACTIONS)

VOCAB = VOCAB + [n.lower() for n in ALL_NODES] + [ip.lower() for ip in ALL_IPS] + [ioc.lower() for ioc in ALL_IOCS]
VOCAB = list(dict.fromkeys(VOCAB))
VOCAB_SIZE = len(VOCAB)
STATE_DIM = VOCAB_SIZE + ACTION_SIZE

# ─────────────────────────────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────────────────────────────
def obs_to_vec(text: str, last_action: int = -1, prev_state: torch.Tensor = None) -> torch.Tensor:
    lo  = text.lower()
    vec = torch.zeros(STATE_DIM, dtype=torch.float32)
    for i, w in enumerate(VOCAB):
        if w in lo:
            vec[i] = 1.0
    if last_action >= 0:
        vec[VOCAB_SIZE + last_action] = 1.0
    
    if prev_state is not None:
        vec = torch.clamp(vec + prev_state * 0.9, 0.0, 1.0)
    return vec

# ─────────────────────────────────────────────────────────────────────
# DUELING DQN NETWORK
#   Input  → shared trunk → split into
#     Value stream     V(s)        →  scalar
#     Advantage stream A(s, a)     →  vector[ACTION_SIZE]
#   Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))
# ─────────────────────────────────────────────────────────────────────
class DuelingDQN(nn.Module):
    def __init__(self, in_dim: int, act_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256),    nn.LayerNorm(256), nn.ReLU(),
        )
        # Value stream
        self.value = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h   = self.trunk(x)
        v   = self.value(h)                        # (B, 1)
        a   = self.advantage(h)                    # (B, A)
        q   = v + a - a.mean(dim=1, keepdim=True)  # (B, A)
        return q

# ─────────────────────────────────────────────────────────────────────
# REPLAY BUFFER
# ─────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            torch.stack(s),
            torch.tensor(a,  dtype=torch.long),
            torch.tensor(r,  dtype=torch.float32),
            torch.stack(ns),
            torch.tensor(d,  dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buf)

# ─────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────
def train(args):
    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)

    env       = SecureNetEnvironment()
    online    = DuelingDQN(STATE_DIM, ACTION_SIZE)
    target    = DuelingDQN(STATE_DIM, ACTION_SIZE)
    target.load_state_dict(online.state_dict())
    target.eval()

    opt       = optim.Adam(online.parameters(), lr=args.lr)
    buf       = ReplayBuffer(args.buffer_size)

    difficulty     = "easy" if args.difficulty == "progressive" else args.difficulty
    all_rewards    = []
    all_scores     = []
    score_history  = []
    global_step    = 0
    start_ep       = 1

    # Resume
    if os.path.exists(args.checkpoint):
        try:
            ckpt = torch.load(args.checkpoint, weights_only=True)
            online.load_state_dict(ckpt["online"])
            target.load_state_dict(ckpt["target"])
            opt.load_state_dict(ckpt["optimizer"])
            start_ep   = ckpt.get("episode", 1) + 1
            difficulty = ckpt.get("difficulty", "easy")
            global_step = ckpt.get("global_step", 0)
            print(f"▶ Resumed from ep {start_ep-1}  difficulty={difficulty}  step={global_step}")
        except Exception as e:
            print(f"⚠ Checkpoint load failed ({e}) — starting fresh.")

    # Epsilon schedule
    def epsilon(ep: int) -> float:
        frac = min(ep / args.eps_decay_steps, 1.0)
        return args.eps_start + (args.eps_end - args.eps_start) * frac

    print("\n" + "═" * 72)
    print("  ⚡ SecureNet — Dueling DQN + Replay Buffer + Target Network")
    print("═" * 72)
    print(f"  Network params  : {sum(p.numel() for p in online.parameters()):,}")
    print(f"  Action space    : {ACTION_SIZE}")
    print(f"  Replay buffer   : {args.buffer_size:,}")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  Target update   : every {args.target_update} steps")
    print(f"  Epsilon         : {args.eps_start} → {args.eps_end} over {args.eps_decay_steps} ep")
    print(f"  Episodes        : {args.episodes}")
    print("═" * 72 + "\n")

    t0 = time.time()

    for ep in range(start_ep, args.episodes + 1):

        # ── Curriculum ───────────────────────────────────────────────
        if args.difficulty == "progressive" and len(score_history) >= 10:
            rolling = sum(score_history[-10:]) / 10
            for cur, nxt, thresh in CURRICULUM:
                if difficulty == cur and rolling >= thresh:
                    difficulty = nxt
                    print(f"\n  🎓 CURRICULUM → {nxt.upper()}  (rolling avg = {rolling:.3f})\n")
                    break

        # ── Collect episode ──────────────────────────────────────────
        eps      = epsilon(ep)
        resp     = env.reset(difficulty)
        obs_vec  = obs_to_vec(resp.get("result", ""), -1, None)
        done     = resp.get("done", False)
        ep_reward, final_score = 0.0, 0.0

        while not done:
            global_step += 1

            # Epsilon-greedy action selection
            if (len(buf) < args.batch_size) or (random.random() < eps):
                action_idx = random.randrange(ACTION_SIZE)
            else:
                with torch.no_grad():
                    action_idx = online(obs_vec.unsqueeze(0)).argmax(dim=1).item()

            act_type, node, ip, ioc, proc = ACTIONS[action_idx]
            resp     = env.step(action_type=act_type, target_node=node,
                                ip_address=ip, ioc=ioc, process_name=proc)
            reward   = resp.get("reward", 0.0)
            done     = resp.get("done", False)
            nxt_vec  = obs_to_vec(resp.get("result", ""), action_idx, obs_vec)

            buf.push(obs_vec, action_idx, reward, nxt_vec, float(done))
            obs_vec   = nxt_vec
            ep_reward += reward

            if "total_score" in resp.get("info", {}):
                final_score = resp["info"]["total_score"]

            # ── Learn ─────────────────────────────────────────────────
            if len(buf) >= args.batch_size:
                s, a, r, ns, d = buf.sample(args.batch_size)

                with torch.no_grad():
                    # Double DQN: online selects, target evaluates
                    best_next_actions = online(ns).argmax(dim=1, keepdim=True)
                    next_q = target(ns).gather(1, best_next_actions).squeeze(1)
                    td_target = r + args.gamma * next_q * (1 - d)

                q_vals = online(s).gather(1, a.unsqueeze(1)).squeeze(1)
                loss   = F.smooth_l1_loss(q_vals, td_target)  # Huber loss

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                opt.step()

                # Hard target network update
                if global_step % args.target_update == 0:
                    target.load_state_dict(online.state_dict())

        all_rewards.append(ep_reward)
        all_scores.append(final_score)
        score_history.append(final_score)

        # ── Logging ───────────────────────────────────────────────────
        if ep % args.log_interval == 0:
            avg_r   = np.mean(all_rewards[-args.log_interval:])
            avg_s   = np.mean(all_scores[-args.log_interval:])
            elapsed = time.time() - t0
            print(
                f"Ep {ep:>5}/{args.episodes} │ "
                f"AvgReward {avg_r:>+7.3f} │ "
                f"AvgScore {avg_s:.3f} │ "
                f"Tier {difficulty:<9} │ "
                f"ε={eps:.3f} │ "
                f"Buf {len(buf):>6} │ "
                f"Elapsed {elapsed:>5.0f}s"
            )

        if ep % 100 == 0:
            torch.save({
                "online":      online.state_dict(),
                "target":      target.state_dict(),
                "optimizer":   opt.state_dict(),
                "episode":     ep,
                "difficulty":  difficulty,
                "global_step": global_step,
            }, args.checkpoint)
            with open(args.stats_out, "w") as f:
                json.dump({"rewards": all_rewards,
                           "scores":  all_scores,
                           "difficulty": difficulty}, f)

    # Final saves
    torch.save({"online": online.state_dict(), "target": target.state_dict(),
                "optimizer": opt.state_dict(), "episode": args.episodes,
                "difficulty": difficulty, "global_step": global_step}, args.checkpoint)
    print(f"\n✅ Done — {args.checkpoint}")
    print(f"   Total : {(time.time()-t0)/60:.1f} min")
    print(f"   Final Avg Score (last {args.log_interval} ep): "
          f"{np.mean(all_scores[-args.log_interval:]):.3f}")

    # Training curve
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), facecolor="#030d0a")
        cols  = ["#ff3050", "#00ff88", "#00c8ff"]
        labels = ["Episode Reward", "Containment Score", "Reward (smoothed)"]
        data   = [all_rewards, all_scores, all_rewards]
        W = 100
        roll = lambda a: np.convolve(a, np.ones(W)/W, mode="valid")

        for ax, col, lab, dat in zip(axes, cols, labels, data):
            ax.set_facecolor("#061410")
            ax.tick_params(colors="#2a5040")
            ax.spines[:].set_color("#0d2b22")
            ax.plot(dat, alpha=0.18, color=col, lw=.6, label="Raw")
            if len(dat) >= W:
                ax.plot(range(W-1, len(dat)), roll(dat), color=col, lw=2.2, label=f"Roll({W})")
            ax.set_title(lab, color="#b0d8c8", fontsize=11)
            ax.legend(framealpha=0.2, labelcolor="#b0d8c8", fontsize=9)
            if "Score" in lab:
                ax.set_ylim(0, 1.05)

        plt.tight_layout(pad=2.5)
        path = "securenet_env/checkpoints/training_curves.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"   Chart → {path}")
    except ImportError:
        pass


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SecureNet Dueling DQN Trainer")
    p.add_argument("--episodes",        type=int,   default=DEFAULTS["episodes"])
    p.add_argument("--lr",              type=float, default=DEFAULTS["lr"])
    p.add_argument("--gamma",           type=float, default=DEFAULTS["gamma"])
    p.add_argument("--batch_size",      type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--buffer_size",     type=int,   default=DEFAULTS["buffer_size"])
    p.add_argument("--target_update",   type=int,   default=DEFAULTS["target_update"])
    p.add_argument("--eps_start",       type=float, default=DEFAULTS["eps_start"])
    p.add_argument("--eps_end",         type=float, default=DEFAULTS["eps_end"])
    p.add_argument("--eps_decay_steps", type=int,   default=DEFAULTS["eps_decay_steps"])
    p.add_argument("--log_interval",    type=int,   default=DEFAULTS["log_interval"])
    p.add_argument("--checkpoint",      type=str,   default=DEFAULTS["checkpoint"])
    p.add_argument("--stats_out",       type=str,   default=DEFAULTS["stats_out"])
    p.add_argument("--difficulty",      type=str,   default=DEFAULTS["difficulty"])
    train(p.parse_args())
