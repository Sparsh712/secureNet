"""
SecureNet - Dueling DQN Trainer v3
====================================
Key improvements over v2:
  * ACTION MASKING per scenario  - only ~15-20 valid actions per difficulty
    instead of 110 random ones. Cuts wasted exploration by ~85%.
  * RICHER STATE  - adds step fraction, queried/analyzed coverage ratios,
    threats-remaining count, blocked-IP flags. Agent can now reason about
    episode progress, not just raw log text.
  * SOFT TARGET UPDATES  - smoother Q-target tracking (tau=0.01 per step)
    instead of periodic hard copies that cause instability.
  * COSINE LR SCHEDULE  - gradually anneals learning rate for stable fine-tuning.
  * SEQUENTIAL TIER TRAINING  - one difficulty at a time avoids CPU contention.
  * HONEST SAVE  - checkpoints written every 50 episodes so interruptions lose
    at most 50 episodes, not 100.

Usage:
  # Train one tier (5000 episodes)
  python securenet_env/train_dqn.py --difficulty easy --episodes 5000

  # Sequential auto (easy -> medium -> ... -> nightmare, 5000 each)
  python securenet_env/train_dqn.py --difficulty sequential --episodes 5000
"""

import os, sys, json, time, argparse, random, math
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "server"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from securenet_environment import SecureNetEnvironment, NETWORK_TEMPLATES, THREAT_INTEL_DB

STDOUT_ENC = sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None

# ─────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────
DEFAULTS = dict(
    episodes        = 5000,
    lr              = 2e-4,
    gamma           = 0.99,
    batch_size      = 128,
    buffer_size     = 50_000,
    tau             = 0.005,         # soft target update coefficient
    eps_start       = 1.0,
    eps_end         = 0.05,
    eps_decay_steps = 4000,
    log_interval    = 50,
    save_interval   = 50,
    difficulty      = "sequential",
    checkpoint_dir  = "checkpoints",
)

TIERS = ["easy", "medium", "hard", "critical", "nightmare"]

# ─────────────────────────────────────────────────────────────────────
# PER-TIER ACTION MASKS
# Only include nodes/IPs/IOCs that are actually present in each scenario.
# This reduces the effective action space from 110 to ~15-25 actions.
# ─────────────────────────────────────────────────────────────────────
def build_action_mask(difficulty: str):
    """Return the list of (action_type, node, ip, ioc, proc) tuples valid for this difficulty."""
    tmpl = NETWORK_TEMPLATES[difficulty]
    nodes = list(tmpl["nodes"].keys())
    att_ips = tmpl.get("attacker_ips", [])
    all_iocs = list(THREAT_INTEL_DB.keys())

    # Quarantine pairs specific to this scenario
    sus_processes = {}
    for node, ndata in tmpl["nodes"].items():
        for proc in ndata.get("processes", []):
            if any(w in proc.lower() for w in ["malicious", "mimikatz", "lockbit", "psexec",
                                                "/tmp/", "/bin/sh", "svc_update", "webshell",
                                                "cryptominer", "ntds-dumper", "keylogger",
                                                "xtrabackup", "python3 (c2", "malicious-build"]):
                sus_processes.setdefault(node, []).append(proc)

    actions = []
    for n in nodes:
        actions.append(("query_logs",      n,    None, None, None))
        actions.append(("analyze_process", n,    None, None, None))
        actions.append(("isolate_host",    n,    None, None, None))
    for ip in att_ips:
        actions.append(("block_ip",        None, ip,   None, None))
    for ioc in all_iocs:
        actions.append(("threat_intel",    None, None, ioc,  None))
    for node, procs in sus_processes.items():
        for proc in procs:
            actions.append(("quarantine_process", node, None, None, proc))

    return actions


MASKED_ACTIONS = {d: build_action_mask(d) for d in TIERS}
MAX_ACTIONS = max(len(v) for v in MASKED_ACTIONS.values())

# ─────────────────────────────────────────────────────────────────────
# STATE REPRESENTATION
# Text BOW (118 VOCAB) + structured scalars (step %, coverage %, threats, IPs blocked, ...)
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
    "easy","medium","hard","critical","nightmare",
    "reconnaissance","initial","access","persistence","exfiltration","impact",
    "sla","violation","catastrophic","failure","timeout","score","grader",
    # node identifiers
    "workstation-a","server-1","workstation-b","database-primary","mail-server","hr-workstation",
    "node-1","node-2","node-3","ceo-laptop","firewall-gw","backup-server",
    "dmz-webserver","ad-domain-controller","finance-workstation",
    "edr-server","vpn-gateway","siem-platform",
    "internet-router","core-switch","prod-db-primary","k8s-master",
    "ci-cd-pipeline","hsm-server","backup-vault","ciso-workstation",
]
VOCAB = list(dict.fromkeys(VOCAB))   # deduplicate
VOCAB_SIZE = len(VOCAB)
SCALAR_DIM = 8    # step_frac, threats_remain_norm, n_isolated_norm, n_blocked_norm,
                  # queried_frac, analyzed_frac, ep_reward_norm, last_reward_norm
STATE_DIM = VOCAB_SIZE + SCALAR_DIM


def obs_to_vec(text: str, scalars: list, prev: torch.Tensor = None) -> torch.Tensor:
    lo  = text.lower()
    bow = torch.zeros(VOCAB_SIZE, dtype=torch.float32)
    for i, w in enumerate(VOCAB):
        if w in lo:
            bow[i] = 1.0
            
    if prev is not None:
        # Apply smoothing only to the textual Bag of Words, NOT the scalars
        prev_bow = prev[:VOCAB_SIZE]
        bow = torch.clamp(bow + prev_bow * 0.7, 0.0, 1.0)
        
    sc = torch.tensor(scalars, dtype=torch.float32)
    vec = torch.cat([bow, sc])
    return vec


def get_scalars(env, resp) -> list:
    info = resp.get("info", {})
    n_nodes    = max(len(env.network), 1)
    n_infected = len(env._tmpl.get("compromised", []))
    remain     = len(env.compromised) / max(n_infected, 1)
    step_frac  = env.step_count / max(env.max_steps, 1)
    iso_frac   = len(env.isolated)  / max(n_nodes, 1)
    blk_frac   = len(env.blocked_ips) / max(len(env._tmpl.get("attacker_ips", [1])), 1)
    q_frac     = len(env.queried_nodes)    / max(n_nodes, 1)
    a_frac     = len(env.analyzed_nodes)   / max(n_nodes, 1)
    ep_r       = max(min(info.get("episode_reward", 0) / 10.0, 1.0), -1.0)
    last_r     = max(min(resp.get("reward", 0) / 2.0, 1.0), -1.0)
    return [step_frac, remain, iso_frac, blk_frac, q_frac, a_frac, ep_r, last_r]


# ─────────────────────────────────────────────────────────────────────
# DUELING DQN
# ─────────────────────────────────────────────────────────────────────
class DuelingDQN(nn.Module):
    def __init__(self, in_dim: int, act_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256),    nn.LayerNorm(256), nn.ReLU(),
        )
        self.value = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        self.adv   = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, act_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        v = self.value(h)
        a = self.adv(h)
        return v + a - a.mean(dim=1, keepdim=True)


# ─────────────────────────────────────────────────────────────────────
# REPLAY BUFFER
# ─────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buf.append((s, a, r, ns, d))

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
# TRAIN ONE TIER
# ─────────────────────────────────────────────────────────────────────
def train_tier(difficulty: str, episodes: int, args, resume_path: str = None,
               online: DuelingDQN = None, target: DuelingDQN = None,
               opt=None, buf: ReplayBuffer = None):

    actions    = MASKED_ACTIONS[difficulty]
    act_dim    = len(actions)
    env        = SecureNetEnvironment()

    # Build fresh networks if not provided (first tier or no transfer)
    if online is None:
        online = DuelingDQN(STATE_DIM, act_dim)
        target = DuelingDQN(STATE_DIM, act_dim)
        target.load_state_dict(online.state_dict())
        target.eval()
        opt = optim.Adam(online.parameters(), lr=args.lr)
    else:
        # Re-build output head for new action dim while keeping trunk weights
        old_adv_in = online.adv[0].in_features
        online.adv = nn.Sequential(nn.Linear(old_adv_in, 128), nn.ReLU(), nn.Linear(128, act_dim))
        target.adv = nn.Sequential(nn.Linear(old_adv_in, 128), nn.ReLU(), nn.Linear(128, act_dim))
        target.load_state_dict(online.state_dict())
        target.eval()
        opt = optim.Adam(online.parameters(), lr=args.lr)

    if buf is None:
        buf = ReplayBuffer(args.buffer_size)

    ckpt_path  = os.path.join(args.checkpoint_dir, f"{difficulty}.pt")
    stats_path = os.path.join(args.checkpoint_dir, f"{difficulty}.json")

    all_rewards, all_scores = [], []
    global_step = 0
    start_ep    = 1

    # Resume from checkpoint
    if resume_path and os.path.exists(resume_path):
        try:
            ckpt = torch.load(resume_path, weights_only=True)
            online.load_state_dict(ckpt["online"])
            target.load_state_dict(ckpt["target"])
            opt.load_state_dict(ckpt["optimizer"])
            start_ep    = ckpt.get("episode", 1) + 1
            global_step = ckpt.get("global_step", 0)
            print(f"  Resumed from ep {start_ep-1}  step={global_step}")
        except Exception as e:
            print(f"  Checkpoint load failed ({e}) - starting fresh.")

    # Load existing stats
    if os.path.exists(stats_path):
        try:
            with open(stats_path) as f:
                d = json.load(f)
            all_rewards = d.get("rewards", [])
            all_scores  = d.get("scores",  [])
        except Exception:
            pass

    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=episodes, eta_min=args.lr/10)

    def epsilon(ep: int) -> float:
        frac = min(ep / args.eps_decay_steps, 1.0)
        base = args.eps_start + (args.eps_end - args.eps_start) * frac
        
        # Disable cyclic epsilon in the final 500 episodes to allow fully exploitative convergence
        if ep > episodes - 500:
            return base

        # Cyclic epsilon: every 1000 episodes, boost exploration to 0.4 for 200 eps
        if ep > 500 and (ep % 1000) < 200:
            return max(base, 0.4)
            
        return base

    t0 = time.time()
    print(f"\n  {'='*60}")
    print(f"  TIER: {difficulty.upper():10s}  |  actions={act_dim}  |  state={STATE_DIM}")
    print(f"  params={sum(p.numel() for p in online.parameters()):,}  |  episodes={episodes}")
    print(f"  {'='*60}")

    for ep in range(start_ep, episodes + 1):
        eps      = epsilon(ep)
        resp     = env.reset(difficulty)
        scalars  = get_scalars(env, resp)
        obs_vec  = obs_to_vec(resp.get("result", ""), scalars, None)
        done     = resp.get("done", False)
        ep_reward, final_score = 0.0, 0.0

        while not done:
            global_step += 1

            # Epsilon-greedy with masked actions
            if len(buf) < args.batch_size or random.random() < eps:
                action_idx = random.randrange(act_dim)
            else:
                with torch.no_grad():
                    action_idx = online(obs_vec.unsqueeze(0)).argmax(dim=1).item()

            act_type, node, ip, ioc, proc = actions[action_idx]
            resp    = env.step(action_type=act_type, target_node=node,
                               ip_address=ip, ioc=ioc, process_name=proc)
            reward  = resp.get("reward", 0.0)
            done    = resp.get("done", False)
            nxt_sc  = get_scalars(env, resp)
            nxt_vec = obs_to_vec(resp.get("result", ""), nxt_sc, obs_vec)

            buf.push(obs_vec, action_idx, reward, nxt_vec, float(done))
            obs_vec    = nxt_vec
            ep_reward += reward

            if "total_score" in resp.get("info", {}):
                final_score = resp["info"]["total_score"]

            # Learn
            if len(buf) >= args.batch_size:
                s, a, r, ns, d = buf.sample(args.batch_size)
                with torch.no_grad():
                    best_next = online(ns).argmax(dim=1, keepdim=True)
                    next_q    = target(ns).gather(1, best_next).squeeze(1)
                    td_target = r + args.gamma * next_q * (1 - d)
                q_vals = online(s).gather(1, a.unsqueeze(1)).squeeze(1)
                loss   = F.smooth_l1_loss(q_vals, td_target)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 5.0)
                opt.step()
                scheduler.step()

                # Soft target update
                for tp, op in zip(target.parameters(), online.parameters()):
                    tp.data.copy_(args.tau * op.data + (1 - args.tau) * tp.data)

        all_rewards.append(ep_reward)
        all_scores.append(final_score)

        # Log
        if ep % args.log_interval == 0:
            avg_r = np.mean(all_rewards[-args.log_interval:])
            avg_s = np.mean(all_scores [-args.log_interval:])
            elapsed = time.time() - t0
            pct60 = 100 * sum(1 for s in all_scores[-args.log_interval:] if s >= 0.6) / args.log_interval
            print(
                f"  Ep {ep:>5}/{episodes}  AvgR={avg_r:>+7.3f}  AvgScore={avg_s:.3f}"
                f"  Success%={pct60:.0f}%  eps={eps:.3f}  Buf={len(buf):>5}  {elapsed:>5.0f}s"
            )

        # Save
        if ep % args.save_interval == 0 or ep == episodes:
            torch.save({
                "online": online.state_dict(), "target": target.state_dict(),
                "optimizer": opt.state_dict(), "episode": ep,
                "difficulty": difficulty, "global_step": global_step,
            }, ckpt_path)
            with open(stats_path, "w") as f:
                json.dump({"rewards": all_rewards, "scores": all_scores,
                           "difficulty": difficulty}, f)

    # Final summary
    w = min(500, len(all_scores))
    final_avg = np.mean(all_scores[-w:])
    pct60     = 100 * sum(1 for s in all_scores[-w:] if s >= 0.6) / w
    elapsed   = time.time() - t0
    print(f"\n  DONE [{difficulty.upper()}]  last{w}_avg={final_avg:.3f}  success%={pct60:.1f}%  time={elapsed/60:.1f}min")

    return online, target, opt, buf, final_avg


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def main(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.difficulty == "sequential":
        online, target, opt, buf = None, None, None, None
        results = {}
        for tier in TIERS:
            ckpt_path = os.path.join(args.checkpoint_dir, f"{tier}.pt")
            online, target, opt, buf, avg = train_tier(
                tier, args.episodes, args,
                resume_path=ckpt_path,
                online=online, target=target, opt=opt,
                buf=None,   # fresh buffer each tier to avoid cross-tier contamination
            )
            results[tier] = avg

        print("\n" + "="*60)
        print("  SEQUENTIAL TRAINING COMPLETE")
        print("="*60)
        for tier, avg in results.items():
            bar = "#" * int(avg * 40)
            print(f"  {tier:10s}  last500={avg:.3f}  |{bar}")
    else:
        ckpt_path = os.path.join(args.checkpoint_dir, f"{args.difficulty}.pt")
        train_tier(args.difficulty, args.episodes, args, resume_path=ckpt_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SecureNet Dueling DQN Trainer v3")
    p.add_argument("--difficulty",      type=str,   default=DEFAULTS["difficulty"])
    p.add_argument("--episodes",        type=int,   default=DEFAULTS["episodes"])
    p.add_argument("--lr",              type=float, default=DEFAULTS["lr"])
    p.add_argument("--gamma",           type=float, default=DEFAULTS["gamma"])
    p.add_argument("--batch_size",      type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--buffer_size",     type=int,   default=DEFAULTS["buffer_size"])
    p.add_argument("--tau",             type=float, default=DEFAULTS["tau"])
    p.add_argument("--eps_start",       type=float, default=DEFAULTS["eps_start"])
    p.add_argument("--eps_end",         type=float, default=DEFAULTS["eps_end"])
    p.add_argument("--eps_decay_steps", type=int,   default=DEFAULTS["eps_decay_steps"])
    p.add_argument("--log_interval",    type=int,   default=DEFAULTS["log_interval"])
    p.add_argument("--save_interval",   type=int,   default=DEFAULTS["save_interval"])
    p.add_argument("--checkpoint_dir",  type=str,   default=DEFAULTS["checkpoint_dir"])
    main(p.parse_args())
