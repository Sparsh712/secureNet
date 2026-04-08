"""
SecureNet — DQN Evaluation / Test Suite
========================================
"Test data" in RL = running the trained policy greedily across held-out
scenarios. This script:

  1. Loads the trained DQN checkpoint (policy_dqn.pt)
  2. Sets ε = 0 (pure exploitation — no exploration noise)
  3. Runs N episodes per difficulty tier (default 20 each → 100 total)
  4. Reports mean score, std dev, success rate, and grader breakdown
  5. Compares trained DQN vs rule-based baseline (inference.py playbooks)
  6. Saves a full JSON test report

Usage:
  python securenet_env/evaluate.py
  python securenet_env/evaluate.py --episodes_per_tier 50
  python securenet_env/evaluate.py --tier nightmare --episodes_per_tier 100
  python securenet_env/evaluate.py --checkpoint securenet_env/checkpoints/policy_dqn.pt
"""

import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "server"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from securenet_environment import SecureNetEnvironment

# ─────────────────────────────────────────────────────────────────────
# SHARED CONSTANTS (must match train_dqn.py exactly)
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

TIERS        = ["easy", "medium", "hard", "critical", "nightmare"]
PASS_THRESH  = 0.60   # score >= this → "success"


# ─────────────────────────────────────────────────────────────────────
# DUELING DQN (identical to train_dqn.py)
# ─────────────────────────────────────────────────────────────────────
class DuelingDQN(nn.Module):
    def __init__(self, in_dim, act_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256),    nn.LayerNorm(256), nn.ReLU(),
        )
        self.value     = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        self.advantage = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, act_dim))

    def forward(self, x):
        h = self.trunk(x)
        v = self.value(h)
        a = self.advantage(h)
        return v + a - a.mean(dim=1, keepdim=True)


def obs_to_vec(text: str) -> torch.Tensor:
    lo  = text.lower()
    vec = torch.zeros(VOCAB_SIZE, dtype=torch.float32)
    for i, w in enumerate(VOCAB):
        if w in lo:
            vec[i] = 1.0
    return vec


# ─────────────────────────────────────────────────────────────────────
# RULE-BASED BASELINE (optimal hand-crafted playbooks)
# ─────────────────────────────────────────────────────────────────────
PLAYBOOKS = {
    "easy": [
        ("query_logs","Workstation-A",None,None,None),("query_logs","Server-1",None,None,None),
        ("threat_intel",None,None,"10.0.0.99",None),("threat_intel",None,None,"45.33.32.156",None),
        ("analyze_process","Server-1",None,None,None),
        ("quarantine_process","Server-1",None,None,"/tmp/.x (malicious)"),
        ("block_ip",None,"10.0.0.99",None,None),("block_ip",None,"45.33.32.156",None,None),
        ("isolate_host","Server-1",None,None,None),
    ],
    "medium": [
        ("query_logs","Workstation-B",None,None,None),("query_logs","Database-Primary",None,None,None),
        ("query_logs","Mail-Server",None,None,None),("query_logs","HR-Workstation",None,None,None),
        ("threat_intel",None,None,"172.16.0.200",None),("threat_intel",None,None,"185.220.101.45",None),
        ("analyze_process","Workstation-B",None,None,None),("analyze_process","Mail-Server",None,None,None),
        ("quarantine_process","Workstation-B",None,None,"mimikatz.exe"),
        ("quarantine_process","Mail-Server",None,None,"/bin/sh -i (malicious)"),
        ("block_ip",None,"172.16.0.200",None,None),("block_ip",None,"185.220.101.45",None,None),
        ("isolate_host","Workstation-B",None,None,None),("isolate_host","Mail-Server",None,None,None),
    ],
    "hard": [
        ("query_logs","Node-1",None,None,None),("analyze_process","Node-1",None,None,None),
        ("isolate_host","Node-1",None,None,None),
        ("query_logs","Backup-Server",None,None,None),("query_logs","Firewall-GW",None,None,None),
        ("threat_intel",None,None,"193.56.29.11",None),
        ("analyze_process","Backup-Server",None,None,None),
        ("quarantine_process","Backup-Server",None,None,"svc_update (malicious)"),
        ("block_ip",None,"193.56.29.11",None,None),("isolate_host","Backup-Server",None,None,None),
    ],
    "critical": [
        ("query_logs","DMZ-WebServer",None,None,None),
        ("query_logs","AD-Domain-Controller",None,None,None),
        ("query_logs","Finance-Workstation",None,None,None),
        ("threat_intel",None,None,"91.92.109.200",None),("threat_intel",None,None,"45.142.212.100",None),
        ("analyze_process","DMZ-WebServer",None,None,None),
        ("analyze_process","AD-Domain-Controller",None,None,None),
        ("analyze_process","Finance-Workstation",None,None,None),
        ("quarantine_process","DMZ-WebServer",None,None,"webshell.php (malicious)"),
        ("quarantine_process","AD-Domain-Controller",None,None,"ntds-dumper.exe (malicious)"),
        ("quarantine_process","Finance-Workstation",None,None,"keylogger.dll (malicious)"),
        ("block_ip",None,"91.92.109.200",None,None),("block_ip",None,"45.142.212.100",None,None),
        ("isolate_host","DMZ-WebServer",None,None,None),("isolate_host","Finance-Workstation",None,None,None),
    ],
    "nightmare": [
        ("query_logs","Internet-Router",None,None,None),
        ("analyze_process","Internet-Router",None,None,None),
        ("isolate_host","Internet-Router",None,None,None),
        ("query_logs","Prod-DB-Primary",None,None,None),("query_logs","K8s-Master",None,None,None),
        ("query_logs","CI-CD-Pipeline",None,None,None),
        ("threat_intel",None,None,"45.142.212.100",None),("threat_intel",None,None,"91.92.109.200",None),
        ("analyze_process","Prod-DB-Primary",None,None,None),("analyze_process","K8s-Master",None,None,None),
        ("analyze_process","CI-CD-Pipeline",None,None,None),
        ("quarantine_process","Prod-DB-Primary",None,None,"xtrabackup (malicious)"),
        ("quarantine_process","K8s-Master",None,None,"cryptominer (malicious)"),
        ("quarantine_process","CI-CD-Pipeline",None,None,"malicious-build-step.sh"),
        ("block_ip",None,"45.142.212.100",None,None),("block_ip",None,"91.92.109.200",None,None),
        ("isolate_host","Prod-DB-Primary",None,None,None),("isolate_host","K8s-Master",None,None,None),
        ("isolate_host","CI-CD-Pipeline",None,None,None),
    ],
}


def run_baseline_episode(env, difficulty: str):
    resp     = env.reset(difficulty)
    done     = resp.get("done", False)
    score    = 0.0
    steps    = 0
    for act_type, node, ip, ioc, proc in PLAYBOOKS[difficulty]:
        if done:
            break
        steps += 1
        resp  = env.step(action_type=act_type, target_node=node,
                         ip_address=ip, ioc=ioc, process_name=proc)
        done  = resp.get("done", False)
        if "total_score" in resp.get("info", {}):
            score = resp["info"]["total_score"]
    return score, steps, resp.get("info", {})


def run_dqn_episode(policy: DuelingDQN, env, difficulty: str):
    resp     = env.reset(difficulty)
    obs_vec  = obs_to_vec(resp.get("result", ""))
    done     = resp.get("done", False)
    score    = 0.0
    steps    = 0
    ep_reward= 0.0
    action_counts = {}

    while not done:
        with torch.no_grad():
            q_vals = policy(obs_vec.unsqueeze(0))
            idx    = q_vals.argmax(dim=1).item()

        act_type, node, ip, ioc, proc = ACTIONS[idx]
        action_counts[act_type] = action_counts.get(act_type, 0) + 1

        resp     = env.step(action_type=act_type, target_node=node,
                            ip_address=ip, ioc=ioc, process_name=proc)
        ep_reward += resp.get("reward", 0.0)
        obs_vec   = obs_to_vec(resp.get("result", ""))
        done      = resp.get("done", False)
        steps    += 1

        if "total_score" in resp.get("info", {}):
            score = resp["info"]["total_score"]

    return score, steps, ep_reward, resp.get("info", {}), action_counts


# ─────────────────────────────────────────────────────────────────────
# MAIN EVALUATION
# ─────────────────────────────────────────────────────────────────────
def evaluate(args):
    env  = SecureNetEnvironment()
    tiers = [args.tier] if args.tier else TIERS

    # Load DQN
    policy = DuelingDQN(VOCAB_SIZE, ACTION_SIZE)
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print("   Run training first:  python securenet_env/train_dqn.py")
        sys.exit(1)
    ckpt = torch.load(args.checkpoint, weights_only=True)
    policy.load_state_dict(ckpt["online"])
    policy.eval()
    trained_at_ep   = ckpt.get("episode", "?")
    trained_at_diff = ckpt.get("difficulty", "?")
    print(f"✅ Loaded DQN checkpoint  (ep={trained_at_ep}, final_tier={trained_at_diff})")

    results = {}
    report_rows = []

    print("\n" + "═" * 78)
    print("  SecureNet DQN — Evaluation Report")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Episodes   : {args.episodes_per_tier} per tier   |   Pass threshold: {PASS_THRESH:.0%}")
    print("═" * 78)

    for tier in tiers:
        print(f"\n  ── Tier: {tier.upper()} {'─'*55}")
        n = args.episodes_per_tier

        # ── Baseline ────────────────────────────────────────────────
        bl_scores = []
        for _ in range(n):
            s, _, _ = run_baseline_episode(env, tier)
            bl_scores.append(s)
        bl_mean = np.mean(bl_scores)
        bl_std  = np.std(bl_scores)
        bl_pass = np.mean([s >= PASS_THRESH for s in bl_scores])

        # ── Trained DQN ─────────────────────────────────────────────
        dqn_scores, dqn_steps_list, dqn_rewards = [], [], []
        all_action_counts = {}
        grader_details = []

        for i in range(n):
            s, steps, ep_rew, info, act_counts = run_dqn_episode(policy, env, tier)
            dqn_scores.append(s)
            dqn_steps_list.append(steps)
            dqn_rewards.append(ep_rew)
            grader_details.append(info)
            for k, v in act_counts.items():
                all_action_counts[k] = all_action_counts.get(k, 0) + v

            bar = "█" * int(s * 20) + "░" * (20 - int(s * 20))
            print(f"  Ep {i+1:>3}/{n}  [{bar}]  {s:.3f}  steps={steps}", end="\r")

        dqn_mean   = np.mean(dqn_scores)
        dqn_std    = np.std(dqn_scores)
        dqn_pass   = np.mean([s >= PASS_THRESH for s in dqn_scores])
        avg_steps  = np.mean(dqn_steps_list)
        avg_reward = np.mean(dqn_rewards)
        delta      = dqn_mean - bl_mean
        delta_str  = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"

        # Per-tier grader averages
        def grader_avg(key):
            vals = [d.get(key, 0) for d in grader_details if d]
            return np.mean(vals) if vals else 0.0

        containment  = grader_avg("containment_rate")
        kill_chain   = grader_avg("kill_chain_disruption")
        time_bonus   = grader_avg("time_bonus")
        fp_rate      = grader_avg("false_positive_rate")
        catastrophic = np.mean([d.get("catastrophic_failure", False) for d in grader_details if d])

        print(f"\n  {'':3}  Score  {dqn_mean:.3f} ± {dqn_std:.3f}  │  "
              f"Pass {dqn_pass:.0%}  │  "
              f"vs Baseline {bl_mean:.3f} ({delta_str})")
        print(f"       Containment {containment:.3f}  │  "
              f"Kill Chain {kill_chain:.3f}  │  "
              f"Time Bonus {time_bonus:.3f}  │  "
              f"FP Rate {fp_rate:.3f}")
        print(f"       Catastrophic {catastrophic:.0%}  │  "
              f"Avg Steps {avg_steps:.1f}  │  "
              f"Avg Reward {avg_reward:+.2f}")
        if all_action_counts:
            total_acts = sum(all_action_counts.values())
            usage = "  │  ".join(
                f"{k}: {v/total_acts:.0%}"
                for k, v in sorted(all_action_counts.items(), key=lambda x: -x[1])
            )
            print(f"       Actions: {usage}")

        results[tier] = {
            "dqn":      {"mean": dqn_mean, "std": dqn_std, "pass_rate": dqn_pass},
            "baseline": {"mean": bl_mean,  "std": bl_std,  "pass_rate": bl_pass},
            "delta":    delta,
            "grader":   {
                "containment_rate":      containment,
                "kill_chain_disruption": kill_chain,
                "time_bonus":            time_bonus,
                "false_positive_rate":   fp_rate,
                "catastrophic_rate":     catastrophic,
            },
            "episodes": {"avg_steps": avg_steps, "avg_reward": avg_reward},
        }
        report_rows.append((tier, bl_mean, dqn_mean, dqn_pass, containment, kill_chain, delta_str))

    # ── Summary Table ────────────────────────────────────────────────
    print("\n" + "═" * 78)
    print("  SUMMARY TABLE")
    print(f"  {'Tier':<12} {'Baseline':>10} {'DQN':>10} {'Pass%':>8} {'Contain':>9} {'KillChn':>9} {'Δ':>8}")
    print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*8} {'─'*9} {'─'*9} {'─'*8}")
    for tier, bl, dqn, prate, cont, kc, delta in report_rows:
        flag = "✅" if dqn >= PASS_THRESH else "⚠️ "
        print(f"  {flag} {tier:<10} {bl:>10.3f} {dqn:>10.3f} {prate:>8.0%} {cont:>9.3f} {kc:>9.3f} {delta:>8}")

    overall_dqn = np.mean([results[t]["dqn"]["mean"] for t in results])
    overall_bl  = np.mean([results[t]["baseline"]["mean"] for t in results])
    print(f"\n  Overall DQN   : {overall_dqn:.3f}")
    print(f"  Overall BL    : {overall_bl:.3f}")
    print(f"  DQN vs BL     : {overall_dqn - overall_bl:+.3f}")
    print("═" * 78)

    # ── Save report ─────────────────────────────────────────────────
    report = {
        "checkpoint":          args.checkpoint,
        "trained_at_episode":  trained_at_ep,
        "trained_at_tier":     trained_at_diff,
        "episodes_per_tier":   args.episodes_per_tier,
        "pass_threshold":      PASS_THRESH,
        "overall_dqn_score":   overall_dqn,
        "overall_baseline":    overall_bl,
        "per_tier":            results,
    }
    out = args.report_out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved → {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SecureNet DQN Evaluation")
    p.add_argument("--checkpoint",        type=str, default="securenet_env/checkpoints/policy_dqn.pt")
    p.add_argument("--episodes_per_tier", type=int, default=20)
    p.add_argument("--tier",              type=str, default=None,
                   help="Test single tier only: easy|medium|hard|critical|nightmare")
    p.add_argument("--report_out",        type=str,
                   default="securenet_env/checkpoints/eval_report.json")
    evaluate(p.parse_args())
