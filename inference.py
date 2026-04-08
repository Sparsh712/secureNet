"""
SecureNet Inference Demo — v2
==============================
Runs a scripted "perfect analyst" through ALL 5 difficulty tiers.
Uses all 6 action types including THREAT_INTEL and QUARANTINE_PROCESS.

Exact log format (required by grader):
  Start:  task=<name> env=securenet model=<model>
  Step:   step=<n> action=<str> reward=<+0.00> done=<true|false> error=<msg|null>
  End:    success=<true|false> steps=<n> rewards=<r1,r2,...>
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client import SecureNetClient

MODEL_NAME   = os.getenv("MODEL_NAME",   "rule_based_analyst_v2")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8001")

# ─────────────────────────────────────────────────────────────────────
# PLAYBOOKS — optimal action sequences per difficulty
# Each entry: (action_type, target_node, ip_address, timeframe, process_name, ioc)
# ─────────────────────────────────────────────────────────────────────
PLAYBOOKS = {
    # ── EASY: SSH brute-force + C2 on Server-1 ─────────────────────
    "easy": [
        ("query_logs",       "Workstation-A", None,             "last_hour", None, None),
        ("query_logs",       "Server-1",      None,             "last_hour", None, None),
        ("threat_intel",     None,            "10.0.0.99",      None,        None, "10.0.0.99"),
        ("threat_intel",     None,            "45.33.32.156",   None,        None, "45.33.32.156"),
        ("analyze_process",  "Server-1",      None,             None,        None, None),
        ("quarantine_process","Server-1",     None,             None,        "/tmp/.x (malicious)", None),
        ("block_ip",         None,            "10.0.0.99",      None,        None, None),
        ("block_ip",         None,            "45.33.32.156",   None,        None, None),
        ("isolate_host",     "Server-1",      None,             None,        None, None),
    ],

    # ── MEDIUM: PowerShell creds + phishing mail-server ──────────────
    "medium": [
        ("query_logs",       "Workstation-B",    None,             "last_hour", None, None),
        ("query_logs",       "Database-Primary", None,             "last_hour", None, None),
        ("query_logs",       "Mail-Server",      None,             "last_hour", None, None),
        ("query_logs",       "HR-Workstation",   None,             "last_hour", None, None),
        ("threat_intel",     None,               "172.16.0.200",   None,        None, "172.16.0.200"),
        ("threat_intel",     None,               "185.220.101.45", None,        None, "185.220.101.45"),
        ("analyze_process",  "Workstation-B",    None,             None,        None, None),
        ("analyze_process",  "Mail-Server",      None,             None,        None, None),
        ("quarantine_process","Workstation-B",   None,             None,        "mimikatz.exe", None),
        ("quarantine_process","Mail-Server",     None,             None,        "/bin/sh -i (malicious)", None),
        ("block_ip",         None,               "172.16.0.200",   None,        None, None),
        ("block_ip",         None,               "185.220.101.45", None,        None, None),
        ("isolate_host",     "Workstation-B",    None,             None,        None, None),
        ("isolate_host",     "Mail-Server",      None,             None,        None, None),
    ],

    # ── HARD: Ransomware Node-1 + Backup-Server (must act fast!) ─────
    "hard": [
        # Isolate Patient Zero within 3 steps to prevent drift to Node-2
        ("query_logs",       "Node-1",      None,           "last_hour", None, None),
        ("analyze_process",  "Node-1",      None,           None,        None, None),
        ("isolate_host",     "Node-1",      None,           None,        None, None),   # BEFORE step 3 drift
        # Now investigate the rest
        ("query_logs",       "Backup-Server", None,         "last_hour", None, None),
        ("query_logs",       "Firewall-GW",  None,          "last_hour", None, None),
        ("query_logs",       "CEO-Laptop",   None,          "last_hour", None, None),
        ("query_logs",       "Node-2",       None,          "last_hour", None, None),
        ("query_logs",       "Node-3",       None,          "last_hour", None, None),
        ("threat_intel",     None,           "193.56.29.11", None,       None, "193.56.29.11"),
        ("analyze_process",  "Backup-Server", None,         None,        None, None),
        ("quarantine_process","Backup-Server", None,        None,        "svc_update (malicious)", None),
        ("block_ip",         None,           "193.56.29.11", None,       None, None),
        ("isolate_host",     "Backup-Server", None,         None,        None, None),
    ],

    # ── CRITICAL: AD domain compromise + Finance keylogger + DMZ web shell ──
    "critical": [
        ("query_logs",       "DMZ-WebServer",         None,             "last_hour", None, None),
        ("query_logs",       "AD-Domain-Controller",  None,             "last_hour", None, None),
        ("query_logs",       "Finance-Workstation",   None,             "last_hour", None, None),
        ("query_logs",       "EDR-Server",            None,             "last_hour", None, None),
        ("query_logs",       "VPN-Gateway",           None,             "last_hour", None, None),
        ("threat_intel",     None,                    "91.92.109.200",  None,        None, "91.92.109.200"),
        ("threat_intel",     None,                    "45.142.212.100", None,        None, "45.142.212.100"),
        ("threat_intel",     None,                    None,             None,        None, "cafebabe5678"),
        ("analyze_process",  "DMZ-WebServer",         None,             None,        None, None),
        ("analyze_process",  "AD-Domain-Controller",  None,             None,        None, None),
        ("analyze_process",  "Finance-Workstation",   None,             None,        None, None),
        ("quarantine_process","DMZ-WebServer",        None,             None,        "webshell.php (malicious)", None),
        ("quarantine_process","AD-Domain-Controller", None,             None,        "ntds-dumper.exe (malicious)", None),
        ("quarantine_process","Finance-Workstation",  None,             None,        "keylogger.dll (malicious)", None),
        ("block_ip",         None,                    "91.92.109.200",  None,        None, None),
        ("block_ip",         None,                    "45.142.212.100", None,        None, None),
        ("isolate_host",     "DMZ-WebServer",         None,             None,        None, None),
        ("isolate_host",     "Finance-Workstation",   None,             None,        None, None),
        # AD-DC is CRITICAL so we cannot isolate it — route around the SOC dilemma
    ],

    # ── NIGHTMARE: Supply chain + k8s + CI/CD + DB (nation-state) ────
    "nightmare": [
        # Must isolate Internet-Router ASAP (step-3 drift to Core-Switch critical!)
        ("query_logs",       "Internet-Router",  None,             "last_hour", None, None),
        ("analyze_process",  "Internet-Router",  None,             None,        None, None),
        ("isolate_host",     "Internet-Router",  None,             None,        None, None),   # Before step 3
        # Investigate remaining nodes
        ("query_logs",       "Prod-DB-Primary",  None,             "last_hour", None, None),
        ("query_logs",       "K8s-Master",       None,             "last_hour", None, None),
        ("query_logs",       "CI-CD-Pipeline",   None,             "last_hour", None, None),
        ("query_logs",       "Core-Switch",      None,             "last_hour", None, None),
        ("query_logs",       "HSM-Server",       None,             "last_hour", None, None),
        ("query_logs",       "Backup-Vault",     None,             "last_hour", None, None),
        ("query_logs",       "CISO-Workstation", None,             "last_hour", None, None),
        ("threat_intel",     None,               "45.142.212.100", None,        None, "45.142.212.100"),
        ("threat_intel",     None,               "91.92.109.200",  None,        None, "91.92.109.200"),
        ("threat_intel",     None,               None,             None,        None, "update-cdn.ru"),
        ("threat_intel",     None,               None,             None,        None, "a3f4b9c1d2e5"),
        ("analyze_process",  "Prod-DB-Primary",  None,             None,        None, None),
        ("analyze_process",  "K8s-Master",       None,             None,        None, None),
        ("analyze_process",  "CI-CD-Pipeline",   None,             None,        None, None),
        ("quarantine_process","Prod-DB-Primary",  None,            None,        "xtrabackup (malicious)", None),
        ("quarantine_process","K8s-Master",       None,            None,        "cryptominer (malicious)", None),
        ("quarantine_process","CI-CD-Pipeline",   None,            None,        "malicious-build-step.sh", None),
        ("block_ip",         None,               "45.142.212.100", None,        None, None),
        ("block_ip",         None,               "91.92.109.200",  None,        None, None),
        ("isolate_host",     "Prod-DB-Primary",  None,             None,        None, None),
        ("isolate_host",     "K8s-Master",       None,             None,        None, None),
        ("isolate_host",     "CI-CD-Pipeline",   None,             None,        None, None),
    ],
}


def run_scenario(client: SecureNetClient, difficulty: str):
    task = f"soc_triage_{difficulty}"
    print(f"\nStart:  task={task} env=securenet model={MODEL_NAME}")

    obs     = client.reset(difficulty)
    rewards = []
    steps   = 0
    done    = obs.get("done", False)

    for act_type, target, ip, timeframe, process_name, ioc in PLAYBOOKS[difficulty]:
        if done:
            break
        steps += 1
        obs   = client.step(act_type, target, ip, timeframe, process_name, ioc)
        r     = obs.get("reward", 0.0)
        d     = obs.get("done", False)
        e     = obs.get("error") or "null"
        rewards.append(r)
        done  = d

        act_str = f"{act_type}:{target or ip or ioc or process_name or ''}"
        print(f"Step:   step={steps} action={act_str} reward={r:+.2f} done={str(d).lower()} error={e}")
        time.sleep(0.05)

    info    = obs.get("info", {})
    score   = info.get("total_score", 0.0)
    success = score >= 0.70 if done else False
    rewards_str = ",".join(f"{r:+.2f}" for r in rewards)
    print(f"End:    success={str(success).lower()} steps={steps} rewards={rewards_str}")

    if "total_score" in info:
        print(f"\n  ┌── Grader Report [{difficulty.upper()}] ─────────────────┐")
        print(f"  │  Total Score        : {info['total_score']:.3f}")
        print(f"  │  Containment Rate   : {info.get('containment_rate', 0):.3f}")
        print(f"  │  Kill Chain Disrupt : {info.get('kill_chain_disruption', 0):.3f}")
        print(f"  │  Time Bonus         : {info.get('time_bonus', 0):.3f}")
        print(f"  │  FP Rate            : {info.get('false_positive_rate', 0):.3f}")
        print(f"  │  Correctly Isolated : {info.get('correctly_isolated', 0)}")
        print(f"  │  Missed Threats     : {info.get('missed_threats', 0)}")
        print(f"  │  Catastrophic       : {info.get('catastrophic_failure', False)}")
        print(f"  │  Elapsed (s)        : {info.get('elapsed_seconds', 0):.1f}")
        print(f"  └─────────────────────────────────────────────┘")


def main():
    client = SecureNetClient(API_BASE_URL)

    # Verify server health
    try:
        h = client.health()
        print(f"✅ Server online — {h}")
    except RuntimeError as e:
        print(f"❌ {e}")
        sys.exit(1)

    tiers = ["easy", "medium", "hard", "critical", "nightmare"]
    run_tiers = os.getenv("TIERS", "easy,medium,hard,critical,nightmare").split(",")

    print("\n" + "═" * 60)
    print("  SecureNet SOC — v2 Rule-Based Analyst Demo")
    print("═" * 60)

    for diff in tiers:
        if diff not in run_tiers:
            continue
        print(f"\n{'─'*60}")
        print(f"  TIER: {diff.upper()}")
        print(f"{'─'*60}")
        run_scenario(client, diff)

    print("\n" + "═" * 60)
    print("  Demo complete. Check http://localhost:8001 for dashboard.")
    print("═" * 60)


if __name__ == "__main__":
    main()
