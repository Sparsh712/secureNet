"""
SecureNet Environment — Enhanced v2
=====================================
• 5 difficulty tiers: easy, medium, hard, critical, nightmare
• MITRE ATT&CK technique tagging on every log entry
• Kill chain stage tracking (Recon → Access → Persist → Lateral → Exfil)
• Adversarial drift with kill-chain-aware propagation
• New actions: THREAT_INTEL lookup, QUARANTINE_PROCESS
• Max-step timeout per episode with partial grading
• Time bonus in grader (faster = higher score)
• Mock IOC threat intelligence feed (in-memory)
"""

import copy
import uuid
import time
from typing import Dict, Any, Optional, List

# ─────────────────────────────────────────────────────────────────────
# MITRE ATT&CK KILL CHAIN STAGES
# ─────────────────────────────────────────────────────────────────────
KILL_CHAIN = ["Reconnaissance", "Initial Access", "Persistence", "Lateral Movement", "Exfiltration"]

# ─────────────────────────────────────────────────────────────────────
# MOCK THREAT INTELLIGENCE FEED (in-memory IOC database)
# ─────────────────────────────────────────────────────────────────────
THREAT_INTEL_DB = {
    # IPs
    "10.0.0.99":       {"type": "ip",   "threat": "HIGH",     "actor": "APT-28",    "tags": ["brute-force", "ssh"], "mitre": "T1110.001"},
    "45.33.32.156":    {"type": "ip",   "threat": "CRITICAL", "actor": "APT-28",    "tags": ["c2", "cobalt-strike"], "mitre": "T1071.001"},
    "172.16.0.200":    {"type": "ip",   "threat": "HIGH",     "actor": "FIN7",      "tags": ["lateral-movement", "wmi"], "mitre": "T1021.006"},
    "185.220.101.45":  {"type": "ip",   "threat": "CRITICAL", "actor": "TA505",     "tags": ["c2", "phishing-drop"], "mitre": "T1566.001"},
    "193.56.29.11":    {"type": "ip",   "threat": "CRITICAL", "actor": "LockBit3",  "tags": ["ransomware", "c2", "data-exfil"], "mitre": "T1486"},
    "45.142.212.100":  {"type": "ip",   "threat": "HIGH",     "actor": "Midnight",  "tags": ["recon", "scanner"], "mitre": "T1595"},
    "91.92.109.200":   {"type": "ip",   "threat": "CRITICAL", "actor": "Lazarus",   "tags": ["apt", "supply-chain"], "mitre": "T1195.002"},
    # File hashes
    "a3f4b9c1d2e5":    {"type": "hash", "threat": "CRITICAL", "actor": "LockBit3",  "tags": ["ransomware", "encryptor"], "mitre": "T1486"},
    "deadbeef1234":    {"type": "hash", "threat": "HIGH",     "actor": "FIN7",      "tags": ["trojan", "backdoor"], "mitre": "T1055"},
    "cafebabe5678":    {"type": "hash", "threat": "MEDIUM",   "actor": "Unknown",   "tags": ["keylogger"], "mitre": "T1056"},
    # Domains
    "update-cdn.ru":   {"type": "domain", "threat": "CRITICAL", "actor": "APT-29", "tags": ["c2", "dga"], "mitre": "T1568"},
    "telemetry.xyz":   {"type": "domain", "threat": "HIGH",    "actor": "Unknown",  "tags": ["beacon", "exfil"], "mitre": "T1041"},
}

# ─────────────────────────────────────────────────────────────────────
# NETWORK SCENARIO TEMPLATES
# ─────────────────────────────────────────────────────────────────────
NETWORK_TEMPLATES = {

    # ── EASY ─────────────────────────────────────────────────────────
    "easy": {
        "max_steps": 20,
        "nodes": {
            "Workstation-A": {
                "node_type": "endpoint", "status": "healthy", "critical": False,
                "processes": ["explorer.exe", "chrome.exe"],
                "connections": ["192.168.1.1"],
                "logs": (
                    "[INFO] Normal user logon events. No anomalies detected. "
                    "[MITRE:T1078.001 - Valid Accounts] No indicators."
                ),
            },
            "Server-1": {
                "node_type": "server", "status": "infected", "critical": False,
                "processes": ["sshd", "/tmp/.x (malicious)"],
                "connections": ["10.0.0.99", "45.33.32.156"],
                "logs": (
                    "[ALERT:T1110.001 - Brute Force:SSH] 347 failed SSH attempts from 10.0.0.99 in the last hour. "
                    "[ALERT:T1053.005 - Scheduled Task] Suspicious cron job added: '*/5 * * * * /tmp/.x'. "
                    "[ALERT:T1071.001 - C2 over HTTP] Outbound beacon to 45.33.32.156:4444 (Cobalt Strike IOC). "
                    "Kill Chain Stage: Persistence → C2."
                ),
            },
        },
        "compromised":  ["Server-1"],
        "attacker_ips": ["10.0.0.99", "45.33.32.156"],
        "kill_chain_stage": 2,  # Persistence
        "drift_map": {},
        "topology": {"Server-1": ["Workstation-A"]},
    },

    # ── MEDIUM ───────────────────────────────────────────────────────
    "medium": {
        "max_steps": 25,
        "nodes": {
            "Workstation-B": {
                "node_type": "endpoint", "status": "infected", "critical": False,
                "processes": ["powershell.exe (suspicious)", "mimikatz.exe"],
                "connections": ["192.168.10.5", "172.16.0.200"],
                "logs": (
                    "[ALERT:T1059.001 - PowerShell] Encoded execution: powershell -enc JABzAD0ATgBlAHcALQBPAGIAagBlAGMA... "
                    "[ALERT:T1003.001 - LSASS Dump] LSASS memory dump attempted. Credential harvesting. "
                    "[ALERT:T1021.006 - WMI Lateral] WMI connection established to 172.16.0.200. "
                    "Kill Chain Stage: Credential Access → Lateral Movement."
                ),
            },
            "Database-Primary": {
                "node_type": "database", "status": "healthy", "critical": True,
                "processes": ["postgres", "pgagent"],
                "connections": ["192.168.10.1"],
                "logs": (
                    "[INFO] Normal pg query patterns. Scheduled backup at 02:00. "
                    "[INFO:T1078 - Valid Accounts] Authenticated sessions from known application IPs."
                ),
            },
            "Mail-Server": {
                "node_type": "server", "status": "infected", "critical": False,
                "processes": ["sendmail", "/bin/sh -i (malicious)"],
                "connections": ["0.0.0.0:25", "185.220.101.45"],
                "logs": (
                    "[ALERT:T1566.001 - Phishing] Mass outbound phishing campaign: 2,400 emails to external domains. "
                    "[ALERT:T1059.004 - Unix Shell] sendmail spawned shell: '/bin/sh -i'. "
                    "[ALERT:T1041 - Exfiltration] Outbound data transfer to 185.220.101.45:443. "
                    "Kill Chain Stage: Initial Access → Exfiltration."
                ),
            },
            "HR-Workstation": {
                "node_type": "endpoint", "status": "healthy", "critical": False,
                "processes": ["outlook.exe", "teams.exe"],
                "connections": ["10.10.1.1"],
                "logs": "[INFO] Standard Office 365 activity. VPN connected. No anomalies. [T1078 - No indicators]",
            },
        },
        "compromised":  ["Workstation-B", "Mail-Server"],
        "attacker_ips": ["172.16.0.200", "185.220.101.45"],
        "kill_chain_stage": 3,   # Lateral Movement
        "drift_map": {},
        "topology": {
            "Workstation-B": ["Database-Primary", "Mail-Server"],
            "Mail-Server": ["HR-Workstation"],
        },
    },

    # ── HARD ─────────────────────────────────────────────────────────
    "hard": {
        "max_steps": 30,
        "nodes": {
            "Node-1": {
                "node_type": "endpoint", "status": "infected", "critical": False,
                "processes": ["LockBit3.exe", "PsExec.exe", "vssadmin.exe"],
                "connections": ["10.0.0.2", "10.0.0.3", "193.56.29.11"],
                "logs": (
                    "[CRITICAL:T1486 - Data Encrypted] LockBit3 ransomware executing. "
                    "Hash match: a3f4b9c1d2e5. ~3,400 files encrypted in 47s. "
                    "[ALERT:T1490 - Shadow Copy Delete] vssadmin delete shadows /all executed. "
                    "[ALERT:T1569.002 - PsExec] Lateral movement attempt via PsExec to 10.0.0.2, 10.0.0.3. "
                    "Kill Chain Stage: Impact → Lateral Movement."
                ),
            },
            "Node-2": {
                "node_type": "endpoint", "status": "healthy", "critical": False,
                "processes": ["explorer.exe", "winlogon.exe"],
                "connections": ["10.0.0.1"],
                "logs": "[INFO] Clean system. Standard user logon. [T1078 - No indicators]",
            },
            "Node-3": {
                "node_type": "endpoint", "status": "healthy", "critical": False,
                "processes": ["explorer.exe"],
                "connections": ["10.0.0.1"],
                "logs": "[INFO] Standard workstation. No unusual processes or connections.",
            },
            "CEO-Laptop": {
                "node_type": "endpoint", "status": "healthy", "critical": True,
                "processes": ["zoom.exe", "outlook.exe"],
                "connections": ["10.0.0.1"],
                "logs": "[INFO] Board presentation traffic. Zoom calls. No anomalies. Executive device.",
            },
            "Firewall-GW": {
                "node_type": "network", "status": "healthy", "critical": False,
                "processes": ["iptables", "snort"],
                "connections": ["193.56.29.11"],
                "logs": (
                    "[ALERT:T1041 - Exfiltration] Outbound spike to 193.56.29.11 from 10.0.0.x. "
                    "[ALERT:T1571 - Non-standard Port] Port 4444 outbound (reverse shell indicator). "
                    "[INFO] Geo-block triggered: 47 attempts blocked from RU/CN subnet."
                ),
            },
            "Backup-Server": {
                "node_type": "server", "status": "infected", "critical": False,
                "processes": ["backup_agent.exe", "svc_update (malicious)"],
                "connections": ["10.0.0.4", "193.56.29.11"],
                "logs": (
                    "[CRITICAL:T1486 - Ransomware] Backup catalog encrypted. Archives destroyed. "
                    "[ALERT:T1136.001 - New Account] Unauthorized admin account 'svc_update' created. "
                    "[ALERT:T1041 - Exfil] Backup data exfiltrated to 193.56.29.11. "
                    "Kill Chain Stage: Impact (recovery prevention)."
                ),
            },
        },
        "compromised":  ["Node-1", "Backup-Server"],
        "attacker_ips": ["193.56.29.11"],
        "kill_chain_stage": 4,   # Exfiltration
        "drift_map": {3: ("Node-1", "Node-2"), 6: ("Node-1", "Node-3")},
        "topology": {
            "Node-1": ["Node-2", "Node-3", "Backup-Server"],
            "Firewall-GW": ["Node-1"],
        },
    },

    # ── CRITICAL ─────────────────────────────────────────────────────
    "critical": {
        "max_steps": 35,
        "nodes": {
            "DMZ-WebServer": {
                "node_type": "server", "status": "infected", "critical": False,
                "processes": ["nginx", "webshell.php (malicious)", "curl"],
                "connections": ["0.0.0.0:80", "91.92.109.200"],
                "logs": (
                    "[CRITICAL:T1190 - Exploit Public App] SQL injection → RCE on DMZ-WebServer. "
                    "[ALERT:T1505.003 - Web Shell] webshell.php uploaded and executing. "
                    "[ALERT:T1041] Reverse shell beacon to 91.92.109.200:443. "
                    "Actor: Lazarus Group (supply-chain TTPs). Kill Chain: Initial Access."
                ),
            },
            "AD-Domain-Controller": {
                "node_type": "server", "status": "infected", "critical": True,
                "processes": ["lsass.exe", "ntds-dumper.exe (malicious)", "mimikatz.exe"],
                "connections": ["192.168.1.0/24", "91.92.109.200"],
                "logs": (
                    "[CRITICAL:T1003.003 - NTDS.dit Dump] Active Directory database extracted. "
                    "ALL domain credentials compromised. "
                    "[CRITICAL:T1558.003 - Kerberoasting] 147 service account hashes extracted. "
                    "[ALERT:T1484 - GPO Modification] Malicious Group Policy pushed to all endpoints. "
                    "Kill Chain: Credential Access → Domain Dominance."
                ),
            },
            "Finance-Workstation": {
                "node_type": "endpoint", "status": "infected", "critical": False,
                "processes": ["excel.exe", "keylogger.dll (malicious)", "chrome.exe"],
                "connections": ["192.168.1.50", "45.142.212.100"],
                "logs": (
                    "[ALERT:T1056.001 - Keylogger] cafebabe5678 DLL injected into Excel. "
                    "Banking credentials captured. Wire transfer initiated: $2.3M. "
                    "[ALERT:T1071.001] Beacon to 45.142.212.100 every 60s. "
                    "Kill Chain: Collection → Exfiltration."
                ),
            },
            "EDR-Server": {
                "node_type": "server", "status": "healthy", "critical": False,
                "processes": ["crowdstrike.exe", "sysmon.exe"],
                "connections": ["192.168.1.0/24"],
                "logs": (
                    "[INFO] EDR telemetry active across all endpoints. "
                    "[ALERT] 3 process alerts suppressed on AD-Domain-Controller (tampered?). "
                    "[INFO:T1562.001] EDR tampering attempt blocked on Finance-Workstation."
                ),
            },
            "VPN-Gateway": {
                "node_type": "network", "status": "healthy", "critical": True,
                "processes": ["openvpn", "strongswan"],
                "connections": ["0.0.0.0:443"],
                "logs": (
                    "[INFO] 14 active VPN sessions. 2 from unusual geos (RO, BY). "
                    "[ALERT:T1133 - External Remote Services] Auth with valid stolen credentials from unusual IP. "
                    "Kill Chain: Initial Access (credential reuse)."
                ),
            },
            "SIEM-Platform": {
                "node_type": "server", "status": "healthy", "critical": False,
                "processes": ["elasticsearch", "logstash", "kibana"],
                "connections": ["192.168.1.0/24"],
                "logs": "[INFO] Ingesting 14,000 events/min. 3 critical alerts suppressed externally.",
            },
        },
        "compromised":  ["DMZ-WebServer", "AD-Domain-Controller", "Finance-Workstation"],
        "attacker_ips": ["91.92.109.200", "45.142.212.100"],
        "kill_chain_stage": 4,   # Exfiltration
        "drift_map": {4: ("AD-Domain-Controller", "EDR-Server"), 8: ("DMZ-WebServer", "VPN-Gateway")},
        "topology": {
            "DMZ-WebServer":         ["AD-Domain-Controller", "Finance-Workstation"],
            "AD-Domain-Controller":  ["Finance-Workstation", "EDR-Server", "VPN-Gateway"],
            "Finance-Workstation":   ["SIEM-Platform"],
        },
    },

    # ── NIGHTMARE ────────────────────────────────────────────────────
    "nightmare": {
        "max_steps": 40,
        "nodes": {
            "Internet-Router": {
                "node_type": "network", "status": "infected", "critical": False,
                "processes": ["firmware-v2.3 (backdoored)"],
                "connections": ["0.0.0.0", "update-cdn.ru"],
                "logs": (
                    "[CRITICAL:T1195.002 - Supply Chain] Router firmware backdoored via supply chain attack. "
                    "[ALERT:T1557 - AiTM] All internal traffic intercepted and proxied through 45.142.212.100. "
                    "[ALERT:T1568 - DGA] Beacon to update-cdn.ru (Lazarus DGA domain). "
                    "Kill Chain: Supply Chain Compromise (nation-state TTPs)."
                ),
            },
            "Core-Switch": {
                "node_type": "network", "status": "healthy", "critical": True,
                "processes": ["ios-xe-15.6"],
                "connections": ["10.0.0.0/8"],
                "logs": "[INFO] Normal spanning-tree activity. No MAC flooding detected.",
            },
            "Prod-DB-Primary": {
                "node_type": "database", "status": "infected", "critical": True,
                "processes": ["mysqld", "xtrabackup (malicious)", "python3 (c2-agent)"],
                "connections": ["10.0.1.5", "update-cdn.ru"],
                "logs": (
                    "[CRITICAL:T1005 - Data from Local System] Full DB dump initiated. 847GB PII exfiltrated. "
                    "[ALERT:T1486] Ransomware encryption pending (trigger on C2 command). "
                    "[CRITICAL:T1048 - Exfil Alt Channel] Data leaving via DNS over HTTPS to update-cdn.ru. "
                    "Kill Chain: Collection → Exfiltration (pre-encryption hostage)."
                ),
            },
            "K8s-Master": {
                "node_type": "server", "status": "infected", "critical": False,
                "processes": ["kubelet", "kube-apiserver", "cryptominer (malicious)"],
                "connections": ["10.0.2.0/24", "45.142.212.100"],
                "logs": (
                    "[ALERT:T1610 - Deploy Container] Malicious container deployed via exposed k8s API. "
                    "[ALERT:T1496 - Resource Hijack] Cryptominer consuming 94% CPU across 23 pods. "
                    "[ALERT:T1552.007 - K8s Secrets] All Kubernetes secrets extracted. "
                    "Kill Chain: Execution → Resource Hijacking."
                ),
            },
            "CI-CD-Pipeline": {
                "node_type": "server", "status": "infected", "critical": False,
                "processes": ["jenkins", "malicious-build-step.sh"],
                "connections": ["10.0.2.5", "91.92.109.200"],
                "logs": (
                    "[CRITICAL:T1195.002 - Supply Chain] Build pipeline poisoned. "
                    "Malicious code injected into production artifact. 6 products affected. "
                    "[ALERT:T1552.001 - Credentials in File] AWS keys found in env vars and exfiltrated. "
                    "Kill Chain: Supply Chain (downstream impact)."
                ),
            },
            "HSM-Server": {
                "node_type": "server", "status": "healthy", "critical": True,
                "processes": ["pkcs11d", "hsm-monitor"],
                "connections": ["10.0.0.100"],
                "logs": "[INFO] Hardware Security Module operational. Key access logging normal. FIPS 140-2 L3.",
            },
            "Backup-Vault": {
                "node_type": "server", "status": "healthy", "critical": False,
                "processes": ["veeam-agent", "backup-scheduler"],
                "connections": ["10.0.3.1"],
                "logs": "[INFO] Immutable backup vault. Air-gap policy enforced. Last backup: 2h ago.",
            },
            "CISO-Workstation": {
                "node_type": "endpoint", "status": "healthy", "critical": True,
                "processes": ["outlook.exe", "teams.exe", "1password.exe"],
                "connections": ["10.0.0.50"],
                "logs": "[INFO] CISO device. MFA enforced. Privileged access workstation (PAW) policy active.",
            },
        },
        "compromised":  ["Internet-Router", "Prod-DB-Primary", "K8s-Master", "CI-CD-Pipeline"],
        "attacker_ips": ["45.142.212.100", "91.92.109.200", "update-cdn.ru"],
        "kill_chain_stage": 4,   # Full compromise
        "drift_map": {
            3:  ("Internet-Router",  "Core-Switch"),
            6:  ("Prod-DB-Primary",  "Backup-Vault"),
            9:  ("CI-CD-Pipeline",   "HSM-Server"),
            12: ("K8s-Master",       "CISO-Workstation"),
        },
        "topology": {
            "Internet-Router": ["Core-Switch", "K8s-Master", "Prod-DB-Primary"],
            "Core-Switch":     ["Prod-DB-Primary", "HSM-Server", "CISO-Workstation"],
            "CI-CD-Pipeline":  ["K8s-Master", "Prod-DB-Primary"],
        },
    },
}

# ─────────────────────────────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────
class SecureNetEnvironment:
    REWARDS = {
        "step_penalty":      -0.10,
        "query_logs":        +0.20,
        "analyze_hit":       +0.30,
        "analyze_miss":       0.00,
        "threat_intel_hit":  +0.25,
        "threat_intel_miss":  0.00,
        "block_ip":          +0.40,
        "quarantine_hit":    +0.50,
        "quarantine_miss":   -0.20,
        "isolate_infected":  +1.00,
        "isolate_healthy":   -0.50,
        "catastrophic":      -2.00,
    }

    CURRICULUM_THRESHOLDS = {
        "easy":      ("medium",    0.70),
        "medium":    ("hard",      0.75),
        "hard":      ("critical",  0.80),
        "critical":  ("nightmare", 0.85),
        "nightmare": (None,        1.0 ),
    }

    def __init__(self):
        self.episode_count  = 0
        self.total_steps    = 0
        self.episode_log: List[dict] = []
        self.stats = {"rewards": [], "scores": [], "episodes": 0, "difficulty": "easy"}
        self.score_history: List[float] = []
        self.curriculum_level = "easy"
        self._tmpl = None
        self.reset("easy")

    # ── Curriculum ───────────────────────────────────────────────────
    def _maybe_advance_curriculum(self):
        if len(self.score_history) < 10:
            return
        avg = sum(self.score_history[-10:]) / 10
        next_diff, threshold = self.CURRICULUM_THRESHOLDS.get(self.curriculum_level, (None, 1.0))
        if next_diff and avg >= threshold:
            self.curriculum_level = next_diff
            self.stats["difficulty"] = next_diff

    # ── Reset ─────────────────────────────────────────────────────────
    def reset(self, difficulty: str = None) -> Dict[str, Any]:
        diff = difficulty or self.curriculum_level
        tmpl = NETWORK_TEMPLATES.get(diff, NETWORK_TEMPLATES["easy"])
        self._tmpl        = tmpl
        self.network      = copy.deepcopy(tmpl["nodes"])
        self.compromised  = list(tmpl["compromised"])
        self.isolated     = []
        self.quarantined  = {}   # node → process_name
        self.blocked_ips  = []
        self.difficulty   = diff
        self.max_steps    = tmpl["max_steps"]
        self.step_count   = 0
        self.start_time   = time.time()
        self.episode_reward = 0.0
        self.done         = False
        self.kill_chain_stage = tmpl.get("kill_chain_stage", 0)
        self.episode_count += 1
        self.episode_log   = []

        return self._obs(
            f"🛡️  SECURENET SOC INITIALIZED\n"
            f"Difficulty: {diff.upper()} | Nodes: {len(self.network)} | "
            f"Threats Detected: {len(self.compromised)} | "
            f"Kill Chain Stage: {KILL_CHAIN[self.kill_chain_stage]}\n"
            f"Max Steps: {self.max_steps} | Investigate before containment.",
            True, 0.0, False, {}
        )

    # ── Step ─────────────────────────────────────────────────────────
    def step(self, action_type: str, target_node: str = None, ip_address: str = None,
             timeframe: str = None, process_name: str = None, ioc: str = None) -> Dict[str, Any]:

        if self.done:
            return self._obs("Episode finished. Call /reset.", False, 0.0, True, {})

        self.step_count  += 1
        self.total_steps += 1
        reward = self.REWARDS["step_penalty"]

        # ── Timeout check ────────────────────────────────────────────
        if self.step_count > self.max_steps:
            self.done = True
            return self._grade_episode(
                f"⏰ TIMEOUT: Max steps ({self.max_steps}) exceeded. Partial score awarded.", reward
            )

        # ── Adversarial drift ────────────────────────────────────────
        drift = self._tmpl.get("drift_map", {})
        if self.step_count in drift:
            src, tgt = drift[self.step_count]
            if src in self.compromised and tgt not in self.isolated and tgt not in self.compromised:
                self.compromised.append(tgt)
                self.network[tgt]["status"] = "infected"
                old = self.network[tgt]["logs"]
                self.network[tgt]["logs"] = (
                    f"[CRITICAL:T1210 - Lateral Movement] Ransomware/malware spread from {src}. "
                    f"Node now compromised. Previous baseline: {old}"
                )

        # ── Dispatch ─────────────────────────────────────────────────
        try:
            if action_type == "query_logs":
                result, reward = self._do_query_logs(target_node, timeframe, reward)

            elif action_type == "analyze_process":
                result, reward = self._do_analyze(target_node, reward)

            elif action_type == "threat_intel":
                result, reward = self._do_threat_intel(ioc or ip_address, reward)

            elif action_type == "block_ip":
                result, reward = self._do_block_ip(ip_address, reward)

            elif action_type == "quarantine_process":
                result, reward = self._do_quarantine(target_node, process_name, reward)

            elif action_type == "isolate_host":
                result, reward, end = self._do_isolate(target_node, reward)
                if end:
                    self.done = True
                    return self._grade_episode(result, reward)

            else:
                result = f"Unknown action_type: '{action_type}'."

        except ValueError as e:
            self._log(action_type, target_node, ip_address, ioc, -0.20)
            return self._obs("", False, -0.20, False, {"error": str(e)}, error=str(e))

        if not self.compromised:
            self.done = True
            return self._grade_episode(result + " ✅ All threats contained!", reward)

        self._log(action_type, target_node, ip_address, ioc, reward)
        return self._obs(result, True, reward, self.done, {})

    # ── Actions ──────────────────────────────────────────────────────
    def _do_query_logs(self, node, timeframe, base):
        if not node or node not in self.network:
            raise ValueError(f"Node '{node}' not found. Available: {list(self.network.keys())}")
        n   = self.network[node]
        tf  = f" [{timeframe}]" if timeframe else ""
        r   = base + self.REWARDS["query_logs"]
        return (
            f"LOGS{tf} — {node} ({n['node_type'].upper()}) [Status: {n['status'].upper()}]\n"
            f"{n['logs']}"
        ), r

    def _do_analyze(self, node, base):
        if not node or node not in self.network:
            raise ValueError(f"Node '{node}' not found.")
        n   = self.network[node]
        procs = n.get("processes", [])
        sus = [p for p in procs if any(w in p.lower() for w in
              ["malicious","mimikatz","lockbit","psexec","/tmp/","/bin/sh","svc_update",
               "webshell","cryptominer","ntds-dumper","keylogger","xtrabackup","python3","curl"])]
        if sus:
            r = base + self.REWARDS["analyze_hit"]
            res = (
                f"PROCESS ANALYSIS — {node}:\n"
                f"⚠️  SUSPICIOUS: {', '.join(sus)}\n"
                f"All processes: {', '.join(procs)}"
            )
        else:
            r = base + self.REWARDS["analyze_miss"]
            res = (
                f"PROCESS ANALYSIS — {node}:\n"
                f"✅ No suspicious processes detected.\n"
                f"Running: {', '.join(procs)}"
            )
        return res, r

    def _do_threat_intel(self, ioc, base):
        if not ioc:
            raise ValueError("Provide 'ioc' or 'ip_address' for threat_intel lookup.")
        rec = THREAT_INTEL_DB.get(ioc)
        if rec:
            r = base + self.REWARDS["threat_intel_hit"]
            res = (
                f"🔍 THREAT INTEL — {ioc}\n"
                f"Type: {rec['type'].upper()} | Threat Level: {rec['threat']}\n"
                f"Actor: {rec['actor']} | MITRE: {rec['mitre']}\n"
                f"Tags: {', '.join(rec['tags'])}"
            )
        else:
            r = base + self.REWARDS["threat_intel_miss"]
            res = f"🔍 THREAT INTEL — {ioc}: No records found in IOC feed. Likely benign."
        return res, r

    def _do_block_ip(self, ip, base):
        if not ip:
            raise ValueError("ip_address required for block_ip.")
        if ip in self.blocked_ips:
            return f"IP {ip} already blocked.", base
        self.blocked_ips.append(ip)
        att = self._tmpl.get("attacker_ips", [])
        if ip in att:
            r   = base + self.REWARDS["block_ip"]
            res = f"🛡️  {ip} BLOCKED. C2/attacker communication severed."
        else:
            r   = base
            res = f"IP {ip} blocked. (Not a known IOC in this scenario.)"
        return res, r

    def _do_quarantine(self, node, process_name, base):
        if not node or node not in self.network:
            raise ValueError(f"Node '{node}' not found.")
        if not process_name:
            raise ValueError("process_name required for quarantine_process.")
        procs      = self.network[node].get("processes", [])
        sus_procs  = [p for p in procs if any(w in p.lower() for w in
                      ["malicious","mimikatz","lockbit","psexec","/tmp/","/bin/sh","svc_update",
                       "webshell","cryptominer","ntds-dumper","keylogger","xtrabackup"])]
        matched    = any(process_name.lower() in p.lower() for p in sus_procs)
        if matched:
            # Reduce impact without full isolation
            self.network[node]["processes"] = [p for p in procs if process_name.lower() not in p.lower()]
            self.quarantined[node]           = process_name
            r   = base + self.REWARDS["quarantine_hit"]
            res = (
                f"🔒 QUARANTINE [{node}]: Process '{process_name}' terminated and sandboxed. "
                f"Note: Node still active on network — monitor for re-infection."
            )
        else:
            r   = base + self.REWARDS["quarantine_miss"]
            res = f"❌ QUARANTINE FAILED [{node}]: '{process_name}' not identified as malicious."
        return res, r

    def _do_isolate(self, node, base):
        if not node or node not in self.network:
            raise ValueError(f"Node '{node}' not found.")
        if node in self.isolated:
            return f"'{node}' already isolated.", base, False
        n = self.network[node]
        self.isolated.append(node)
        n["status"] = "isolated"

        if n.get("critical") and node not in self.compromised:
            r   = base + self.REWARDS["catastrophic"]
            res = (
                f"💥 CATASTROPHIC: {node} offline — CRITICAL infrastructure destroyed. "
                f"SLA VIOLATION. Incident escalated to CISO & Board."
            )
            return res, r, True

        if node in self.compromised:
            self.compromised.remove(node)
            r   = base + self.REWARDS["isolate_infected"]
            res = f"🔒 {node} isolated. Threat contained. ({len(self.compromised)} threats remaining)"
            done = len(self.compromised) == 0
        else:
            r   = base + self.REWARDS["isolate_healthy"]
            res = f"⚠️  FALSE POSITIVE: {node} isolated but was clean. Unnecessary downtime."
            done = False
        return res, r, done

    # ── Grader ───────────────────────────────────────────────────────
    def _grade_episode(self, message: str, last_reward: float) -> Dict[str, Any]:
        original = set(self._tmpl["compromised"])
        iso_set  = set(self.isolated)
        healthy  = set(self.network.keys()) - original

        correctly_isolated = len(iso_set & original)
        missed             = len(original - iso_set)
        false_positives    = len(iso_set & healthy)
        catastrophic       = any(
            self.network[n].get("critical") and n not in original and n in iso_set
            for n in iso_set
        )

        # Kill chain disruption score: did agent block IPs before isolating?
        att_ips         = set(self._tmpl.get("attacker_ips", []))
        blocked_att     = len(set(self.blocked_ips) & att_ips)
        kc_score        = blocked_att / max(len(att_ips), 1)

        containment    = correctly_isolated / max(len(original), 1)
        fp_pen         = false_positives    / max(len(healthy), 1)

        # Time bonus: faster resolution = higher score (linear scale)
        elapsed        = time.time() - self.start_time
        time_bonus     = max(0.0, 0.15 * (1.0 - elapsed / 120.0))  # up to +0.15

        score = max(0.0, containment * 0.65 + kc_score * 0.20 + time_bonus - fp_pen * 0.5)
        if catastrophic:
            score = 0.0

        self.score_history.append(score)
        self._maybe_advance_curriculum()

        self.stats["rewards"].append(round(self.episode_reward + last_reward, 3))
        self.stats["scores"].append(round(score, 3))
        self.stats["episodes"] = self.episode_count

        self._log("grader", None, None, None, last_reward)

        obs = self._obs(message, True, last_reward, True, {
            "total_score":           round(score, 3),
            "containment_rate":      round(containment, 3),
            "kill_chain_disruption": round(kc_score, 3),
            "time_bonus":            round(time_bonus, 3),
            "false_positive_rate":   round(fp_pen, 3),
            "correctly_isolated":    correctly_isolated,
            "missed_threats":        missed,
            "false_positives":       false_positives,
            "catastrophic_failure":  catastrophic,
            "elapsed_seconds":       round(elapsed, 1),
        })
        return obs

    # ── Helpers ──────────────────────────────────────────────────────
    def _obs(self, result, success, reward, done, info, error=""):
        self.episode_reward += reward
        info.update({
            "episode":        self.episode_count,
            "step":           self.step_count,
            "max_steps":      self.max_steps,
            "episode_reward": round(self.episode_reward, 3),
            "compromised":    list(self.compromised),
            "isolated":       list(self.isolated),
            "blocked_ips":    list(self.blocked_ips),
            "difficulty":     self.difficulty,
            "kill_chain":     KILL_CHAIN[self.kill_chain_stage],
            "curriculum":     self.curriculum_level,
        })
        return {
            "result": result, "success": success, "reward": round(reward, 3),
            "done": done, "error": error, "info": info,
        }

    def _log(self, action_type, target, ip, ioc, reward):
        self.episode_log.append({
            "step":        self.step_count,
            "action_type": action_type,
            "target_node": target,
            "ip_address":  ip,
            "ioc":         ioc,
            "reward":      round(reward, 3),
            "timestamp":   time.time(),
            "compromised_count": len(self.compromised),
        })
