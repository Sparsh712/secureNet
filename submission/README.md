# SecureNet — Autonomous SOC Agent RL Environment

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![OpenEnv](https://img.shields.io/badge/OpenEnv-Spec%20v1-purple)
![License](https://img.shields.io/badge/License-MIT-gray)

> A production-ready **Reinforcement Learning execution environment** simulating a full Security Operations Center (SOC) workflow. An AI agent autonomously investigates threats, queries logs, and contains compromised infrastructure without human intervention.

---

## Business Case

SOC analysts are drowning in alert volumes — an average enterprise generates 11,000+ security events per day. **SecureNet** trains an autonomous AI analyst to triage, investigate, and contain threats across a full cyber kill chain, reducing mean time to contain (MTTC) from hours to seconds.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    SecureNet System                          │
│                                                              │
│  ┌─────────────┐    HTTP /step    ┌──────────────────────┐  │
│  │  AI Agent   │◄───────────────►│  FastAPI SOC Engine  │  │
│  │  (DQN/LLM)  │   /reset /state  │  (Stateful POMDP)    │  │
│  └─────────────┘                  └──────────────────────┘  │
│        │                                    │               │
│  ┌─────▼──────┐                  ┌──────────▼───────────┐  │
│  │  train_dqn │                  │  Network Template    │  │
│  │  Curriculum│                  │  5 Difficulty Tiers  │  │
│  │ easy→night │                  │  Adversarial Drift   │  │
│  └────────────┘                  └──────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
submission/
├── openenv.yaml               # OpenEnv spec declaration
├── Dockerfile                 # Container definition for HF Spaces
├── models.py                  # Pydantic typed Action / Observation / Reward models
├── client.py                  # Python HTTP client helper
├── inference.py               # Playbook baseline — perfect scripted analyst (scores 1.0)
├── inference_llm.py           # LLM reasoning baseline — GPT/OpenRouter live agent
├── checkpoints/
│   ├── easy.pt                # Trained Dueling DQN weights (easy tier)
│   ├── medium.pt              # Trained Dueling DQN weights (medium tier)
│   ├── hard.pt                # Trained Dueling DQN weights (hard tier)
│   ├── critical.pt            # Trained Dueling DQN weights (critical tier)
│   ├── nightmare.pt           # Trained Dueling DQN weights (nightmare tier)
│   └── *_training.png         # Training curve graphs per tier
└── server/
    ├── __init__.py
    ├── securenet_environment.py  # Core POMDP environment logic
    ├── app.py                    # FastAPI REST API (OpenEnv compliant)
    └── requirements.txt
```

---

## Network Scenarios by Difficulty

| Tier | Nodes | Compromised | Max Steps | Special Mechanic |
|------|-------|-------------|-----------|-----------------|
| **EASY** | 2 | 1 | 20 | None |
| **MEDIUM** | 4 | 2 | 25 | Critical DB node (catastrophic if isolated) |
| **HARD** | 6 | 2 | 30 | Adversarial drift every 3 steps, CEO honeytrap |
| **CRITICAL** | 8 | 4 | 35 | AD domain compromise, rapid lateral movement |
| **NIGHTMARE** | 12 | 6 | 40 | Nation-state APT — supply chain + k8s + CI/CD |

---

## Action Space

| Action | Description | Required Fields |
|--------|-------------|-----------------|
| `query_logs` | Read syslog / PowerShell traces of a node | `target_node` |
| `analyze_process` | Inspect running processes for malicious indicators | `target_node` |
| `threat_intel` | Look up IP/hash in mock IOC threat feed | `ip_address` or `ioc` |
| `block_ip` | Block attacker IP at the firewall | `ip_address` |
| `quarantine_process` | Kill a specific process without full isolation | `target_node`, `process_name` |
| `isolate_host` | Remove node from network (containment) | `target_node` |

---

## Reward Function

| Event | Reward |
|-------|--------|
| Any step (efficiency penalty) | **-0.02** |
| `query_logs` (first time on node) | **+0.10** |
| `query_logs` (on infected node) | **+0.35** *(total)* |
| `analyze_process` (malicious found) | **+0.40** |
| `threat_intel` (matched malicious IOC) | **+0.25** |
| `block_ip` (known attacker IP) | **+0.50** |
| `quarantine_process` (malicious process) | **+0.60** |
| `isolate_host` — infected node | **+1.00** |
| `isolate_host` — healthy node | **-0.80** |
| Isolate critical healthy node | **-3.00** + `done=True` |

---

## Grader / Score

```
Containment Rate  = Correctly Isolated / Total Infected Nodes
FP Penalty        = False Positives / Total Healthy Nodes
Kill Chain Bonus  = Stages disrupted / Total Kill Chain stages
Time Bonus        = max(0, 1 - steps_used / max_steps)

Total Score = 0.5 * Containment Rate
            + 0.2 * Kill Chain Bonus
            + 0.15 * Time Bonus
            - 0.15 * FP Penalty
```

Scores range `0.0 – 1.0`. Catastrophic failure (isolating critical infrastructure) → Score = **0.0**, `done=True`.

---

## RL Training Results

| Tier | Last-500 Win Rate | Overall Win Rate |
|------|-------------------|-----------------|
| Easy | **86.1%** | 60.2% |
| Medium | **79.3%** | 38.1% |
| Hard | **53.7%** | 30.8% |
| Critical | **18.4%** | 16.8% |
| Nightmare | **1.8%** | 1.5% |

Training curves are available in `checkpoints/*_training.png`. The agent was trained using a **Dueling DQN** with curriculum learning (easy → nightmare) for 5,000 episodes per tier.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r server/requirements.txt
```

### 2. Start the SOC server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 3. Run the Playbook baseline (guaranteed 1.0 score)

```bash
export SERVER_URL=http://localhost:8000
python inference.py
```

### 4. Run the LLM Reasoning baseline

```bash
export OPENAI_API_KEY=sk-...         # or your OpenRouter key
export API_BASE_URL=https://openrouter.ai/api/v1    # optional
export MODEL_NAME=meta-llama/llama-3.3-70b-instruct:free  # optional
export SERVER_URL=http://localhost:8000
python inference_llm.py
```

---

## Docker / Hugging Face Spaces

```bash
docker build -t securenet .
docker run -p 8000:8000 securenet
```

100% in-memory — no databases, no VMs, no external APIs required for the environment itself. Runs under **2 vCPU / 8 GB RAM**.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset?difficulty=easy` | POST | Start a new episode |
| `/step` | POST | Take an action (JSON body) |
| `/state` | GET | Partial observable state |
| `/grade?task=soc_triage_easy` | GET | Get score for last episode |
| `/tasks` | GET | List all available tasks |
| `/health` | GET | Health check |
| `/kill_chain` | GET | Current MITRE ATT&CK stage |
| `/threat_intel_db` | GET | Full IOC feed |