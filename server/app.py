import os
import sys

# Support both legacy invocation (from securenet_env/) and root invocation
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Support both `from models import ...` (legacy) and `from securenet_env.models import ...`
try:
    from securenet_env.models import SecureNetAction
    from securenet_env.server.securenet_environment import SecureNetEnvironment, NETWORK_TEMPLATES, KILL_CHAIN
except ImportError:
    from models import SecureNetAction
    from server.securenet_environment import SecureNetEnvironment, NETWORK_TEMPLATES, KILL_CHAIN

app = FastAPI(
    title="SecureNet SOC RL Environment",
    version="2.0.0",
    description=(
        "Stateful POMDP cybersecurity SOC simulation. OpenEnv compliant. "
        "Implements MITRE ATT&CK kill chain with 5 difficulty tiers and adversarial drift."
    ),
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

env = SecureNetEnvironment()


@app.post("/reset")
def reset_env(difficulty: str = Query(None, description="Difficulty: easy|medium|hard|critical|nightmare")):
    """Reset the POMDP environment. Omit difficulty to use curriculum level."""
    return env.reset(difficulty)


@app.post("/step")
def step_env(action: SecureNetAction):
    """Take an action in the SOC simulation."""
    return env.step(
        action_type=action.action_type.value,
        target_node=action.target_node,
        ip_address=action.ip_address,
        timeframe=action.timeframe,
        process_name=action.process_name,
        ioc=action.ioc,
    )


@app.get("/state")
def get_state():
    """Partial observable state — agents cannot see node infection status directly."""
    visible = {
        name: {
            "node_type":   node["node_type"],
            "status":      node["status"] if node["status"] == "isolated" else "unknown",
            "critical":    node.get("critical", False),
            "connections": node.get("connections", []),
        }
        for name, node in env.network.items()
    }
    return {
        "episode":        env.episode_count,
        "step":           env.step_count,
        "max_steps":      env.max_steps,
        "difficulty":     env.difficulty,
        "curriculum":     env.curriculum_level,
        "kill_chain":     KILL_CHAIN[env.kill_chain_stage],
        "kill_chain_idx": env.kill_chain_stage,
        "network":        visible,
        "topology":       env._tmpl.get("topology", {}),
        "isolated":       env.isolated,
        "blocked_ips":    env.blocked_ips,
        "quarantined":    env.quarantined,
    }


@app.get("/grade")
def grade_task(
    task: str = Query(..., description="Task id: soc_triage_easy | soc_triage_medium | soc_triage_hard"),
):
    """
    Per-task grader endpoint — required by OpenEnv spec.
    Returns the deterministic score (0.0–1.0) for the last completed episode.
    """
    difficulty_map = {
        "soc_triage_easy":      "easy",
        "soc_triage_medium":    "medium",
        "soc_triage_hard":      "hard",
        "soc_triage_critical":  "critical",
        "soc_triage_nightmare": "nightmare",
        "easy": "easy", "medium": "medium", "hard": "hard",
        "critical": "critical", "nightmare": "nightmare",
    }
    diff = difficulty_map.get(task)
    if not diff:
        return {
            "error": (
                f"Unknown task '{task}'. "
                "Valid values: soc_triage_easy, soc_triage_medium, soc_triage_hard"
            )
        }
    return {
        "task":           task,
        "difficulty":     env.difficulty,
        "last_score":     env.score_history[-1] if env.score_history else None,
        "score_range":    [0.0, 1.0],
        "score_history":  env.score_history[-5:] if env.score_history else [],
        "episode_reward": round(env.episode_reward, 3),
        "stats":          env.stats,
    }


@app.get("/tasks")
def list_tasks():
    """Enumerate all available tasks with metadata."""
    return {
        "tasks": [
            {
                "id":                "soc_triage_easy",
                "difficulty":        "easy",
                "max_steps":         20,
                "success_threshold": 0.60,
                "description":       "2-node network — SSH brute-force + Cobalt Strike C2 on Server-1.",
                "nodes":             list(NETWORK_TEMPLATES["easy"]["nodes"].keys()),
                "compromised":       NETWORK_TEMPLATES["easy"]["compromised"],
            },
            {
                "id":                "soc_triage_medium",
                "difficulty":        "medium",
                "max_steps":         25,
                "success_threshold": 0.60,
                "description":       "4-node network — PowerShell credential theft + phishing mail server.",
                "nodes":             list(NETWORK_TEMPLATES["medium"]["nodes"].keys()),
                "compromised":       NETWORK_TEMPLATES["medium"]["compromised"],
            },
            {
                "id":                "soc_triage_hard",
                "difficulty":        "hard",
                "max_steps":         30,
                "success_threshold": 0.60,
                "description":       "6-node ransomware network — adversarial drift, CEO laptop honeytrap.",
                "nodes":             list(NETWORK_TEMPLATES["hard"]["nodes"].keys()),
                "compromised":       NETWORK_TEMPLATES["hard"]["compromised"],
            },
        ]
    }


@app.get("/threat_intel_db")
def get_ioc_db():
    """Return the full in-memory threat intelligence feed."""
    try:
        from securenet_env.server.securenet_environment import THREAT_INTEL_DB
    except ImportError:
        from server.securenet_environment import THREAT_INTEL_DB
    return THREAT_INTEL_DB


@app.get("/stats")
def get_stats():
    return env.stats


@app.get("/episode_log")
def get_episode_log():
    return {"log": env.episode_log, "episode": env.episode_count}


@app.get("/kill_chain")
def get_kill_chain():
    return {
        "stages":      KILL_CHAIN,
        "current_idx": env.kill_chain_stage,
        "current":     KILL_CHAIN[env.kill_chain_stage],
    }


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0", "service": "SecureNet SOC"}


# Serve live dashboard from either location
_this_dir = os.path.dirname(os.path.abspath(__file__))
dashboard_path = os.path.join(os.path.dirname(_this_dir), "dashboard")
if os.path.exists(dashboard_path):
    app.mount("/ui", StaticFiles(directory=dashboard_path, html=True), name="dashboard")

    @app.get("/")
    def root():
        return FileResponse(os.path.join(dashboard_path, "index.html"))
