from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# OpenEnv spec aliases — validators expect these exact names
Action      = None   # assigned below after class definition
Observation = None   # assigned below
Reward      = None   # assigned below


class ActionType(str, Enum):
    QUERY_LOGS         = "query_logs"
    ANALYZE_PROCESS    = "analyze_process"
    ISOLATE_HOST       = "isolate_host"
    BLOCK_IP           = "block_ip"
    THREAT_INTEL       = "threat_intel"     # Lookup IP/hash in mock IOC feed
    QUARANTINE_PROCESS = "quarantine_process"  # Kill a process without full isolation


class SecureNetAction(BaseModel):
    action_type: ActionType = Field(..., description="The cybersecurity tool to invoke.")
    target_node: Optional[str] = Field(None, description="Hostname to target.")
    timeframe:   Optional[str] = Field(None, description="Timeframe e.g. 'last_hour'.")
    ip_address:  Optional[str] = Field(None, description="IP to block or look up.")
    process_name: Optional[str] = Field(None, description="Process name for quarantine.")
    ioc:         Optional[str] = Field(None, description="IOC hash/IP/domain to query threat intel.")


class SecureNetObservation(BaseModel):
    result:  str   = Field(..., description="System output, log lines, or command feedback.")
    success: bool  = Field(..., description="Whether the action executed successfully.")
    reward:  float = Field(0.0)
    done:    bool  = Field(False)
    error:   str   = Field("")
    info:    Dict[str, Any] = Field(default_factory=dict)


class SecureNetReward(BaseModel):
    """Typed reward model required by OpenEnv spec."""
    value:         float = Field(..., ge=0.0, le=1.0,  description="Normalised episode score (0.0–1.0).")
    partial_score: float = Field(0.0,                  description="Cumulative intermediate reward this episode.")
    step_reward:   float = Field(0.0,                  description="Reward earned on this step.")
    reason:        str   = Field("",                   description="Human-readable explanation of this reward.")


class StepResult(BaseModel):
    """Typed return value of step() — wraps observation, reward, done, info."""
    observation: SecureNetObservation
    reward:      SecureNetReward
    done:        bool            = Field(False)
    info:        Dict[str, Any]  = Field(default_factory=dict)


# ── OpenEnv spec-required top-level aliases ──────────────────────────────
Action      = SecureNetAction
Observation = SecureNetObservation
Reward      = SecureNetReward
