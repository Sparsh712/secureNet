import requests
from requests.exceptions import ConnectionError as ReqConnError
from models import SecureNetAction, ActionType


class SecureNetClient:
    """HTTP client for the SecureNet SOC RL environment server."""

    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url.rstrip("/")

    # ── Internal ────────────────────────────────────────────────────
    def _get(self, path: str) -> dict:
        try:
            r = requests.get(f"{self.base_url}{path}", timeout=10)
            r.raise_for_status()
            return r.json()
        except ReqConnError:
            raise RuntimeError(f"Cannot connect to SecureNet at {self.base_url}. Is it running?")

    def _post(self, path: str, **kwargs) -> dict:
        try:
            r = requests.post(f"{self.base_url}{path}", timeout=10, **kwargs)
            r.raise_for_status()
            return r.json()
        except ReqConnError:
            raise RuntimeError(f"Cannot connect to SecureNet at {self.base_url}. Is it running?")

    # ── Public API ──────────────────────────────────────────────────
    def reset(self, difficulty: str = "easy") -> dict:
        """Reset the environment. difficulty: easy|medium|hard|critical|nightmare"""
        return self._post("/reset", params={"difficulty": difficulty})

    def step(
        self,
        action_type:  str,
        target_node:  str  = None,
        ip_address:   str  = None,
        timeframe:    str  = None,
        process_name: str  = None,
        ioc:          str  = None,
    ) -> dict:
        """
        Take one action in the environment.

        action_type choices:
          query_logs          – target_node required
          analyze_process     – target_node required
          quarantine_process  – target_node + process_name required
          isolate_host        – target_node required
          block_ip            – ip_address required
          threat_intel        – ioc (IP / hash / domain) required
        """
        payload = SecureNetAction(
            action_type=ActionType(action_type),
            target_node=target_node,
            ip_address=ip_address,
            timeframe=timeframe,
            process_name=process_name,
            ioc=ioc,
        ).model_dump()
        return self._post("/step", json=payload)

    def state(self) -> dict:
        """Return the partial observable state (no hidden infection flags)."""
        return self._get("/state")

    def stats(self) -> dict:
        """Return aggregate training statistics."""
        return self._get("/stats")

    def episode_log(self) -> dict:
        """Return the current episode's step-by-step action log."""
        return self._get("/episode_log")

    def threat_intel(self) -> dict:
        """Return the full in-memory IOC threat intelligence feed."""
        return self._get("/threat_intel_db")

    def health(self) -> dict:
        """Health check."""
        return self._get("/health")
