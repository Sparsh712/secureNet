import sys, os, time, json
from client import SecureNetClient
from openai import OpenAI

# Required Environment Variables
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY      = os.getenv("OPENAI_API_KEY", os.getenv("HF_TOKEN", "mock-key"))

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str):
    print(f"[STEP] step={step} action={action!r} reward={reward:+.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list):
    rewards_str = ",".join(f"{r:+.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

SYSTEM_PROMPT = """You are an elite SOC cybersecurity analyst tasked with containing a live incident in a corporate network. 
The environment is a POMDP. You must explore the nodes, find the compromised hosts, and remediate them.

Available Actions:
- query_logs: Fetch system/network logs. (Requires 'target_node')
- analyze_process: View running processes. (Requires 'target_node')
- threat_intel: Lookup an IP/IoC. (Requires 'ip_address' or 'ioc')
- block_ip: Block a malicious IP. (Requires 'ip_address')
- quarantine_process: Terminate a specific process. (Requires 'target_node' and 'process_name')
- isolate_host: Isolate a node from the network. (Requires 'target_node')

CRITICAL RULES:
1. Isolating healthy hosts gives a massive penalty (-0.5). Only isolate if you are certain it's compromised.
2. Querying logs and threat intel yields positive reward (+0.2). Give it time to see results.
3. Don't repeat the exact same action repeatedly unless something changed.
4. If you see lateral movement, isolate the Patient Zero first to stop the spread.

You MUST respond with a valid JSON object ONLY, containing your action. 
Example 1: {"action_type": "query_logs", "target_node": "Server-1"}
Example 2: {"action_type": "threat_intel", "ip_address": "45.33.32.156"}
Example 3: {"action_type": "quarantine_process", "target_node": "Workstation-B", "process_name": "mimikatz.exe"}
"""

def extract_json(response: str) -> dict:
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end != -1:
            return json.loads(response[start:end])
    except Exception:
        pass
    # Safest fallback if the LLM hallucinates formatting
    return {"action_type": "query_logs", "target_node": "unknown-error"}

def run_scenario(client: SecureNetClient, difficulty: str, openai_client: OpenAI):
    task = f"soc_triage_{difficulty}"
    log_start(task=task, env="securenet", model=MODEL_NAME)

    obs = client.reset(difficulty)
    rewards = []
    steps = 0
    done = obs.get("done", False)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"New incident deployed. Difficulty: {difficulty.upper()}.\nObservation: {json.dumps(obs.get('result', 'Started.'), indent=2)}\n\nWhat is your first action? Respond with JSON."}
    ]

    while not done and steps < 30: # Hard cap at 30 steps to prevent infinite loop bleeding
        steps += 1
        
        try:
            response = openai_client.chat.completions.create(
                model=MODEL_NAME, 
                messages=messages,
                temperature=0.2, # Low temp for deterministic logic
            )
            llm_text = response.choices[0].message.content
        except Exception as exc:
            print(f"[DEBUG] Model failed or timeout: {exc}")
            break
            
        action_payload = extract_json(llm_text)
        act_type = action_payload.get("action_type", "query_logs")
        target   = action_payload.get("target_node")
        ip       = action_payload.get("ip_address")
        proc     = action_payload.get("process_name")
        ioc      = action_payload.get("ioc")
        
        # Build the exact string format for strict compliance logging
        act_str = f"{act_type}:{target or ip or proc or ioc or ''}"
        
        # Issue environment action
        obs = client.step(act_type, target, ip, None, proc, ioc)
        r = obs.get("reward", 0.0)
        d = obs.get("done", False)
        e = obs.get("error")
        rewards.append(r)
        done = d

        # Required Competition output spec
        log_step(step=steps, action=act_str, reward=r, done=d, error=e)
        
        # Append to LLM memory context for next loop
        messages.append({"role": "assistant", "content": llm_text})
        messages.append({
            "role": "user", 
            "content": f"Result: {obs.get('result')}\nReward: {r}\nError: {e}\n\nWhat is your next action? Respond with JSON."
        })
        
        # Prune context window to prevent blowing out context limits
        if len(messages) > 15:
            messages = [messages[0]] + messages[-12:]

        time.sleep(0.5)

    info = obs.get("info", {})
    score = info.get("total_score", 0.0)
    success = score >= 0.70 if done else False
    log_end(success=success, steps=steps, score=score, rewards=rewards)
    
    if "total_score" in info:
        print(f"\n  ┌── Grader Report [{difficulty.upper()}] ─────────────────┐")
        print(f"  │  Total Score        : {info['total_score']:.3f}")
        print(f"  │  Containment Rate   : {info.get('containment_rate', 0):.3f}")
        print(f"  │  Elapsed (s)        : {info.get('elapsed_seconds', 0):.1f}")
        print(f"  └─────────────────────────────────────────────┘")


def main():
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("HF_TOKEN"):
        sys.stdout.reconfigure(encoding='utf-8')
        print("Warning: Please set OPENAI_API_KEY environment variable. If using a local LLM API (like Ollama or vLLM), set API_BASE_URL.")
        
    openai_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env_client = SecureNetClient(os.getenv("SERVER_URL", "http://localhost:7860"))

    try:
        h = env_client.health()
        sys.stdout.reconfigure(encoding='utf-8')
        print(f"SERVER_HEALTH: Online \u2014 {h}")
    except Exception as e:
        sys.stdout.reconfigure(encoding='utf-8')
        print(f"SERVER_HEALTH_ERROR: {e}")
        sys.exit(1)

    tiers = os.getenv("TIERS", "easy,medium,hard").split(",")

    print("\n" + "═" * 60)
    print("  SecureNet SOC — TRUE LLM REASONING AGENT")
    print("═" * 60)

    for diff in tiers:
        print(f"\n{'─'*60}")
        print(f"  TIER: {diff.upper()}")
        print(f"{'─'*60}")
        run_scenario(env_client, diff, openai_client)

if __name__ == "__main__":
    main()
