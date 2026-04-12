import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
from server.securenet_environment import SecureNetEnvironment
import time

def test_easy_scenario():
    print("==================================================")
    print("      TESTING THE ENVIRONMENT (EASY TIER)")
    print("==================================================")
    
    env = SecureNetEnvironment()
    obs = env.reset(difficulty="easy")
    print(f"\n[ENV RESET]\n{obs['result']}\n")

    steps = [
        {"action_type": "query_logs", "target_node": "Server-1", "ip_address": None, "process_name": None},
        {"action_type": "analyze_process", "target_node": "Server-1", "ip_address": None, "process_name": None},
        {"action_type": "block_ip", "target_node": None, "ip_address": "10.0.0.99", "process_name": None},
        {"action_type": "block_ip", "target_node": None, "ip_address": "45.33.32.156", "process_name": None},
        {"action_type": "quarantine_process", "target_node": "Server-1", "ip_address": None, "process_name": "/tmp/.x (malicious)"},
        {"action_type": "isolate_host", "target_node": "Server-1", "ip_address": None, "process_name": None},
    ]

    for step_num, action in enumerate(steps, 1):
        time.sleep(0.5)
        print(f"👉 ACTION {step_num}: {action['action_type'].upper()} targeting {action.get('target_node') or action.get('ip_address')}")
        obs = env.step(**action)
        print(f"   REWARD: {obs['reward']:+.2f}")
        print(f"   RESULT: {obs['result']}")
        
        if obs['done']:
            print("\n==================================================")
            print("   EPISODE FINISHED! FINAL SCORE SUMMARY:")
            print("==================================================")
            info = obs.get("info", {})
            for key, val in info.items():
                if key in ["total_score", "containment_rate", "kill_chain_disruption", "false_positives"]:
                    print(f"   {key.upper()}: {val}")
            print("\nEnvironment is working correctly!")
            break

if __name__ == '__main__':
    test_easy_scenario()
