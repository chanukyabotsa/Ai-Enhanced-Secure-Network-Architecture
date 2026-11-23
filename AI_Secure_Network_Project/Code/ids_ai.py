#!/usr/bin/env python3
"""
ids_ai.py - Dual-layer detection: Suricata (signatures) + AI model (behaviour)

For every connection (flow) that reaches the server segment, this script:
  1. Reads the flow record from Suricata's eve.json
  2. Builds the same 6 features the model was trained on (CIC-IDS2017 style)
  3. Scores it with the trained MLP (weights exported from PyTorch, run with NumPy;
     equivalence verified in Colab: same decision on 100% of 20,000 flows)
  4. Shows Suricata's verdict and the AI's verdict + confidence side by side

Usage (run on the Ubuntu server, from the folder that has ids_model_weights.json):
  sudo python3 ids_ai.py report   # analyse everything already in eve.json, then a summary
  sudo python3 ids_ai.py live     # watch new traffic in real time (Ctrl+C to stop)

Notes / honest limitations:
  - "Total Length of Fwd Packets" in CIC-IDS2017 counts payload bytes; Suricata's
    bytes_toserver includes headers, so this feature is an approximation.
  - Confidence = the model's real output probability P(attack). Nothing is invented.
  - The script only RECOMMENDS a block; a human confirms and applies it in pfSense.
"""
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

EVE = "/var/log/suricata/eve.json"
HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "ids_model_weights.json")
OUT = os.path.join(HERE, "ai_results.jsonl")
SERVER_NET = "192.168.43."          # only score traffic going TO the server segment
REPORT_PRINT_LIMIT = 40             # report mode: print at most this many flagged flows

# ---------------- load the trained model ----------------
m = json.load(open(WEIGHTS))
W1, b1, W2, b2, W3, b3 = (np.array(m[k], dtype=float) for k in ("W1", "b1", "W2", "b2", "W3", "b3"))
MEAN = np.array(m["scaler_mean"], dtype=float)
SCALE = np.array(m["scaler_scale"], dtype=float)
THR = float(m["threshold"])


def ai_score(x):
    """Return P(attack) for one flow's 6 features (same maths as the PyTorch model)."""
    z = (np.array(x, dtype=float) - MEAN) / SCALE      # same StandardScaler as training
    h = np.maximum(0, W1 @ z + b1)                      # Linear(6,32) + ReLU
    h = np.maximum(0, W2 @ h + b2)                      # Linear(32,16) + ReLU
    logit = float((W3 @ h + b3)[0])                     # Linear(16,1)
    logit = max(min(logit, 500.0), -500.0)
    return 1.0 / (1.0 + np.exp(-logit))                 # sigmoid


def parse_ts(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f%z")


def features(ev):
    """Map a Suricata flow event to the 6 CIC-IDS2017 features used in training."""
    f = ev.get("flow", {})
    try:
        dur_us = (parse_ts(f["end"]) - parse_ts(f["start"])).total_seconds() * 1e6
    except Exception:
        dur_us = 0.0
    dur_s = max(dur_us / 1e6, 1e-6)
    pts, ptc = f.get("pkts_toserver", 0), f.get("pkts_toclient", 0)
    bts, btc = f.get("bytes_toserver", 0), f.get("bytes_toclient", 0)
    return [
        ev.get("dest_port", 0),        # Destination Port
        dur_us,                        # Flow Duration (microseconds)
        pts,                           # Total Fwd Packets
        bts,                           # Total Length of Fwd Packets (approx.)
        (bts + btc) / dur_s,           # Flow Bytes/s
        (pts + ptc) / dur_s,           # Flow Packets/s
    ]


# ---------------- correlation state ----------------
alerts = {}                            # flow_id -> set of Suricata signatures
stats = {"BOTH": 0, "SURICATA ONLY": 0, "AI ONLY": 0, "CLEAN": 0}
flagged_src = {}                       # src_ip -> number of suspicious flows
recommended = set()                    # src IPs we already printed a recommendation for
printed = 0


def handle(line, out, live):
    global printed
    try:
        ev = json.loads(line)
    except ValueError:
        return
    et = ev.get("event_type")

    if et == "alert":
        alerts.setdefault(ev.get("flow_id"), set()).add(ev["alert"]["signature"])
        return
    if et != "flow" or not str(ev.get("dest_ip", "")).startswith(SERVER_NET):
        return

    p = ai_score(features(ev))
    ai_attack = p >= THR
    sigs = alerts.pop(ev.get("flow_id"), set())
    sur = bool(sigs) or bool(ev.get("flow", {}).get("alerted", False))

    if sur and ai_attack:
        verdict = "BOTH"
    elif sur:
        verdict = "SURICATA ONLY"
    elif ai_attack:
        verdict = "AI ONLY"
    else:
        verdict = "CLEAN"
    stats[verdict] += 1

    src = ev.get("src_ip")
    if verdict != "CLEAN":
        flagged_src[src] = flagged_src.get(src, 0) + 1

    out.write(json.dumps({
        "time": ev.get("timestamp"), "src": src, "dest": ev.get("dest_ip"),
        "dport": ev.get("dest_port"), "proto": ev.get("proto"),
        "suricata_alert": sorted(sigs) if sigs else sur,
        "ai_attack": bool(ai_attack), "ai_prob_attack": round(p, 4), "verdict": verdict,
    }) + "\n")

    show = live or (verdict != "CLEAN" and printed < REPORT_PRINT_LIMIT)
    if not show:
        return
    printed += 1
    sig_txt = "; ".join(sorted(sigs))[:55] if sigs else ("(alerted)" if sur else "-")
    print(f"{str(ev.get('timestamp'))[:19]}  {src} -> {ev.get('dest_ip')}:{ev.get('dest_port', '-')}/{ev.get('proto')}")
    print(f"    Suricata: {sig_txt}")
    print(f"    AI      : {'ATTACK' if ai_attack else 'normal'}  (P(attack) = {p * 100:.1f}%)   =>  {verdict}")
    if verdict != "CLEAN" and src not in recommended:
        recommended.add(src)
        print(f"    >> RECOMMEND: investigate {src}; if confirmed, block it in pfSense")
    print()


def summary():
    total = sum(stats.values())
    print("=" * 60)
    print("SUMMARY - flows to the server segment")
    for k, v in stats.items():
        print(f"  {k:<14}: {v}")
    print(f"  {'TOTAL':<14}: {total}")
    if total:
        print(f"\n  AI flagged      : {stats['BOTH'] + stats['AI ONLY']}")
        print(f"  Suricata flagged: {stats['BOTH'] + stats['SURICATA ONLY']}")
        print(f"  Caught ONLY by AI (missed by Suricata rules): {stats['AI ONLY']}")
    if flagged_src:
        print("\nTop suspicious sources:")
        for ip, c in sorted(flagged_src.items(), key=lambda x: -x[1])[:5]:
            print(f"  {ip}: {c} suspicious flows")
    print(f"\nPer-flow results saved to {OUT}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "report"
    print(f"Model loaded: {len(MEAN)} features, threshold {THR}")
    with open(OUT, "a") as out, open(EVE) as f:
        if mode == "live":
            f.seek(0, 2)  # start at the end: only new traffic
            print("LIVE mode - watching new traffic to the server segment (Ctrl+C to stop)\n")
            try:
                while True:
                    line = f.readline()
                    if not line:
                        time.sleep(0.5)
                        continue
                    handle(line, out, live=True)
                    out.flush()
            except KeyboardInterrupt:
                print()
        else:
            print("REPORT mode - analysing existing eve.json ...\n")
            for line in f:
                handle(line, out, live=False)
    summary()


if __name__ == "__main__":
    main()
