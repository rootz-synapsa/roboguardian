#!/usr/bin/env python3
"""Patch FINAL_REPORT.md — Founder-ratified 2026-09-15"""
from pathlib import Path
import sys

p = Path("results/FINAL_REPORT.md")
text = p.read_text(encoding="utf-8")

REPLACEMENTS = [
    ("A formal proof, via simulation, that a governed control loop with",
     "Controlled simulation evidence that a governed control loop with"),
    ("observe → evaluate → re-plan semantics outperforms an open-loop",
     "observe → evaluate → re-plan semantics outperformed an open-loop"),
    ("baseline when the world changes mid-execution.",
     "baseline in the tested disturbance condition (object moved mid-execution)."),
    ("Governed control is strictly superior under uncertainty.",
     "Governed control is more resilient than open-loop control under uncertainty."),
    ("H0 rejected. H1 accepted.",
     "Result: the evidence supports H1 under the tested scenario. This is\ncontrolled simulation evidence, not a formal proof; no statistical test\nwas performed (trials are deterministic re-runs)."),
    ("**strictly dominates**\nan open-loop baseline when the world changes mid-execution.",
     "**outperformed**\nthe open-loop baseline in the tested disturbance condition."),
    ("Core thesis: PROVEN.",
     "Core thesis: SUPPORTED by controlled simulation evidence."),
    ("fails 100% of the time.",
     "fails 100% of the time (10 deterministic trials per arm)."),
    ("6-DOF SO-101 arm manipulating a plate on a table.",
     "bimanual dual SO-101 simulation environment (experiments exercise\nthe left arm manipulating a plate on a table)."),
]

missing = [old for old, _ in REPLACEMENTS if old not in text]
if missing:
    print("HALT: patterns not found (fail-closed):")
    for m in missing:
        print("  -", m[:70])
    sys.exit(1)

for old, new in REPLACEMENTS:
    text = text.replace(old, new)

SCOPE = """

## Scope of Validity

All quantitative claims hold under the tested configuration only:
single disturbance type (`object_move`, +0.10 m in X, fired at
`before_grasp`), single place target, deterministic seeds
(`noise_std=0`), kinematic carry grasping model, CPU execution, MuJoCo
simulation. "10/10" expresses reproducibility of deterministic re-runs,
not a statistical confidence interval.

## Claim Revision Log

| Date | Change | Reason |
|---|---|---|
| 2026-09-15 | "formal proof" -> "controlled simulation evidence" | Founder review: 10 deterministic trials cannot constitute a proof |
| 2026-09-15 | "H0 rejected / H1 accepted" -> "supports H1 under tested scenario" | same as above |
| 2026-09-15 | "strictly dominates" -> "outperformed ... in tested condition" | same as above |
| 2026-09-15 | thesis "faster" -> "more resilient" (README) | evidence measures success/recovery, not speed; G7 shows OpenVINO 0.44x |
| 2026-09-15 | "6-DOF SO-101 arm" -> "bimanual dual SO-101 simulation" | match actual repo infrastructure and challenge framing |
"""

if "Scope of Validity" not in text:
    text = text.rstrip() + "\n" + SCOPE

p.write_text(text, encoding="utf-8")
print(f"OK: {len(REPLACEMENTS)} replacements applied + scope/revision log appended")
