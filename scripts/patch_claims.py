#!/usr/bin/env python3
"""Patch FINAL_REPORT.md v2 — Founder-ratified 2026-09-15

Reads FINAL_REPORT.md, applies exact replacements based on actual file content,
appends Scope of Validity + Claim Revision Log, verifies no forbidden patterns remain.
"""
from pathlib import Path
import sys
import re

p = Path("results/FINAL_REPORT.md")
text = p.read_text(encoding="utf-8")
original = text

# --- EXACT REPLACEMENTS (based on grep output of actual file) ---

# 1. Thesis headline (line 3)
text = text.replace(
    '> **"Governed AI is faster than ungoverned AI under uncertainty."**',
    '> **"Governed execution is more resilient than open-loop execution when the world changes."**'
)

# 2. Opening sentence (line 5)
text = text.replace(
    '> A formal proof, via simulation, that a governed control loop with',
    '> Controlled simulation evidence that a governed control loop with'
)

# 3. "outperforms" -> "outperformed" (around line 6)
text = text.replace(
    'observe → evaluate → re-plan semantics outperforms an open-loop',
    'observe → evaluate → re-plan semantics outperformed an open-loop'
)

# 4. Core Thesis section (line 42)
text = text.replace(
    'Governed control is strictly superior under uncertainty.',
    'Governed control is more resilient than open-loop control under uncertainty.'
)

# 5. H0/H1 rejection lines (lines 54, 73) — handle both bold and plain variants
text = text.replace(
    '**H0 rejected. H1 accepted.**',
    '**Result: the evidence supports H1 under the tested scenario. This is controlled simulation evidence, not a formal proof; no statistical test was performed (trials are deterministic re-runs).**'
)
text = text.replace(
    'H0 rejected. H1 accepted.',
    'Result: the evidence supports H1 under the tested scenario. This is controlled simulation evidence, not a formal proof; no statistical test was performed (trials are deterministic re-runs).'
)

# 6. "strictly dominates" (line 298)
text = text.replace(
    'semantics strictly dominates',
    'semantics outperformed'
)

# 7. Conclusion line
text = text.replace(
    '**Core thesis: PROVEN. ✅**',
    '**Core thesis: SUPPORTED by controlled simulation evidence. ✅**'
)
text = text.replace(
    'Core thesis: PROVEN.',
    'Core thesis: SUPPORTED by controlled simulation evidence.'
)

# 8. "fails 100% of the time" — add context
text = text.replace(
    'fails 100% of the time.',
    'fails 100% of the time (10 deterministic trials per arm).'
)

# 9. Arm description
text = text.replace(
    '6-DOF SO-101 arm manipulating a plate on a table.',
    'bimanual dual SO-101 simulation environment (experiments exercise the left arm manipulating a plate on a table).'
)

# --- VERIFY REPLACEMENTS TOOK EFFECT ---
if text == original:
    print("HALT: no replacements were applied (fail-closed)")
    sys.exit(1)

# --- APPEND SCOPE + REVISION LOG ---
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
| 2026-09-15 | "strictly dominates" -> "outperformed" | same as above |
| 2026-09-15 | thesis "faster" -> "more resilient" (README + FINAL_REPORT) | evidence measures success/recovery, not speed; G7 shows OpenVINO 0.44x |
| 2026-09-15 | "6-DOF SO-101 arm" -> "bimanual dual SO-101 simulation" | match actual repo infrastructure and challenge framing |
"""

if "Scope of Validity" not in text:
    text = text.rstrip() + "\n" + SCOPE

p.write_text(text, encoding="utf-8")
print("OK: replacements applied + scope/revision log appended")

# --- FAIL-CLOSED VERIFICATION ---
# Check that forbidden patterns are gone (except in legitimate contexts)
forbidden_in_wrong_context = [
    (r'faster than ungoverned', "thesis headline"),
    (r'A formal proof, via simulation', "opening sentence"),
    (r'H0 rejected\. H1 accepted', "H0/H1 rejection"),
    (r'strictly dominates', "dominance claim"),
]

violations = []
for pattern, label in forbidden_in_wrong_context:
    if re.search(pattern, text):
        violations.append(f"  - {label}: still present")

if violations:
    print("HALT: forbidden patterns still present after patch:")
    for v in violations:
        print(v)
    sys.exit(1)

print("PASS: no forbidden patterns remain in wrong context")
