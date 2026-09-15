#!/usr/bin/env bash
# RoboGuardian Final Audit Suite v2
# Safe: no `set -e`, safe arithmetic PASS=$((PASS+1)), python3 for JSON checks.

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
PASS=0; FAIL=0; WARN=0

ok()   { printf "${GREEN}PASS${NC}  %s\n" "$1"; PASS=$((PASS+1)); }
bad()  { printf "${RED}FAIL${NC}  %s\n" "$1"; FAIL=$((FAIL+1)); }
warn() { printf "${YELLOW}WARN${NC}  %s\n" "$1"; WARN=$((WARN+1)); }

echo "=== SECTION 1: Git & repo ==="
if git rev-parse --git-dir >/dev/null 2>&1; then ok "git repository"; else bad "not a git repo"; fi
if git diff-index --quiet HEAD -- 2>/dev/null; then ok "working tree clean"; else warn "uncommitted changes"; fi
BAK=$(find . -path ./.git -prune -o \( -name '*.bak' -o -name '*.save' \) -print 2>/dev/null | wc -l)
if [ "$BAK" -eq 0 ]; then ok "no .bak/.save files"; else bad "found $BAK backup files"; fi

echo "=== SECTION 2: Artifacts ==="
for f in results/g1_environment.json results/g2_baseline.json results/g3_failure.json \
         results/g4_detection.json results/g5_recovery.json results/g6_reliability.json \
         results/g7_openvino.json README.md results/FINAL_REPORT.md \
         docs/DEMO_SCRIPT.md docs/VIDEO_OUTLINE.md docs/SLIDE_DECK.md \
         docs/PROJECT_DESCRIPTION.md docs/SUBMISSION_CHECKLIST.md; do
  if [ -f "$f" ]; then ok "exists: $f"; else bad "missing: $f"; fi
done
for f in results/g8_robustness_matrix.json results/g8_latency_budget.json docs/SIM_TO_REAL.md; do
  if [ -f "$f" ]; then ok "exists: $f"; else warn "missing (hardening): $f"; fi
done

echo "=== SECTION 3: Evidence validation ==="
G6=$(python3 -c "import json;d=json.load(open('results/g6_reliability.json'));s=d['summary'];print(bool(d.get('g6_pass')) and s['A']['task_success_rate']>=0.9 and s['B']['task_success_rate']<=0.2 and s['C']['task_success_rate']>=0.8 and s['C']['stale_action_execution_rate']==0.0)" 2>/dev/null || echo False)
if [ "$G6" = "True" ]; then ok "G6 evidence valid (A>=90%, B<=20%, C>=80%, stale=0)"; else bad "G6 evidence invalid or missing"; fi

H1=$(python3 -c "import json;d=json.load(open('results/g8_robustness_matrix.json'));conds=d.get('conditions',{});stale_ok=all(  all((not ('stale' in k.lower())) or (not isinstance(v,(int,float))) or v==0       for k,v in (c.items() if isinstance(c,dict) else [])  )  for c in conds.values());print(bool(d.get('g8_h1_pass')) and stale_ok)" 2>/dev/null || echo False)
if [ "$H1" = "True" ]; then ok "H1 robustness valid (stale=0 everywhere)"; else warn "H1 evidence missing/invalid"; fi

echo "=== SECTION 4: Claim audit ==="
FR_TMP=$(mktemp)
sed -n '1,/## Claim Revision Log/p' results/FINAL_REPORT.md > "$FR_TMP" 2>/dev/null || cp results/FINAL_REPORT.md "$FR_TMP" 2>/dev/null
if grep -qiE "faster than ungoverned|A formal proof, via|H0 rejected\. H1 accepted|strictly dominates" README.md "$FR_TMP" docs/PROJECT_DESCRIPTION.md 2>/dev/null; then
  bad "forbidden claim pattern found:"
  grep -niE "faster than ungoverned|A formal proof, via|H0 rejected\. H1 accepted|strictly dominates" README.md "$FR_TMP" docs/PROJECT_DESCRIPTION.md 2>/dev/null | head -5
else
  ok "claim audit clean (revision-log quotes excluded)"
fi
rm -f "$FR_TMP"
for p in "more resilient than open-loop" "controlled simulation evidence" "bimanual dual SO-101"; do
  if grep -rq "$p" README.md results/FINAL_REPORT.md 2>/dev/null; then ok "required claim present: $p"; else bad "required claim missing: $p"; fi
done

echo "=== SECTION 5: Invariants ==="
if [ -f tests/test_recovery_invariants.py ]; then
  if python3 -m pytest tests/test_recovery_invariants.py -q >/dev/null 2>&1; then ok "invariant tests pass"; else bad "invariant tests FAIL"; fi
else
  warn "tests/test_recovery_invariants.py missing"
fi
if grep -q "stale_target=dict(stale_target)" control/recovery.py 2>/dev/null; then ok "Sec18: stale logged in trace"; else bad "Sec18: trace logging missing"; fi
if grep -Eq "data\.ctrl\[.*\] *= *[^=]*stale_target" control/recovery.py 2>/dev/null; then bad "Sec18 VIOLATION: stale written to ctrl"; else ok "Sec18: stale never written to ctrl"; fi

echo "=== SECTION 6: Reproducibility & CI ==="
if [ -f requirements.txt ]; then ok "requirements.txt"; else warn "requirements.txt missing"; fi
if [ -f requirements-lock.txt ]; then ok "requirements-lock.txt (pinned)"; else warn "requirements-lock.txt missing"; fi
if grep -q "manual_seed" perception/train_detector.py 2>/dev/null; then ok "torch seed present"; else warn "torch seed missing in train_detector.py"; fi
if [ -f .github/workflows/ci.yml ]; then ok "CI workflow exists"; else warn "CI workflow missing"; fi

echo "==============================="
echo "PASS=$PASS FAIL=$FAIL WARN=$WARN"
if [ "$FAIL" -eq 0 ]; then echo "RESULT: READY (no failures)"; exit 0; else echo "RESULT: NOT READY ($FAIL failures)"; exit 1; fi
