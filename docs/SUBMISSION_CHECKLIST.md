# G8 Submission Checklist

## Claims audit (run before submit)
- [ ] `grep -rniE "faster than|formal proof|strictly domin|H0 rejected" README.md results/FINAL_REPORT.md docs/ || echo "CLAIM AUDIT CLEAN"`
- [ ] Thesis wording = resilience (README line 3)
- [ ] FINAL_REPORT contains "Scope of Validity" + "Claim Revision Log"
- [ ] Arm description = bimanual dual SO-101 simulation (README + FINAL_REPORT)

## Repo hygiene
- [ ] `results/` artifacts committed (g1–g7 JSON, events/, RESULTS.md, FINAL_REPORT.md)
- [ ] `docs/` contains: G0_CHECK.md, DEMO_SCRIPT.md, VIDEO_OUTLINE.md,
      SLIDE_DECK.md, PROJECT_DESCRIPTION.md, SUBMISSION_CHECKLIST.md
- [ ] README renders on GitHub (tables + code fences intact)
- [ ] No secrets/tokens in repo (`git grep -i "token\|ghp_" || true`)

## Submission assets
- [x] Video ≤ 5:00 recorded per docs/VIDEO_OUTLINE.md, uploaded & linked in README — evidence: presentation/roboguardian_pitch.mp4 (2:40 / ffprobe 160.44s), README 'Submission Assets (G8)' (verified 2026-09-15; platform upload at submission time)
- [x] Slide deck built per docs/SLIDE_DECK.md (PDF exported) — evidence: presentation/RoboGuardian_Governed_Execution_Deck_v2.pdf (verified 2026-09-15)
- [ ] Project description pasted into submission form (docs/PROJECT_DESCRIPTION.md)
- [ ] Live-demo dry run completed per docs/DEMO_SCRIPT.md (timing ≤ 10 min)

## Final pre-submit
- [ ] `git status` clean; pushed to `origin/main`
- [ ] Repo public; About section filled; topics added
- [ ] Killer-demo line appears verbatim in README, slides, video caption

## Gate status
- Engineering core: ✅ G1–G7
- Claims: ✅ tightened (2026-09-15, Founder-ratified)
- Submission assets: ✅ scaffolding in docs/
- **G8: closes when all boxes above are checked and Founder ratifies.**
