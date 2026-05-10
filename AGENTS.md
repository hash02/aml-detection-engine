# AML Engine Agent Instructions

## Mission

The AML Engine is a proof-grade blockchain risk detection project. The caretaker job is to keep the engine verifiable, documented, demo-ready, and current without inflating claims.

## Source Of Truth

- Canonical repo: `hash02/aml-detection-engine`
- Public README: `README.md`
- Rule/capability manifest: `30-infrastructure/aml-engine/RULES_MANIFEST.md` when available from the parent vault, or repo-local docs/manifests when present.
- Engine source: `engine/`
- Tests: `tests/`
- Threat feeds: `addresses/` and `addresses/MANIFEST.json`

## Hard Rules

- Compute counts, do not quote cached memory.
- Public rule counts must match a canonical manifest or source inspection.
- Do not edit `.env` or commit secrets.
- Do not weaken tests to make a claim pass.
- Do not add new detection claims without fixture evidence.
- Demo changes must preserve explainability.
- If a metric changes, update README and supporting docs together.

## Caretaker Loop

Run this when asked to maintain AML Engine:

1. Check `git status --short --branch`.
2. Run the available test suite or the narrow relevant tests.
3. Verify rule count and manifest consistency.
4. Check threat-feed freshness and address manifest timestamps.
5. Produce a short caretaker report:
   - test status
   - rule/ability count
   - threat feed freshness
   - demo readiness
   - documentation drift
   - next repair

## Improvement Workflow

1. Start with a failing test or a documented false negative/false positive.
2. Add the smallest rule, feature, or data source needed.
3. Keep output explainable for compliance users.
4. Update manifest and README if the public claim changes.
5. Run tests.
6. Commit with a clear evidence trail.

## Do Not Touch Without Approval

- Secrets, `.env`, API keys, and deployment credentials.
- Public benchmark numbers unless recomputed.
- Streamlit deployment config unless the task is deployment.
- Any logic that changes scoring thresholds without before/after evidence.

## Useful Reports

- Weekly AML health report.
- Rule manifest drift report.
- Threat feed freshness report.
- Demo readiness report.
- False-positive/false-negative review.

