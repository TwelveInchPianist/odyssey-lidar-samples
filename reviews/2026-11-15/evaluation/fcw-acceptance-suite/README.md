# FCW acceptance suite v1

A frozen reusable offline suite for future releases. It combines17 fresh raw windows and5 older raw windows, exact wall/parked-car/parked-van alert controls, a partial-body-overlap counterexample and stale/incomplete/time-invalid input checks. Inputs and baseline receipts are hashed. No recording or installed application is changed by the runner.

Run `run.py --source <candidate source directory> --output <new output directory>` with the existing project Python environment. It executes recorded replays and fails if positive alerts change or a synthetic hard gate fails. `--reuse-frozen-baseline` is permitted only for an exact source hash match and verified historical result hashes, avoiding duplicate baseline computation. A candidate whose behavior intentionally changes a positive needs explicit review of that case, not silently rewritten expected output.

Nuisance-window alert counts are metrics, not ground-truth false positives. Human/cyclist/animal reports and runtime fault receipts are registered review controls. Automated VRU/animal recall, calibrated clearance, battery/thermal endurance, physical audio and on-road safety are NOT covered. `hard_controls_pass` must not be presented as complete perception qualification or automatic deployment approval.

The stronger body-point-count veto tested November14 demonstrates why the synthetic edge control matters: it can preserve broad wall/car alerts while removing a real partial-body overlap. That candidate must fail this suite even if the total announcement count is lower.
