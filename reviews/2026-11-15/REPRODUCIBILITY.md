# Included evidence and reproducibility limits

This publication is a review snapshot, not a self-contained installer or a newly scored benchmark. No tests were rerun merely to publish it. Results are dated historical measurements with explicit scope.

Included:

- Full textual deployed release-source files (`.py`, `.html`, plus two non-secret module/runtime-contract JSON files), preserved byte-for-byte and SHA-256 indexed.
- A consolidated handoff, review questions, selected structured measurements and source map.
- Relevant deterministic test/replay/acceptance tools, with local path assumptions retained. These cannot replay absent raw packs.
- Explicitly separated rejected experimental patches and their outcomes.
- Links to the four already-public original recording excerpts. Existing case ZIPs/previews are release assets at the repository root's documented URLs.

Not included:

- Credentials, authentication tokens, SSH keys/configuration, account-access links, private correspondence, unrelated personal/legal reports or configuration copied from the tablet.
- Third-party model weights, installers, generated application audio, all raw drive recordings, large caches or every historical experiment.
- The newer raw packs used for0.9.17 target/relevance checks. Their full metadata and hashes remain in the private development checkpoint/local archive. Aggregate evidence here does not let an external reviewer independently rerun those windows.

The code snapshot references omitted assets and dependencies. Some tools refer to the owner's original directory layout. Do not infer successful execution on another machine from this repository. A portable replay package would need an explicit selected-input export, model/license review, dependency environment and path adaptation, with original hashes retained.

`source/` is the deployed-source snapshot. `evaluation/` contains offline review tools, not live code. `rejected-experiments/` must not be installed. No new license grant for third-party material is asserted; preserve upstream notices and check the relevant dependencies/model terms before redistribution or use.

For claims of measured performance, prefer EVIDENCE.json and the scope statements in HANDOFF.md over repository commit dates, file counts or test totals. A software snapshot does not establish physical sensor timing, accurate calibration, real speaker delivery, nighttime recognition range or safety certification.
