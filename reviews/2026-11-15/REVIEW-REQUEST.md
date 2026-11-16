# Review request for Gemini / Hydrogen

Read [HANDOFF.md](HANDOFF.md), [CODE-MAP.md](CODE-MAP.md), [EVIDENCE.json](EVIDENCE.json), and the linked source. This is a custom experimental lidar/camera audible-warning project, not an Autoware deployment. The owner wants a step change in useful nighttime perception while preserving geometric wall/parked-car warnings.

Please deliver:

1. **A blunt feasibility assessment.** Separate object presence, tracking, classification, relevance and collision warning. What can reasonably be expected at70–100m at night from this M1/camera setup, and what is currently unsupported?
2. **A source-grounded architecture review.** Cite file/function names. Identify the highest-risk assumptions, accidental couplings, inadequate tests and any result claims stronger than the evidence. Do not assume more code or passing unit tests means better perception.
3. **A comparison of substantial alternatives.** Evaluate sensor-specific learned3D detection, temporal occupancy/segmentation, camera-led semantics with independent lidar support, and thermal-plus-lidar if appropriate. Compare required data, missing information, target runtime and failure modes. A generic call to use a larger network is insufficient.
4. **Three ranked experiments**, each with exact inputs, a reusable implementation/checkpoint if available, labels/calibration required, expected compute dependencies, success metrics and an explicit stopping rule. Distinguish feasible now from blocked on data/access/hardware.
5. **A concrete next milestone.** Define a useful improvement that could be demonstrated before changing the roadgoing stack. Explain how to measure misses and nuisance speech without converting owner preference labels into object truth or using future motion as an online oracle.

Specific questions:

- Should near-ground ambiguity disable all geometric FCW, or can a narrower defensible capability survive? Specify evidence required; do not simply remove readiness gates.
- What changes would prevent the first infrastructure notice while retaining a wall, parked car, edge-clipping object and shoulder VRU? Two broad positive examples are insufficient.
- Does the short-lived surface-repeat association risk suppressing a distinct nearby object? What stronger evidence is affordable?
- Is adapting a pretrained3D detector to the actual M1 the best next step, or does insufficient semantic information at range make it a poor bet? The SenseBike checkpoint has NOT been obtained or run; do not invent its performance.
- How should physical synchronization, camera distortion and extrinsics be verified with existing naturalistic data and a feasible parked capture? Distinguish identifiable parameters from assumptions.
- Which parts can realistically execute on Apple silicon/AMD Windows, and which require CUDA/custom operators or another machine? Do not equate abundant RAM with compatibility.
- Is the observed elapsed-time/CPU-time disparity enough to prioritize scheduling, memory/I/O or driver investigation? The Normal-priority experiment did not demonstrate fewer deadline failures.
- What is the smallest valuable annotation effort:3D boxes, per-point classes, trajectories, free-space boundaries, scene roles or notice preferences? Explain what each would train or validate.

Use an evidence table with columns **claim / supporting evidence / missing evidence / confidence / decisive test**. Preserve uncertainty and disagreements. Distinguish source inspection, measured results, published research, personal recommendation and speculation. State when you cannot inspect a linked file or when a model/backend assumption is unverified.

Do not contact anyone, publish replies, purchase hardware, upload additional recordings, change installed software or recommend unsafe staged road tests. This request is for independent technical review.
