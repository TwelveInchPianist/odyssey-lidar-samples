# Hardware, objective and measured limitations

## Fixed configuration
- Used original RoboSense RS-LiDAR-M1, reportedly manufactured in 2022; **not M1 Plus**. Original M1 protocol, 10 Hz, strongest single return mode4. Nominal average angular spacing0.2 degrees; no ROI/GAZE enablement established.
- Recorded firmware bytes: PL `00 30 23 07 00`, PS `00 30 20 10 21`. These raw fields are not decoded release dates. Internal device clock; no demonstrated physical hardware synchronization.
- Mounted directly on the exterior roof skin of a 2019 Honda Odyssey, near the driver and rear of the sunroof. Recessed mounting and near-field blind spots are accepted. Approximate physical lidar optical height1.7 m; estimated configuration, not a surveyed extrinsic calibration.
- Comma3 x narrow and wide native cameras, encoded HEVC at nominal20 fps, vehicle telemetry and IMU. Factory radar-derived tracks are available; this is not raw imaging-radar data.
- Current road compute: powered Lenovo Legion Go Z1 Extreme,16 GB RAM, Windows11. Offline: Apple-silicon Mac Mini16 GB plus Ryzen7900 X desktop64 GB. A MacBook is contemplated, not a demonstrated deployment platform.

## Intended behavior
Spoken notice of people, cyclists, animals and credible unknown objects on the road or nearby shoulders/sidewalks, regardless of a predicted collision path. Stationary and parallel-moving people matter. Isolated stopped vehicles and substantial branches/vegetation obstructing road or usable shoulder can matter; routine traffic and ordinary roadside vegetation should generally stay quiet. Class is stated only when supported; cautious unknown-object notices are acceptable.

The initial experimental nighttime target is no more than3 unhelpful advisories per10 evaluatedminutes, with miss evidence reported alongside. This is an evaluation target, not a global quota that mutes subsequent alerts. No minimum headlight-distance cutoff, no speed rating or collision-avoidance guarantee.

Ordinary recorded drives are available. A private test road, surrogate targets, crashes or deliberately hazardous encounters cannot be prerequisites. Neural training is not performed in the vehicle. Maps may provide evidence about particular structures but cannot hide people near them.

## Pipeline and evidence
Original packet decoding feeds geometric ground/cluster processing and tracking. A separate presence policy evaluates measured observations independently of legacy collision/path gates. Native camera inference currently uses pretrained YOLO11 m, temporal confirmation and visual identity correspondence. Camera-only and lidar-only observations can each propose a notice. The public cases distinguish these diagnostic branches; they are not a completed integrated sensor-fusion evaluation.

The frozen313 segment camera candidate pass covered13 routes, approximately306 minutes, both cameras2 Hz:73,476 inferenceimages and90 virtualadvisories. Those are proposal counts, not90 distinct or useful encounters. Only selected proposals and misses have been reviewed. Separate detailed cyclist/night-animal pilots ran5 Hz per camera. Delivered speech in these offline tests is simulated using the existing WAV durations, not proof of live audio.

Measured failures: repeated cyclist descriptions across components/cameras; fragmented nighttime animal tracks; high-scoring palm-as-animal classification; excessive generic lidar notices. A motion-only association change improved one animal case but merged distinct actors in negative tests and was rejected. A larger YOLO11 x did not solve that animal case; near-native input improved individual scores but not the final advisory.

Full nuisance/recall, nighttime long-range animal performance, precise optical/physical timing alignment and complete powered-device runtime remain unqualified. We seek an independent critique, not endorsement of a preselected occupancy or neural architecture.
