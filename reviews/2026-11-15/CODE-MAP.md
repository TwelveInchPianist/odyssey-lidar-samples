# Source map

The files in `source/` are a text-only snapshot of the0.9.17 release-source tree. They are not a new deployment. `SOURCE-MANIFEST.json` hashes each included file against the local release source. Model weights, generated audio, installers, recordings and local configuration are omitted.

| Area | Start here | Questions to inspect |
|---|---|---|
| Main lifecycle | [runtime.py](source/runtime.py), [app.py](source/app.py), [app_lifecycle.py](source/app_lifecycle.py) | Capture readiness, shutdown/wake, configuration/mode separation, logging and bounded background work |
| Sensor decode and timing | [m1_protocol.py](source/m1_protocol.py), [frames.py](source/frames.py), [telemetry.py](source/telemetry.py), [telemetry_freshness.py](source/telemetry_freshness.py), [runtime_clock.py](source/runtime_clock.py) | Raw validity, original timestamps, clock domains, age at consumption |
| Geometry and tracker | [fcw_engine.py](source/fcw_engine.py) | `Ground.fit`, `clusters`, `Tracker`, `threat`, `Engine.update`; sensor transforms, current-point ownership, motion trust and alert authority |
| Road support | [road_surface.py](source/road_surface.py), [ground_recovery.py](source/ground_recovery.py) | Local connected surfaces versus global fit; temporal support and fresh-reference recovery |
| Long-range objects | [far_objects.py](source/far_objects.py) | Accumulation, ownership, sparse support, current versus historical measurements |
| Motion evidence | [motion_trust.py](source/motion_trust.py), [near_edge_motion.py](source/near_edge_motion.py), [longitudinal_evidence.py](source/longitudinal_evidence.py) | Identity inheritance, independent corroboration, unknown versus stationary/moving |
| Path and occupancy | [forced_experiment.py](source/forced_experiment.py), [corridor_occupancy.py](source/corridor_occupancy.py), [corridor.py](source/corridor.py), [map_live.py](source/map_live.py) | What path is authoritative, where it extends, physical body versus padded corridor, provisional calibration |
| FCW speech selection | [warning_arbitration.py](source/warning_arbitration.py), [warning_surface.py](source/warning_surface.py), [lead_gate.py](source/lead_gate.py), [audio_output.py](source/audio_output.py), [audible_review.py](source/audible_review.py) | Measured braking, urgency, distinct hazards, identity churn, submitted versus physically heard audio |
| Inference process | [inference_worker.py](source/inference_worker.py), [inference_worker_selftest.py](source/inference_worker_selftest.py) | Owned child lifecycle, IPC, fresh result deadline versus liveness timeout, late reply cleanup |
| Camera/GPU | [video_processing.py](source/video_processing.py), [awareness_onnx.py](source/awareness_onnx.py), [camera_health.py](source/camera_health.py), [camera_recovery.py](source/camera_recovery.py) | FP32 DirectML, image age, nonblocking optional work, recovery and fallback |
| Identity/awareness | [awareness_vision.py](source/awareness_vision.py), [visual_identity.py](source/visual_identity.py), [awareness.py](source/awareness.py), [awareness_scenes.py](source/awareness_scenes.py), [night_companion.py](source/night_companion.py), [night_awareness.py](source/night_awareness.py), [night_light.py](source/night_light.py) | Fragmentation, shared identity, class evidence, light-state provenance, per-mode activation |
| Fusion | [object_fusion.py](source/object_fusion.py), [radar_fusion.py](source/radar_fusion.py), [empirical_fusion.py](source/empirical_fusion.py), [fusion_release.py](source/fusion_release.py) | Which evidence is optional, which associations have authority, scope of qualification |
| Replay integrity | [learning/replay_checkpoint.py](source/learning/replay_checkpoint.py) | Explicit state codec, source fingerprint, causal input handling, reset/restore parity |
| Calibration | [selfcal.py](source/selfcal.py), [selfcal_alignment.py](source/selfcal_alignment.py), [empirical_calibration.py](source/empirical_calibration.py), [commissioning.py](source/commissioning.py) | Physical evidence versus experimental bypass; bounded parked/naturalistic calibration |

Read each file's local comments as historical explanations to verify, not independent proof. Some legacy branches/flags remain. Presence of a module is not evidence that it was enabled in a particular recording.

Review tools and tests are in `evaluation/`. They retain local path/layout expectations; consult [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Rejected candidate changes are provided separately in `rejected-experiments/`, never in the deployed `source/` snapshot.
